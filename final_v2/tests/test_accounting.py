"""Part 3 acceptance: hand-checkable financial fixtures (spec 26.4, 19.2-19.7)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from pod_platform.accounting import (
    AccountingLedger,
    BorrowSchedule,
    ReturnInputs,
    active_returns,
    annualized_arithmetic,
    annualized_turnover,
    annualized_volatility,
    benchmark_daily,
    benchmark_monthly,
    cagr,
    information_ratio,
    irregular_turnover,
    max_drawdown,
    one_way_turnover,
    portfolio_return,
    sharpe_ratio,
    summarize_periods,
)
from pod_platform.contracts.orders import OrderSide

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
D = Decimal


def _when(**kw) -> dt.datetime:
    return T0 + dt.timedelta(**kw)


def _schedule(**over) -> BorrowSchedule:
    base = dict(
        schedule_id="borrow_v1",
        large_cap_threshold_usd=D("10000000000"),
        large_cap_annual_rate=D("0.005"),
        other_annual_rate=D("0.05"),
        day_count="trading_252",
    )
    base.update(over)
    return BorrowSchedule(**base)


# ------------------------------------------------- 26.4 cash earns cash return


def test_all_cash_earns_only_cash_return():
    ledger = AccountingLedger(starting_cash=D("1000000"))
    ledger.accrue_cash(annual_rate=D("0.024"), periods_per_year=12, at=T0)
    assert ledger.cash_income == D("1000000") * D("0.024") / D("12")
    assert ledger.nav() == D("1000000") + ledger.cash_income
    ledger.assert_nav_identity()


# ------------------------------------- 26.4 one long reconciles stock + cash


def test_one_long_no_leverage_reconciles():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute(
        event_id="e1",
        security_id="AAA",
        side=OrderSide.BUY,
        quantity=D("100"),
        price=D("50"),
        fees=D("1"),
        executed_at=T0,
    )
    assert ledger.cash == D("10000") - D("5000") - D("1")
    ledger.mark("AAA", D("55"), _when(days=1))
    stock_contribution = D("100") * (D("55") - D("50"))
    assert stock_contribution == D("500")
    assert ledger.nav() == D("4999") + D("5500")
    assert ledger.nav() - D("10000") == stock_contribution - D("1")
    ledger.assert_nav_identity()


# ------------------------- 26.4 dollar-neutral book: long, short, collateral


def test_dollar_neutral_book_economics():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute(
        "e1", "AAA", OrderSide.BUY, D("100"), D("50"), D("0"), T0
    )
    ledger.execute(
        "e2", "BBB", OrderSide.SHORT, D("100"), D("50"), D("0"), T0
    )
    ledger.mark("AAA", D("50"), T0)
    ledger.mark("BBB", D("50"), T0)
    assert ledger.cash == D("10000")
    assert ledger.net_exposure() == D("0")
    assert ledger.gross_exposure() == D("10000")
    ledger.mark("AAA", D("55"), _when(days=1))
    ledger.mark("BBB", D("45"), _when(days=1))
    assert ledger.nav() == D("11000")
    ledger.assert_nav_identity()


def test_stock_rally_loses_on_short():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "BBB", OrderSide.SHORT, D("100"), D("50"), D("0"), T0)
    ledger.mark("BBB", D("50"), T0)
    assert ledger.cash == D("15000")
    assert ledger.nav() == D("10000")
    ledger.mark("BBB", D("60"), _when(days=1))
    assert ledger.nav() == D("9000")
    ledger.assert_nav_identity()


def test_short_dividend_liability_consistent_with_nav():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "BBB", OrderSide.SHORT, D("100"), D("50"), D("0"), T0)
    ledger.mark("BBB", D("50"), _when(days=1))
    ledger.charge_funding(D("2"), at=_when(days=1))
    assert ledger.short_liabilities() == D("5000")
    assert ledger.nav() == D("9998")


# --------------------------------------------------- 26.4 borrow on notional


def test_short_borrow_accrues_on_actual_notional():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "BBB", OrderSide.SHORT, D("100"), D("50"), D("0"), T0)
    ledger.mark("BBB", D("50"), _when(days=1))
    fee = ledger.accrue_borrow(
        _schedule(), {"BBB": D("500000000")}, days=252, at=_when(days=1)
    )
    assert fee == D("5000") * D("0.05")
    ledger.mark("BBB", D("100"), _when(days=2))
    fee2 = ledger.accrue_borrow(
        _schedule(), {"BBB": D("500000000")}, days=252, at=_when(days=2)
    )
    assert fee2 == D("10000") * D("0.05")
    assert ledger.borrow_costs == fee + fee2


def test_large_cap_borrow_schedule():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "AAA", OrderSide.SHORT, D("100"), D("50"), D("0"), T0)
    ledger.mark("AAA", D("50"), _when(days=1))
    fee = ledger.accrue_borrow(
        _schedule(), {"AAA": D("20000000000")}, days=252, at=_when(days=1)
    )
    assert fee == D("5000") * D("0.005")


def test_unmarked_short_liability_is_not_deleted():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "BBB", OrderSide.SHORT, D("100"), D("50"), D("0"), T0)
    with pytest.raises(ValueError):
        ledger.position("BBB").short_notional()
    ledger.mark("BBB", D("50"), _when(days=1))
    assert ledger.position("BBB").short_notional() == D("5000")
    assert ledger.short_liabilities() == D("5000")


# --------------------------------------- 26.4 entry/resize/exit/reversal costs


def test_entry_resize_exit_reversal_costs_charged():
    ledger = AccountingLedger(starting_cash=D("100000"))
    ledger.execute("e1", "AAA", OrderSide.BUY, D("100"), D("50"), D("5"), T0)
    ledger.execute("e2", "AAA", OrderSide.BUY, D("50"), D("52"), D("3"), _when(days=1))
    ledger.execute("e3", "AAA", OrderSide.SELL, D("150"), D("54"), D("7"), _when(days=2))
    ledger.execute("e4", "AAA", OrderSide.SHORT, D("40"), D("54"), D("2"), _when(days=3))
    ledger.mark("AAA", D("54"), _when(days=3))
    assert ledger.trading_costs == D("17")
    assert ledger.position("AAA").quantity == D("-40")
    assert ledger.cash == D("102643")
    assert ledger.nav() == D("100483")
    ledger.assert_nav_identity()


def test_missing_return_does_not_delete_liability():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "BBB", OrderSide.SHORT, D("100"), D("50"), D("0"), T0)
    ledger.mark("BBB", D("50"), _when(days=1))
    nav_before = ledger.nav()
    ledger.mark("BBB", D("50"), _when(days=2))
    assert ledger.nav() == nav_before
    assert ledger.short_liabilities() == D("5000")


# ------------------------------------------------------ 26.4 corporate action


def test_split_conserves_economic_value():
    ledger = AccountingLedger(starting_cash=D("10000"))
    ledger.execute("e1", "AAA", OrderSide.BUY, D("100"), D("50"), D("0"), T0)
    ledger.mark("AAA", D("60"), _when(days=1))
    nav_before = ledger.nav()
    ledger.apply_split("split1", "AAA", numerator=2, denominator=1, at=_when(days=2))
    assert ledger.position("AAA").quantity == D("200")
    assert ledger.position("AAA").last_mark == D("30")
    assert ledger.nav() == nav_before
    ledger.assert_nav_identity()


def test_nav_identity_maintained_across_operations():
    ledger = AccountingLedger(starting_cash=D("250000"))
    ledger.execute("e1", "AAA", OrderSide.BUY, D("1000"), D("100"), D("10"), T0)
    ledger.execute("e2", "BBB", OrderSide.SHORT, D("500"), D("40"), D("4"), T0)
    ledger.mark("AAA", D("110"), _when(days=1))
    ledger.mark("BBB", D("35"), _when(days=1))
    ledger.accrue_cash(D("0.02"), 12, _when(days=1))
    ledger.accrue_borrow(_schedule(), {"BBB": D("800000000")}, 21, _when(days=1))
    ledger.charge_funding(D("1.5"), at=_when(days=1))
    ledger.assert_nav_identity()


# ------------------------------------------------------------ 19.2 identity


def test_portfolio_return_identity():
    r = portfolio_return(
        ReturnInputs(
            risk_free=0.001,
            weights=(("AAA", 0.5), ("BBB", -0.3)),
            returns=(("AAA", 0.03), ("BBB", -0.02)),
            trading_cost=0.0005,
            borrow_cost=0.0002,
            funding_cost=0.0001,
        )
    )
    expected = 0.001 + 0.5 * (0.03 - 0.001) + (-0.3) * (-0.02 - 0.001) - 0.0005 - 0.0002 - 0.0001
    assert abs(r - expected) < 1e-12


def test_portfolio_return_requires_declared_returns():
    with pytest.raises(ValueError):
        portfolio_return(
            ReturnInputs(risk_free=0.0, weights=(("AAA", 1.0),), returns=())
        )


# ------------------------------------------------------------------ 19.5


def test_one_way_turnover_identity():
    pre = (("AAA", 0.5), ("BBB", -0.5))
    post = (("AAA", 0.4), ("CCC", -0.4))
    tau = one_way_turnover(pre, post)
    assert abs(tau - 0.5 * (0.1 + 0.5 + 0.4)) < 1e-12


def test_turnover_annualization_variants():
    assert annualized_turnover(0.25, 12) == 3.0
    assert irregular_turnover(2.4, 2.0) == 1.2
    with pytest.raises(ValueError):
        annualized_turnover(0.1, 0)
    with pytest.raises(ValueError):
        irregular_turnover(1.0, 0)


# ------------------------------------------------------------------ 19.6


def test_benchmark_conventions():
    assert abs(benchmark_monthly(2.4) - (2.4 / 1200 + 0.04 / 12)) < 1e-15
    assert abs(benchmark_daily(2.4) - (2.4 / 25200 + 0.04 / 252)) < 1e-15


# ------------------------------------------------------------------ 19.7


def test_information_ratio_and_sharpe():
    active = [0.01, -0.005, 0.008, 0.002]
    ir = information_ratio(active, 12)
    mean = sum(active) / 4
    var = sum((x - mean) ** 2 for x in active) / 3
    assert abs(ir - (12 ** 0.5) * mean / var**0.5) < 1e-12
    excess = [0.012, -0.003, 0.009, 0.001]
    sh = sharpe_ratio(excess, 12)
    assert sh > 0
    with pytest.raises(ValueError):
        information_ratio([0.01], 12)
    with pytest.raises(ValueError):
        information_ratio([0.01, 0.01], 12)


def test_annualization_and_cagr():
    rets = [0.01, -0.005, 0.004]
    assert abs(annualized_arithmetic(rets, 12) - 12 * (sum(rets) / 3)) < 1e-12
    assert annualized_volatility(rets, 12) > 0
    assert abs(cagr(1_000_000.0, 2_000_000.0, 1.0) - 1.0) < 1e-12
    assert 0.10 < cagr(1_000_000.0, 1_640_000.0, 5.0) < 0.11
    with pytest.raises(ValueError):
        cagr(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        cagr(1.0, 2.0, 0.0)


def test_max_drawdown():
    assert abs(max_drawdown([100.0, 110.0, 99.0, 120.0]) - (99.0 / 110.0 - 1.0)) < 1e-12
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_summarize_periods_shape():
    portfolio = [0.011, -0.004, 0.009, 0.003]
    benchmark = [0.004, 0.004, 0.004, 0.004]
    risk_free = [0.003, 0.003, 0.003, 0.003]
    stats = summarize_periods(portfolio, benchmark, risk_free, 12)
    assert stats.information_ratio is not None
    assert stats.sharpe is not None
    assert stats.annual_volatility > 0
    assert stats.periods_per_year == 12
    with pytest.raises(ValueError):
        summarize_periods(portfolio, benchmark, risk_free[:2], 12)
