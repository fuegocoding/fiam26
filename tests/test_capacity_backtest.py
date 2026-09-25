import pandas as pd
import pytest

from src.capacity_backtest import simulate_capacity_path
from src.capacity_constraints import CapacityConfig


def _target(month, permno, weight, ret=0.0, size_grp="nano"):
    return {
        "target_month": pd.Period(month, freq="M"),
        "permno": permno,
        "target_weight": weight,
        "prc": 10.0,
        "dolvol_126d": 100.0,
        "ret_exc_lead1m": ret,
        "size_grp": size_grp,
    }


def test_capacity_backtest_converts_executable_dollars_to_pnl():
    targets = pd.DataFrame([_target("2024-01", 1, 1.0, ret=0.10)])
    config = CapacityConfig(
        nav=100.0,
        execution_days=1,
        liquidation_days=1,
        long_participation=0.10,
        min_long_price=0.0,
    )

    result = simulate_capacity_path(targets, config, transaction_cost_bp={})
    month = result["monthly"].iloc[0]
    holding = result["holdings"].iloc[0]

    assert holding["requested_dollars"] == pytest.approx(100.0)
    assert holding["final_dollars"] == pytest.approx(10.0)
    assert month["pnl_dollars"] == pytest.approx(1.0)
    assert month["gross_return"] == pytest.approx(0.01)
    assert month["net_return"] == pytest.approx(0.01)
    assert month["ending_nav"] == pytest.approx(101.0)


def test_short_borrow_fee_is_charged_on_executed_short_notional():
    targets = pd.DataFrame([_target("2024-01", 1, -1.0)])
    config = CapacityConfig(
        nav=100.0,
        execution_days=1,
        liquidation_days=1,
        short_participation=1.0,
        min_short_price=0.0,
    )

    result = simulate_capacity_path(
        targets,
        config,
        transaction_cost_bp={},
        fallback_borrow_fee_bps=1_200.0,
    )
    month = result["monthly"].iloc[0]

    assert month["short_notional"] == pytest.approx(100.0)
    assert month["borrow_cost_dollars"] == pytest.approx(1.0)
    assert month["net_return"] == pytest.approx(-0.01)
    assert month["ending_nav"] == pytest.approx(99.0)


def test_name_missing_from_next_target_is_explicitly_exited():
    targets = pd.DataFrame(
        [
            _target("2024-01", 1, 0.10),
            _target("2024-02", 2, 0.00, size_grp="mega"),
        ]
    )
    config = CapacityConfig(
        nav=100.0,
        execution_days=1,
        liquidation_days=1,
        long_participation=1.0,
        min_long_price=0.0,
    )

    result = simulate_capacity_path(targets, config, transaction_cost_bp={})
    february = result["holdings"].query(
        "target_month == @pd.Period('2024-02', freq='M') and permno == 1"
    ).iloc[0]

    assert february["requested_dollars"] == 0
    assert february["executed_trade_dollars"] == pytest.approx(-10.0)
    assert february["final_dollars"] == 0


def test_transaction_cost_uses_executed_trade_and_stock_tier():
    targets = pd.DataFrame([_target("2024-01", 1, 0.10)])
    config = CapacityConfig(
        nav=100.0,
        execution_days=1,
        liquidation_days=1,
        long_participation=1.0,
        min_long_price=0.0,
    )

    result = simulate_capacity_path(
        targets, config, transaction_cost_bp={"nano": 100.0}
    )
    month = result["monthly"].iloc[0]

    assert month["transaction_cost_dollars"] == pytest.approx(0.10)
    assert month["net_return"] == pytest.approx(-0.001)
