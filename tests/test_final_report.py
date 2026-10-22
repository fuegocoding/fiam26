"""Fast synthetic checks for final/evaluate.py, final/identity.py and final/report.py."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "final"))
import evaluate  # noqa: E402
import identity  # noqa: E402
import report  # noqa: E402


def _universe():
    months = pd.period_range("2021-01", "2021-03", freq="M")
    rows = [{"return_month": m, "permno": p, "ret_exc_lead1m": r, "size_grp": "large",
             "sector": "45", "beta_60m": 1.0, "me": 1000.0, "prc": 10.0,
             "exit_price": 10.0, "exit_price_date": (m-1).to_timestamp("M")}
            for m in months for p, r in ((1, 0.02), (2, -0.01))]
    return pd.DataFrame(rows)


def _market():
    m = pd.period_range("2021-01", "2021-03", freq="M")
    return pd.DataFrame({"rf": 0.001, "spx": 0.01}, index=m)


def test_evaluate_costs_drift_and_benchmark():
    books = {m: {1: 0.5, 2: -0.5} for m in pd.period_range("2021-01", "2021-03", freq="M")}
    r = evaluate.evaluate(books, _universe(), _market())
    first = r.iloc[0]
    # first month trades the whole book at the large-cap 10 bp rate
    assert np.isclose(first.traded, 1.0) and np.isclose(first.cost, 1.0 * 10 / 1e4)
    assert np.isclose(first.pnl, 0.5 * 0.02 + -0.5 * -0.01)
    assert np.isclose(first.borrow, 0.5 * 40 / 1e4 / 12)
    assert np.isclose(first.total, first.rf + first.pnl - first.cost - first.borrow)
    assert np.isclose(first.active, first.total - (first.rf + 0.04 / 12))
    # later months only rebalance the drift, so they trade far less than the book
    assert (r.traded.iloc[1:] < 0.1).all()


def test_missing_return_exits_at_last_price():
    u = _universe()
    u.loc[(u.permno == 1) & (u.return_month == pd.Period("2021-01", "M")), "ret_exc_lead1m"] = np.nan
    r = evaluate.evaluate({pd.Period("2021-01", "M"): {1: 0.5, 2: -0.5}}, u, _market())
    assert np.isclose(r.pnl.iloc[0], 0.5 * -0.001 - 0.5 * -0.01) and r.missing.iloc[0] == 1
    assert np.isclose(r.traded.iloc[0], 1.5)
    assert np.isclose(r.cost.iloc[0], 1.5 * 10 / 1e4)
    assert len(r.attrs["liquidations"]) == 1


def test_cover_symbol_reads_registered_securities_table():
    text = ("FORM 8-K ... Title of each class Trading Symbol(s) Name of each exchange on which "
            "registered Common Stock, par value $0.01 ACME The Nasdaq Stock Market LLC")
    assert identity.cover_symbol(text) == "ACME"
    assert identity.cover_symbol("no cover table here") is None


def test_holdings_schema_and_compliance():
    m = pd.period_range("2021-01", "2021-02", freq="M")
    p = pd.DataFrame([{"return_month": mm, "permno": i, "ticker": f"T{i}", "company_name": f"C{i}",
                       "target_weight": (0.005 if i < 100 else -0.005)}
                      for mm in m for i in range(200)])
    h = report.holdings_table(p)
    assert list(h.columns) == ["Date", "PERMNO", "TICKER", "COMPANY NAME", "WEIGHT"]
    assert h.Date.iloc[0] == "2021-01-01" and np.isclose(h.WEIGHT.abs().max(), 0.5)
    r = pd.DataFrame({"names": 200, "gross": 1.0, "net": 0.0}, index=m)
    checks = report.compliance(r, h, pd.DataFrame(columns=["return_month", "permno", "final_score"]))
    assert checks["names_between_200_and_500"] and checks["net_within_50pct"]
    assert checks["predictions_carry_no_outcomes"]
    assert not checks["covers_the_test_window"]  # only two months here
