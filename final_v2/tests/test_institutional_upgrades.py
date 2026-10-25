"""Unit tests for the final_v2 institutional upgrades (Phases 1-6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pod_platform.accounting.evaluation import (
    CONSERVATIVE_BORROW_BPS,
    evaluate_cost_borrow_sensitivity,
    evaluate_monthly_books,
)
from pod_platform.reporting.tables import RequiredTable
from pod_platform.risk.overlay import protect
from pod_platform.signals.ported import scale_scores_broad_native
from pod_platform.signals.timesfm_daily import (
    prepare_daily_excess_inputs,
    summarize_trajectory_forecasts,
)


def test_scale_scores_broad_native_ecdf():
    rm = pd.Period("2023-01", "M")
    broad = pd.DataFrame(
        {
            "return_month": [rm] * 5,
            "permno": [1, 2, 3, 4, 5],
            "score": [-2.0, -1.0, 0.0, 1.0, 2.0],
        }
    )
    native = broad.iloc[1:4].copy()
    out = scale_scores_broad_native(broad, native)
    assert len(out) == 5
    # Broad midpoint (permno=3) has broad_score 0.0; native rank(pct=True) is 2/3 -> (2/3 - 0.5)*sqrt(12)
    assert abs(float(out.loc[out["permno"] == 3, "broad_score"].iloc[0])) < 1e-9
    assert abs(float(out.loc[out["permno"] == 3, "native_score"].iloc[0]) - (2.0 / 3.0 - 0.5) * np.sqrt(12.0)) < 1e-9
    # Broad stocks outside native range get bounded ECDF ranks against native
    assert float(out.loc[out["permno"] == 1, "broad_score"].iloc[0]) < float(
        out.loc[out["permno"] == 2, "broad_score"].iloc[0]
    )
    assert float(out.loc[out["permno"] == 5, "broad_score"].iloc[0]) > float(
        out.loc[out["permno"] == 4, "broad_score"].iloc[0]
    )


def test_evaluate_monthly_books_and_section7_borrow_sensitivity():
    m1 = pd.Period("2023-01", "M")
    m2 = pd.Period("2023-02", "M")
    books = {
        m1: {101: 0.50, 202: -0.50},
        m2: {101: 0.50, 202: -0.50},
    }
    universe = pd.DataFrame(
        {
            "return_month": [m1, m1, m2, m2],
            "permno": [101, 202, 101, 202],
            "prc": [50.0, 40.0, 52.0, 38.0],
            "exit_price": [50.0, 40.0, 52.0, 38.0],
            "exit_price_date": [pd.Timestamp("2023-01-31")] * 4,
            "size_grp": ["large", "small", "large", "small"],
            "sector": ["45", "45", "45", "45"],
            "dolvol_126d": [1e8, 1e8, 1e8, 1e8],
            "me": [5000.0, 1500.0, 5200.0, 1400.0],
            "ret_exc_lead1m": [0.04, -0.02, 0.03, -0.01],
            "beta_60m": [1.0, 1.0, 1.0, 1.0],
            "be_me": [0.5, 0.5, 0.5, 0.5],
            "ret_12_1": [0.1, -0.1, 0.1, -0.1],
        }
    )
    mkt = pd.DataFrame(
        {"rf": [0.003, 0.003], "spx": [0.02, -0.01]},
        index=pd.PeriodIndex([m1, m2], freq="M"),
    )
    eval_df = evaluate_monthly_books(books, universe, mkt)
    assert len(eval_df) == 2
    assert (eval_df["pnl"] > 0).all()
    sens = evaluate_cost_borrow_sensitivity(books, universe, mkt)
    assert "tiered_baseline" in sens
    assert "section7_conservative_flat_borrow" in sens
    # Small-cap short borrow in Section 7 conservative flat schedule is 500bp vs 300bp tiered
    assert (
        sens["section7_conservative_flat_borrow"]["annual_borrow_cost_pct"]
        > sens["tiered_baseline"]["annual_borrow_cost_pct"]
    )
    assert CONSERVATIVE_BORROW_BPS["large"] == 50.0
    assert CONSERVATIVE_BORROW_BPS["small"] == 500.0


def test_required_table_21_rows():
    t = RequiredTable(
        ending_value=1_500_000_000.0,
        starting_value=1_000_000_000.0,
        elapsed_years=2.0,
        periods_per_year=12,
        portfolio_returns=(0.02, 0.01) * 12,
        benchmark_returns=(0.005, 0.005) * 12,
        risk_free_returns=(0.002, 0.002) * 12,
        gross_returns=(0.021, 0.011) * 12,
        cost_returns=(0.0005,) * 24,
        borrow_returns=(0.0005,) * 24,
        position_counts=(200,) * 24,
        gross_exposures=(1.0,) * 24,
        net_exposures=(0.0,) * 24,
        turnover_per_rebalance=(0.30,) * 24,
        spx_returns=(0.01, -0.005) * 12,
        max_one_day_loss=-0.014,
        max_one_day_loss_date="2022-05-13",
        beta_daily=-0.05,
        beta_daily_se=0.01,
        beta_daily_t_hac=-5.0,
    ).as_table()
    rows = t["section_9_1_21_rows"]
    assert len(rows) == 22
    assert "01_ending_value_usd" in rows
    assert "09_beta_vs_sp500" in rows
    assert rows["13_max_1m_loss_and_max_1d_loss"]["max_1d_loss_strategy"] == -0.014


def test_overlay_protect_re_centers_net_exposure():
    month = pd.Period("2024-06", "M")
    form = month - 1
    weights = {"10": 0.35, "11": 0.35, "20": -0.25, "21": -0.25}
    panel = pd.DataFrame(
        {
            "permno": ["10", "11", "20", "21"],
            "eom": [form] * 4,
            "gics": ["4510", "4510", "4510", "4510"],
            "size_grp": ["large"] * 4,
            "ret_exc": [0.01, 0.01, 0.80, 0.01],  # short '20' rallied +80% -> capped by RUNUP_CAP
            "rmax1_21d": [0.05] * 4,
        }
    )
    scores = pd.DataFrame(
        {"return_month": [form], "permno": ["10"], "final_score": [0.8], "prc": [20.0]}
    )
    res = protect(weights, month, panel, scores, max_abs_net=0.10)
    net_after = sum(res.weights.values())
    assert abs(net_after) <= 0.10 + 1e-9
    assert res.summary["post_overlay_net_capped"] is True


def test_timesfm_daily_preparation_and_summary():
    days = pd.bdate_range("2023-01-02", periods=80)
    dsf = pd.DataFrame(
        {
            "permno": [1001] * 80,
            "day": days,
            "daily_return": np.linspace(-0.01, 0.01, 80),
        }
    )
    spx = pd.Series(0.0005, index=days)
    permnos, inputs = prepare_daily_excess_inputs(
        dsf, spx, [1001], days[-1], min_context=63
    )
    assert permnos == [1001]
    assert len(inputs) == 1
    assert inputs[0].shape == (80,)

    pf = np.full((1, 21), 0.001, dtype=np.float32)
    qf = np.zeros((1, 21, 10), dtype=np.float32)
    qf[:, :, 1] = -0.002
    qf[:, :, -1] = 0.004
    summary = summarize_trajectory_forecasts(
        permnos,
        days[-1],
        days[-1] + pd.offsets.MonthEnd(1),
        pf,
        qf,
        [80],
    )
    assert len(summary) == 1
    assert float(summary["tfm_daily_cum21"].iloc[0]) > 0.0
    assert float(summary["tfm_daily_q_spread"].iloc[0]) > 0.0
