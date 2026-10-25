"""Offline Arcana attribution lens, Section 6.3 survivorship/coverage bias audit,
and Section 9.4 required exhibits + submission CSV exporter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..accounting.evaluation import (
    daily_alpha_decay,
    earnings_window_decomposition,
    evaluate_cost_borrow_sensitivity,
    ols_hac,
    position_returns,
)
from .arcana import make_bundle as make_arcana_bundle
from .arcana import write_bundle as write_arcana_bundle
from .attribution import (
    PositionContribution,
    SizingPeriod,
    regression_factor_attribution,
    selection_vs_sizing,
)
from .attribution import (
    hit_rate_win_loss as platform_hit_rate_win_loss,
)

MONTHLY_HORIZONS = (1, 2, 3, 6, 12)
DELISTED_RETAIL_TICKERS = (
    "BBBY",
    "PRTY",
    "EXPR",
    "JOAN",
    "TUP",
    "BIG",
    "CONN",
    "LL",
    "RAD",
)


def build_panel_factors(universe: pd.DataFrame) -> pd.DataFrame:
    """Size (SMB), value (HML), and momentum (MOM) long-short factors from the panel."""
    p = universe.dropna(subset=["ret_exc_lead1m"])

    def _spread(col: str) -> pd.Series:
        r = p.groupby("return_month")[col].rank(pct=True)
        hi = p[r >= 0.7].groupby("return_month")["ret_exc_lead1m"].mean()
        lo = p[r <= 0.3].groupby("return_month")["ret_exc_lead1m"].mean()
        return hi - lo

    g = (
        p.groupby(["return_month", "size_grp"])["ret_exc_lead1m"]
        .mean()
        .unstack()
    )
    return pd.DataFrame(
        {
            "smb": g[["micro", "small"]].mean(axis=1)
            - g[["large", "mega"]].mean(axis=1),
            "hml": _spread("be_me"),
            "mom": _spread("ret_12_1"),
        }
    )


def factor_vs_idiosyncratic_split(
    r: pd.DataFrame, factor_returns: pd.DataFrame
) -> dict[str, object]:
    """Decomposes portfolio excess return into factor vs idiosyncratic components (Arcana lens 1)."""
    f = pd.DataFrame(
        {
            "y": r["total"] - r["rf"],
            "mkt": r["spx"] - r["rf"],
            "smb": factor_returns["smb"],
            "hml": factor_returns["hml"],
            "mom": factor_returns["mom"],
        }
    ).dropna()
    x = f[["mkt", "smb", "hml", "mom"]].to_numpy()
    y = f["y"].to_numpy()
    b, t_hac, r2 = ols_hac(y, x, lags=3)
    factor_part = x @ b[1:]
    idio_part = y - factor_part
    total_var = float(np.var(y, ddof=1))
    mean_val = float(y.mean()) if abs(y.mean()) > 1e-12 else 1e-12
    return {
        "months": int(len(f)),
        "factor_share_of_mean_return": float(factor_part.mean() / mean_val),
        "idiosyncratic_share_of_mean_return": float(
            idio_part.mean() / mean_val
        ),
        "factor_share_of_variance": float(
            np.var(factor_part, ddof=1) / total_var
        )
        if total_var > 0
        else 0.0,
        "idiosyncratic_share_of_variance": float(
            np.var(idio_part, ddof=1) / total_var
        )
        if total_var > 0
        else 0.0,
        "factor_contribution_ann": float(factor_part.mean() * 12.0),
        "idiosyncratic_contribution_ann": float(idio_part.mean() * 12.0),
        "alpha_ann_from_decomposition": float(b[0] * 12.0),
        "alpha_t_hac": float(t_hac[0]),
        "r2_four_factor": float(r2),
        "loadings": {
            name: float(v)
            for name, v in zip(["mkt", "smb", "hml", "mom"], b[1:], strict=True)
        },
        "loading_t_hac": {
            name: float(v)
            for name, v in zip(
                ["mkt", "smb", "hml", "mom"], t_hac[1:], strict=True
            )
        },
    }


def sizing_vs_selection_split(positions_df: pd.DataFrame) -> dict[str, float]:
    """Decomposes P&L into stock selection (equal-weight within side) vs position sizing (Arcana lens 2)."""
    q = positions_df.copy()
    q["side_w"] = q["target_weight"].abs()
    q["equal_w"] = q.groupby(["return_month", "side"])["side_w"].transform(
        "mean"
    )
    q["equal_w"] = np.where(q["side"] == "short", -q["equal_w"], q["equal_w"])
    q["actual_pnl"] = q["target_weight"] * q["ret"]
    q["select_pnl"] = q["equal_w"] * q["ret"]
    q["size_pnl"] = q["actual_pnl"] - q["select_pnl"]
    monthly = q.groupby("return_month")[
        ["actual_pnl", "select_pnl", "size_pnl"]
    ].sum()
    total_sum = float(monthly["actual_pnl"].sum())
    denom = total_sum if abs(total_sum) > 1e-12 else 1e-12
    out: dict[str, float] = {}
    for col, name in (
        ("actual_pnl", "total"),
        ("select_pnl", "selection"),
        ("size_pnl", "sizing"),
    ):
        s = monthly[col]
        out[f"{name}_pnl_ann"] = float(s.mean() * 12.0)
        out[f"{name}_share_of_pnl"] = float(s.sum() / denom)
    out["selection_t"] = (
        float(
            monthly["select_pnl"].mean()
            / monthly["select_pnl"].std()
            * math.sqrt(len(monthly))
        )
        if monthly["select_pnl"].std() > 0
        else 0.0
    )
    out["sizing_t"] = (
        float(
            monthly["size_pnl"].mean()
            / monthly["size_pnl"].std()
            * math.sqrt(len(monthly))
        )
        if monthly["size_pnl"].std() > 0
        else 0.0
    )
    return out


def hit_rate_and_win_loss(
    positions_df: pd.DataFrame, r: pd.DataFrame
) -> dict[str, object]:
    """Position and monthly hit rates, win/loss payoff ratios, and long vs short leg attribution (Arcana lens 3)."""
    q = positions_df.copy()
    q["win"] = q["target_weight"] * q["ret"]
    pos = q.loc[q["win"] > 0, "win"]
    neg = q.loc[q["win"] < 0, "win"]
    long_q = q[q["target_weight"] > 0]
    short_q = q[q["target_weight"] < 0]
    active = r["active"]
    n_neg = int((active < 0).sum())
    return {
        "position_hit_rate": float((q["win"] > 0).mean()),
        "position_win_loss_ratio": float(pos.mean() / abs(neg.mean()))
        if len(neg)
        else 0.0,
        "long_position_hit_rate": float((long_q["win"] > 0).mean())
        if len(long_q)
        else 0.0,
        "short_position_hit_rate": float((short_q["win"] > 0).mean())
        if len(short_q)
        else 0.0,
        "long_leg_pnl_ann": float(r["pnl_long"].mean() * 12.0),
        "short_leg_pnl_ann": float(r["pnl_short"].mean() * 12.0),
        "month_hit_rate": float((active > 0).mean()),
        "month_win_loss_ratio": (
            float(active[active > 0].mean() / abs(active[active < 0].mean()))
            if n_neg
            else 0.0
        ),
        "winning_positions": int(len(pos)),
        "losing_positions": int(len(neg)),
    }


def monthly_alpha_decay(
    panel_path: Path, predictions: pd.DataFrame, score_col: str = "final_score"
) -> dict[str, float]:
    """Top-minus-bottom quintile excess return spread at 1, 2, 3, 6, 12 months post-formation."""
    ret = pd.read_parquet(panel_path, columns=["permno", "eom", "ret_exc"])
    ret["eom"] = pd.to_datetime(ret["eom"])
    ret = ret.dropna(subset=["ret_exc"]).drop_duplicates(["permno", "eom"])
    ret = ret.sort_values(["permno", "eom"]).reset_index(drop=True)
    cum = ret.groupby("permno")["ret_exc"].transform(
        lambda s: (1.0 + s).cumprod()
    )
    ret["cum"] = cum
    for h in MONTHLY_HORIZONS:
        ret[f"fwd_{h}"] = (
            ret.groupby("permno")["cum"].transform(
                lambda s, k=h: s.shift(-k) / s
            )
            - 1.0
        )
    ret["formation"] = ret["eom"].dt.to_period("M")
    d = (
        predictions[["return_month", "permno", score_col]]
        .rename(columns={score_col: "score"})
        .copy()
    )
    d["formation"] = pd.to_datetime(d["return_month"]).dt.to_period("M") - 1
    d = d.merge(
        ret.drop(columns=["eom", "cum"]), on=["permno", "formation"], how="left"
    )
    out: dict[str, float] = {}
    for h in MONTHLY_HORIZONS:
        col = f"fwd_{h}"
        g = d.dropna(subset=[col, "score"])
        if g.empty:
            out[f"quintile_spread_{h}m_months"] = 0.0
            continue
        q = g.groupby("return_month")["score"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
        )
        g = g.assign(quintile=q)
        spread = (
            (
                g[g["quintile"] == 4].groupby("return_month")[col].mean()
                - g[g["quintile"] == 0].groupby("return_month")[col].mean()
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        out[f"quintile_spread_{h}m_ann"] = float(spread.mean() * 12.0 / h)
        out[f"quintile_spread_{h}m_t"] = (
            float(spread.mean() / spread.std() * math.sqrt(len(spread)))
            if len(spread) > 1 and spread.std() > 0
            else 0.0
        )
        out[f"quintile_spread_{h}m_months"] = float(len(spread))
    return out


def stress_regime_attribution(r: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Evaluates strategy performance across distinct market and factor stress regimes."""
    regimes = {
        "2021_meme_squeeze_window": (
            pd.Period("2021-01", "M"),
            pd.Period("2021-06", "M"),
        ),
        "2022_rate_hike_bear_market": (
            pd.Period("2022-01", "M"),
            pd.Period("2022-12", "M"),
        ),
        "2023_svb_and_ai_narrow_rally": (
            pd.Period("2023-01", "M"),
            pd.Period("2023-12", "M"),
        ),
        "2024_momentum_expansion": (
            pd.Period("2024-01", "M"),
            pd.Period("2024-12", "M"),
        ),
        "2025_2026_late_cycle": (
            pd.Period("2025-01", "M"),
            pd.Period("2026-08", "M"),
        ),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (start_p, end_p) in regimes.items():
        sub = r.loc[(r.index >= start_p) & (r.index <= end_p)]
        if sub.empty:
            continue
        strat_cum = float((1.0 + sub["total"]).prod() - 1.0)
        bench_cum = float((1.0 + sub["benchmark"]).prod() - 1.0)
        spx_cum = float((1.0 + sub["spx"].fillna(0.0)).prod() - 1.0)
        act_ann = float(sub["active"].mean() * 12.0)
        ir_sub = (
            float(math.sqrt(12.0) * sub["active"].mean() / sub["active"].std())
            if len(sub) > 1 and sub["active"].std() > 0
            else 0.0
        )
        out[name] = {
            "months": float(len(sub)),
            "strategy_cumulative_return": strat_cum,
            "benchmark_cumulative_return": bench_cum,
            "sp500_cumulative_return": spx_cum,
            "active_return_ann": act_ann,
            "information_ratio": ir_sub,
            "long_leg_pnl_cum": float(sub["pnl_long"].sum()),
            "short_leg_pnl_cum": float(sub["pnl_short"].sum()),
        }
    return out


def delisted_and_coverage_bias_audit(
    positions_df: pd.DataFrame,
    universe: pd.DataFrame,
    liquidations_df: pd.DataFrame,
    demand_card_path: Path | None = None,
) -> dict[str, object]:
    """Final Guide Section 6.3 survivorship & coverage bias audit (delisted retailers + alt-data coverage)."""
    retail_delisted = positions_df[
        positions_df["ticker"]
        .astype(str)
        .str.upper()
        .isin(DELISTED_RETAIL_TICKERS)
    ]
    liq_count = len(liquidations_df) if liquidations_df is not None else 0
    liq_pnl = (
        float(
            (
                liquidations_df["opening_weight"]
                * liquidations_df["price_return"]
            ).sum()
        )
        if liquidations_df is not None and not liquidations_df.empty
        else 0.0
    )

    # Check if dropping all consumer retail/durables (GICS 2550, 2520) or delisted names changes P&L
    total_pnl = float(positions_df["pnl"].sum())
    ex_delisted_pnl = float(
        positions_df.loc[
            ~positions_df["ticker"]
            .astype(str)
            .str.upper()
            .isin(DELISTED_RETAIL_TICKERS),
            "pnl",
        ].sum()
    )

    coverage_audit: dict[str, object] = {}
    if demand_card_path is not None and demand_card_path.exists():
        card = pd.read_parquet(demand_card_path)
        covered_tickers = set(
            card["ticker"].dropna().astype(str).str.upper().unique()
        )
        pos_covered = (
            positions_df["ticker"].astype(str).str.upper().isin(covered_tickers)
        )
        coverage_audit = {
            "distinct_covered_tickers": int(len(covered_tickers)),
            "holdings_share_in_alt_data": float(pos_covered.mean()),
            "covered_holdings_avg_me_musd": float(
                positions_df.loc[pos_covered, "me"].mean()
            )
            if pos_covered.any()
            else 0.0,
            "uncovered_holdings_avg_me_musd": float(
                positions_df.loc[~pos_covered, "me"].mean()
            )
            if (~pos_covered).any()
            else 0.0,
            "covered_holdings_pnl_share": float(
                positions_df.loc[pos_covered, "pnl"].sum()
                / max(abs(total_pnl), 1e-12)
            ),
        }

    return {
        "delisted_retailers_tested": list(DELISTED_RETAIL_TICKERS),
        "delisted_retailer_holding_months": int(len(retail_delisted)),
        "delisted_retailer_pnl_contribution_pct_nav": float(
            retail_delisted["pnl"].sum() * 100.0
        ),
        "portfolio_cum_pnl_with_delisted_retailers": total_pnl,
        "portfolio_cum_pnl_ex_delisted_retailers": ex_delisted_pnl,
        "missing_return_last_quote_liquidations_count": int(liq_count),
        "missing_return_liquidations_pnl_impact": liq_pnl,
        "survivorship_handling_rule": "missing next-month returns liquidated at last available quote minus trading cost; never zero-filled",
        "alt_data_coverage_bias_audit": coverage_audit,
    }


def compute_holding_period(positions_df: pd.DataFrame) -> dict[str, float]:
    """Direct holding period in consecutive months and trading days (Final Guide Section 9.3)."""
    held = positions_df.sort_values(["permno", "return_month"]).copy()
    held["seq"] = held["return_month"].astype("int64")
    gap = held.groupby("permno")["seq"].diff()
    held["run"] = gap.ne(1).groupby(held["permno"]).cumsum()
    spans = (
        held.groupby(["permno", "run"])
        .agg(
            months=("seq", "size"),
            weight=("target_weight", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    w_sum = float(spans["weight"].sum())
    months = (
        float((spans["months"] * spans["weight"]).sum() / w_sum)
        if w_sum > 0
        else 1.0
    )
    return {
        "avg_holding_period_months": months,
        "avg_holding_period_trading_days": months * 21.0,
    }


def enrich_positions_with_returns(
    portfolio_df: pd.DataFrame, universe: pd.DataFrame, mkt: pd.DataFrame
) -> pd.DataFrame:
    """Attaches realized next-month excess return and P&L to each holding."""
    p = portfolio_df.copy()
    if not isinstance(p["return_month"].dtype, pd.PeriodDtype):
        p["return_month"] = pd.to_datetime(p["return_month"]).dt.to_period("M")
    p = p.drop(columns=["me"], errors="ignore")
    outcome_cols = [
        "return_month",
        "permno",
        "ret_exc_lead1m",
        "me",
        *[
            c
            for c in ("exit_price", "exit_price_date")
            if c in universe.columns
        ],
    ]
    p = p.merge(
        universe[outcome_cols], on=["return_month", "permno"], how="left"
    )
    rates = mkt["rf"]
    p["ret"] = p["ret_exc_lead1m"]
    for month, ids in p.groupby("return_month").groups.items():
        p.loc[ids, "ret"] = position_returns(
            p.loc[ids], float(rates.get(month, np.nan))
        )[0]
    p["pnl"] = p["target_weight"] * p["ret"]
    return p


def export_submission_and_exhibits(
    *,
    portfolio_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    universe: pd.DataFrame,
    mkt: pd.DataFrame,
    monthly_eval_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    daily_summary: dict[str, object],
    dsf: pd.DataFrame,
    external_path: Path,
    panel_path: Path,
    out_dir: Path,
    score_col: str = "final_score_liquid",
    demand_card_path: Path | None = None,
) -> dict[str, object]:
    """Writes holdings.csv, returns.csv, daily_returns.csv, exhibits_9_4.json, and arcana_audit.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = enrich_positions_with_returns(portfolio_df, universe, mkt)

    # 1. Required holdings.csv (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT in %)
    start_dates = p["return_month"].dt.to_timestamp(how="start")
    holdings_csv = pd.DataFrame(
        {
            "Date": start_dates.dt.strftime("%Y-%m-%d"),
            "PERMNO": p["permno"].astype(int),
            "TICKER": p["ticker"],
            "COMPANY NAME": p["company_name"],
            "WEIGHT": p["target_weight"] * 100.0,
        }
    ).sort_values(["Date", "WEIGHT", "PERMNO"], ascending=[True, False, True])
    holdings_csv.to_csv(out_dir / "holdings.csv", index=False)

    # 2. Required returns.csv (Date, RETURN)
    returns_csv = pd.DataFrame(
        {
            "Date": monthly_eval_df.index.to_timestamp(how="start").strftime(
                "%Y-%m-%d"
            ),
            "RETURN": monthly_eval_df["total"].to_numpy(),
        }
    )
    returns_csv.to_csv(out_dir / "returns.csv", index=False)

    # 3. Daily returns CSV
    if not daily_df.empty:
        daily_export = daily_df.copy()
        daily_export["Date"] = pd.to_datetime(daily_export["day"]).dt.strftime(
            "%Y-%m-%d"
        )
        daily_export[
            [
                "Date",
                "total",
                "gross_daily",
                "rf",
                "spx",
                "benchmark_daily",
                "active",
                "drawdown",
                "rolling_252d_beta",
            ]
        ].rename(columns={"total": "RETURN"}).to_csv(
            out_dir / "daily_returns.csv", index=False
        )

    # 4. Liquidations & monthly detail
    liq_df = monthly_eval_df.attrs.get("liquidations", pd.DataFrame())
    if liq_df is not None and not liq_df.empty:
        liq_df.to_csv(out_dir / "liquidations.csv", index=False)
    monthly_eval_df.to_csv(out_dir / "monthly_detail.csv")

    # 5. Section 9.4 Exhibit 5: Calendar-year return table (2021 through 2026 YTD)
    by_yr = monthly_eval_df.groupby(monthly_eval_df.index.year)
    cal_years = pd.DataFrame(
        {
            "strategy": by_yr["total"].apply(
                lambda s: float((1.0 + s).prod() - 1.0)
            ),
            "benchmark": by_yr["benchmark"].apply(
                lambda s: float((1.0 + s).prod() - 1.0)
            ),
            "sp500": by_yr["spx"].apply(
                lambda s: float((1.0 + s.fillna(0.0)).prod() - 1.0)
            ),
            "active": by_yr["active"].sum(),
            "months": by_yr.size(),
        }
    )
    cal_years.to_csv(out_dir / "calendar_years.csv")

    # 6. Section 9.4 Exhibit 7 & 8: Top 10 long/short holdings & Top 10 positive/negative P&L contributors
    n_months = len(monthly_eval_df)
    held_sorted = p.sort_values("return_month")
    last_labels = held_sorted.drop_duplicates("permno", keep="last").set_index(
        "permno"
    )[["ticker", "company_name"]]
    by_stock = p.groupby("permno")
    avg_weights = pd.DataFrame(
        {
            "avg_weight_pct": by_stock["target_weight"].sum()
            / max(n_months, 1)
            * 100.0,
            "months_held": by_stock.size(),
        }
    ).join(last_labels)
    top_longs = (
        avg_weights[avg_weights["avg_weight_pct"] > 0]
        .nlargest(10, "avg_weight_pct")
        .assign(side="long")
    )
    top_shorts = (
        avg_weights[avg_weights["avg_weight_pct"] < 0]
        .nsmallest(10, "avg_weight_pct")
        .assign(side="short")
    )
    top_hold_df = pd.concat([top_longs, top_shorts]).reset_index()
    top_hold_df.to_csv(out_dir / "top_average_holdings.csv", index=False)

    pnl_by_stock = (
        (by_stock["pnl"].sum() * 100.0)
        .rename("pnl_pct_nav")
        .to_frame()
        .join(last_labels)
    )
    contrib_df = pd.concat(
        [
            pnl_by_stock.nlargest(10, "pnl_pct_nav").assign(
                kind="top_positive"
            ),
            pnl_by_stock.nsmallest(10, "pnl_pct_nav").assign(
                kind="top_negative"
            ),
        ]
    ).reset_index()
    contrib_df.to_csv(out_dir / "contributors.csv", index=False)

    # 7. Position size vs 20-day/126-day ADV distribution (Long vs Short separately)
    adv_df = p[
        [
            "return_month",
            "permno",
            "side",
            "target_weight",
            "dolvol_126d",
            "capital",
        ]
    ].copy()
    adv_df["position_dollars"] = (
        adv_df["target_weight"].abs() * adv_df["capital"]
    )
    adv_df["adv_days"] = adv_df["position_dollars"] / adv_df[
        "dolvol_126d"
    ].replace(0.0, np.nan)
    adv_df = adv_df.dropna(subset=["adv_days"])
    adv_df.to_csv(out_dir / "position_adv_days.csv", index=False)

    long_adv = adv_df.loc[adv_df["side"] == "long", "adv_days"]
    short_adv = adv_df.loc[adv_df["side"] == "short", "adv_days"]
    adv_distribution = {
        "long": {
            "median_adv_days": float(long_adv.median())
            if len(long_adv)
            else 0.0,
            "p90_adv_days": float(long_adv.quantile(0.90))
            if len(long_adv)
            else 0.0,
            "max_adv_days": float(long_adv.max()) if len(long_adv) else 0.0,
        },
        "short": {
            "median_adv_days": float(short_adv.median())
            if len(short_adv)
            else 0.0,
            "p90_adv_days": float(short_adv.quantile(0.90))
            if len(short_adv)
            else 0.0,
            "max_adv_days": float(short_adv.max()) if len(short_adv) else 0.0,
        },
    }

    # 8. Build books dictionary for sensitivity & earnings window decomposition
    books = {
        month: dict(
            zip(
                g["permno"].astype(int),
                g["target_weight"].astype(float),
                strict=True,
            )
        )
        for month, g in p.groupby("return_month")
    }
    cost_sens = evaluate_cost_borrow_sensitivity(books, universe, mkt)
    earn_decomp = (
        earnings_window_decomposition(books, dsf, external_path)
        if not dsf.empty
        else {}
    )
    d_decay = (
        daily_alpha_decay(dsf, predictions_df, score_col=score_col)
        if not dsf.empty and score_col in predictions_df.columns
        else {}
    )
    m_decay = (
        monthly_alpha_decay(panel_path, predictions_df, score_col=score_col)
        if score_col in predictions_df.columns
        else {}
    )

    # 8b. Integrate Franck's pod_platform.reporting.attribution & pod_platform.reporting.arcana controls
    panel_factors = build_panel_factors(universe)
    f_joined = (
        monthly_eval_df[["total", "rf", "spx"]]
        .join(panel_factors, how="inner")
        .dropna()
    )
    reg_attr = regression_factor_attribution(
        labels=[str(idx) for idx in f_joined.index],
        portfolio_excess=(f_joined["total"] - f_joined["rf"])
        .astype(float)
        .tolist(),
        factors={
            "mkt": (f_joined["spx"] - f_joined["rf"]).astype(float).tolist(),
            "smb": f_joined["smb"].astype(float).tolist(),
            "hml": f_joined["hml"].astype(float).tolist(),
            "mom": f_joined["mom"].astype(float).tolist(),
        },
        factor_set_label="internal_diagnostic_4f_mkt_smb_hml_mom",
    )
    sizing_periods = [
        SizingPeriod(
            label=str(month),
            weights=dict(
                zip(
                    g["permno"].astype(int).astype(str),
                    g["target_weight"].astype(float),
                    strict=True,
                )
            ),
            returns=dict(
                zip(
                    g["permno"].astype(int).astype(str),
                    g["ret"].fillna(0.0).astype(float),
                    strict=True,
                )
            ),
        )
        for month, g in p.groupby("return_month")
    ]
    sel_vs_size_ctrl = selection_vs_sizing(sizing_periods, seed=42, n_draws=200)
    pos_contribs = [
        PositionContribution(
            security_id=str(int(r.permno)),
            period=str(r.return_month),
            side=str(r.side),
            pnl=float(r.pnl) if pd.notna(r.pnl) else 0.0,
        )
        for r in p.itertuples()
    ]
    hit_wl_ctrl = platform_hit_rate_win_loss(
        pos_contribs, unit="return_contribution"
    )
    platform_attribution_controls = {
        "regression_factor_vs_idiosyncratic": reg_attr["aggregate"],
        "regression_alpha_per_period": reg_attr["alpha_per_period"],
        "regression_r_squared": reg_attr["r_squared"],
        "selection_vs_sizing_seeded_control": sel_vs_size_ctrl["aggregate"],
        "hit_rate_win_loss_by_side": {
            "all": hit_wl_ctrl["all"],
            "by_side": hit_wl_ctrl["by_side"],
        },
    }
    (out_dir / "platform_attribution.json").write_text(
        json.dumps(
            {
                "regression_factor_attribution": reg_attr,
                "selection_vs_sizing": sel_vs_size_ctrl,
                "hit_rate_win_loss": hit_wl_ctrl,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    arcana_records = [
        {
            "date": str(pd.Timestamp(r.formation_date).date()),
            "security_id": str(int(r.permno)),
            "weight": float(r.target_weight),
            "source_classification": "formation",
        }
        for r in p.itertuples()
    ]
    arcana_bundle = make_arcana_bundle(
        arcana_records,
        weight_unit="decimal",
        source_files=[
            f for f in (panel_path, external_path) if Path(f).exists()
        ],
        descriptive=False,
    )
    write_arcana_bundle(
        arcana_bundle, out_dir / "arcana_bundle", overwrite=True
    )

    arcana_payload = {
        "score_column": score_col,
        "factor_vs_idiosyncratic": factor_vs_idiosyncratic_split(
            monthly_eval_df, panel_factors
        ),
        "sizing_vs_selection": sizing_vs_selection_split(p),
        "hit_rate_and_win_loss": hit_rate_and_win_loss(p, monthly_eval_df),
        "platform_attribution_controls": platform_attribution_controls,
        "arcana_bundle_sha256": arcana_bundle["payload_sha256"],
        "monthly_alpha_decay": m_decay,
        "daily_alpha_decay": d_decay,
        "earnings_vs_ex_earnings_window": earn_decomp,
        "stress_regime_attribution": stress_regime_attribution(monthly_eval_df),
        "cost_and_borrow_sensitivity": cost_sens,
        "section_6_3_bias_audit": delisted_and_coverage_bias_audit(
            p, universe, liq_df, demand_card_path=demand_card_path
        ),
        "holding_period": compute_holding_period(p),
        "adv_days_distribution": adv_distribution,
    }
    (out_dir / "arcana_audit.json").write_text(
        json.dumps(arcana_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # 9. Section 9.4 Exhibits bundle JSON
    wealth_strat = (1.0 + monthly_eval_df["total"]).cumprod()
    wealth_bench = (1.0 + monthly_eval_df["benchmark"]).cumprod()
    wealth_spx = (1.0 + monthly_eval_df["spx"].fillna(0.0)).cumprod()
    dd_strat = wealth_strat / wealth_strat.cummax().clip(lower=1.0) - 1.0
    dd_spx = wealth_spx / wealth_spx.cummax().clip(lower=1.0) - 1.0
    roll_act_12m = monthly_eval_df["active"].rolling(12).sum()
    roll_ir_12m = (
        math.sqrt(12.0)
        * monthly_eval_df["active"].rolling(12).mean()
        / monthly_eval_df["active"].rolling(12).std()
    )
    ex_m = monthly_eval_df["total"] - monthly_eval_df["rf"]
    mkt_m = monthly_eval_df["spx"] - monthly_eval_df["rf"]
    roll_beta_12m = ex_m.rolling(12).cov(mkt_m) / mkt_m.rolling(12).var()

    exhibits_9_4 = {
        "exhibit_1_cumulative_return_and_relative_wealth": [
            {
                "month": str(idx),
                "strategy_wealth": float(wealth_strat.loc[idx]),
                "benchmark_wealth": float(wealth_bench.loc[idx]),
                "sp500_wealth": float(wealth_spx.loc[idx]),
                "relative_wealth_vs_benchmark": float(
                    wealth_strat.loc[idx] / wealth_bench.loc[idx]
                ),
            }
            for idx in monthly_eval_df.index
        ],
        "exhibit_2_underwater_drawdown": [
            {
                "month": str(idx),
                "strategy_drawdown": float(dd_strat.loc[idx]),
                "sp500_drawdown": float(dd_spx.loc[idx]),
            }
            for idx in monthly_eval_df.index
        ],
        "exhibit_3_rolling_12m_active_and_ir": [
            {
                "month": str(idx),
                "rolling_12m_active_return": None
                if pd.isna(roll_act_12m.loc[idx])
                else float(roll_act_12m.loc[idx]),
                "rolling_12m_ir": None
                if pd.isna(roll_ir_12m.loc[idx])
                else float(roll_ir_12m.loc[idx]),
            }
            for idx in monthly_eval_df.index
        ],
        "exhibit_4_rolling_beta_to_sp500": {
            "monthly_12m_rolling": [
                {
                    "month": str(idx),
                    "rolling_12m_beta": None
                    if pd.isna(roll_beta_12m.loc[idx])
                    else float(roll_beta_12m.loc[idx]),
                }
                for idx in monthly_eval_df.index
            ],
            "daily_252d_rolling_summary": {
                "min": float(daily_df["rolling_252d_beta"].min())
                if not daily_df.empty
                and daily_df["rolling_252d_beta"].notna().any()
                else None,
                "median": float(daily_df["rolling_252d_beta"].median())
                if not daily_df.empty
                and daily_df["rolling_252d_beta"].notna().any()
                else None,
                "max": float(daily_df["rolling_252d_beta"].max())
                if not daily_df.empty
                and daily_df["rolling_252d_beta"].notna().any()
                else None,
            },
        },
        "exhibit_5_calendar_year_returns": cal_years.reset_index()
        .rename(columns={"month": "year"})
        .to_dict(orient="records"),
        "exhibit_6_return_distribution": {
            "monthly_returns": [
                {
                    "month": str(idx),
                    "total_return": float(monthly_eval_df.loc[idx, "total"]),
                    "benchmark_return": float(
                        monthly_eval_df.loc[idx, "benchmark"]
                    ),
                }
                for idx in monthly_eval_df.index
            ],
            "daily_quantiles": (
                {
                    "q01": float(daily_df["total"].quantile(0.01)),
                    "q05": float(daily_df["total"].quantile(0.05)),
                    "q50": float(daily_df["total"].quantile(0.50)),
                    "q95": float(daily_df["total"].quantile(0.95)),
                    "q99": float(daily_df["total"].quantile(0.99)),
                }
                if not daily_df.empty
                else {}
            ),
        },
        "exhibit_7_top_10_long_and_short_holdings": top_hold_df.to_dict(
            orient="records"
        ),
        "exhibit_8_top_10_positive_and_negative_contributors": contrib_df.to_dict(
            orient="records"
        ),
        "adv_days_distribution": adv_distribution,
    }
    (out_dir / "exhibits_9_4.json").write_text(
        json.dumps(exhibits_9_4, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return arcana_payload
