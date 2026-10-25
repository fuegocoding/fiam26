"""R3 — short-leg repair (registered 2026-10-25).

(a) foot-traffic 3m growth as a short-side signal (negative tilt), (b) the tilt
conditioned on short-interest crowding, (c) long + index-hedge comparator vs
stock-specific shorts (screen, not a full backtest). Retrospective replay
(spec 2.4); register: docs/review/research_register_20261025.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import external  # noqa: E402

external.FILE = ROOT / "final" / "external_data" / "external.parquet"

OUT = ROOT / "research" / "out"
OUT.mkdir(parents=True, exist_ok=True)
OBS = ROOT / "final_v2" / "artifacts" / "hypothesis_battery_obs.parquet"
PANEL = ROOT / "data" / "raw" / "final" / "chars_final_with_names.parquet"


def monthly_ic(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    rows = []
    for eom, part in frame.groupby("eom"):
        part = part.dropna(subset=[score, target])
        if len(part) < 20:
            continue
        rows.append({"eom": eom, "n": len(part), "ic": part[score].rank().corr(part[target].rank())})
    return pd.DataFrame(rows)


def ic_stats(ic: pd.DataFrame) -> dict:
    if ic.empty:
        return {"months": 0}
    mean, sd = ic.ic.mean(), ic.ic.std(ddof=1)
    t = mean / (sd / np.sqrt(len(ic))) if sd and sd > 0 else np.nan
    return {
        "months": int(len(ic)),
        "mean_ic": float(mean),
        "t_stat": float(t),
        "hit_rate": float((ic.ic > 0).mean()),
    }


def load_crowding() -> pd.DataFrame:
    """Dated short interest (days_to_cover) by ticker, lagged 14 days."""
    finra = external.read("finra")
    finra["settlement"] = pd.to_datetime(finra.settlement)
    finra["public"] = finra.settlement + pd.Timedelta(days=14)
    finra["symbol"] = finra.symbol.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    panel = pd.read_parquet(PANEL, columns=["permno", "eom", "ticker"])
    panel["eom"] = pd.to_datetime(panel.eom)
    panel["symbol"] = panel.ticker.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    merged = panel.merge(finra[["symbol", "public", "days_to_cover", "short_shares"]], on="symbol", how="left")
    merged = merged.loc[merged.public.le(merged.eom)]
    merged = merged.sort_values(["permno", "eom", "public"])
    out = merged.groupby(["permno", "eom"], as_index=False).tail(1)
    return out[["permno", "eom", "days_to_cover", "short_shares"]]


def load_sp500() -> pd.Series:
    fred = external.read("fred")
    sp = fred.loc[fred.series.astype(str).str.contains("SP500|SP_500|spx", case=False, na=False)]
    if sp.empty:
        sp = fred.loc[fred.series.eq("SP500")]
    sp = sp.copy()
    sp["date"] = pd.to_datetime(sp.date)
    sp = sp.sort_values("date").set_index("date").value
    sp = sp.resample("ME").last()
    return sp.pct_change().dropna()


def hedge_comparator(monthly: pd.DataFrame, sp: pd.Series) -> dict:
    """Screen: long leg + index short at the book's short-side notional."""
    m = monthly.copy()
    m["month"] = pd.to_datetime(m.month).dt.to_period("M").dt.to_timestamp("M")
    m = m.set_index("month").join(sp.rename("sp500"), how="inner")
    m = m.dropna(subset=["pnl_long", "pnl_short", "gross_exposure", "sp500"])
    rf = m["rf_monthly"] if "rf_monthly" in m else 0.0
    actual = m.pnl_long + m.pnl_short - m.get("cost", 0) - m.get("borrow", 0)
    short_notional = m.gross_exposure / 2.0
    hedged = m.pnl_long - short_notional * m.sp500 - m.get("cost", 0) - m.get("borrow", 0)

    def stats(r: pd.Series) -> dict:
        active = r - rf
        sd = active.std(ddof=1)
        ir = active.mean() / sd * np.sqrt(12) if sd and sd > 0 else np.nan
        return {
            "months": int(len(r)),
            "ann_return": float(r.mean() * 12),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(12)),
            "ir_vs_rf": float(ir),
        }

    out = {"actual_book": stats(actual), "long_plus_index_hedge": stats(hedged)}
    out["short_leg_ann_pnl"] = float(m.pnl_short.mean() * 12)
    for label, mask in (
        ("2021_2022", m.index < pd.Timestamp("2023-01-01")),
        ("2023_2026", m.index >= pd.Timestamp("2023-01-01")),
    ):
        part = m.loc[mask]
        out[f"short_leg_ann_pnl_{label}"] = float(part.pnl_short.mean() * 12)
    return out


def main() -> None:
    obs = pd.read_parquet(OBS)
    obs["eom"] = pd.to_datetime(obs.eom)
    results: dict = {}

    # (a) registered short-side hypothesis: traffic 3m growth has a negative
    # tilt on next-month excess returns.
    for col in ("foot_traffic_3m_yoy", "foot_traffic_12m_yoy"):
        ic = monthly_ic(obs, col, "ret_exc_lead1m")
        results[f"a_return_ic__{col}"] = ic_stats(ic)
        ic["period"] = np.where(ic.eom < pd.Timestamp("2023-01-01"), "2019_2022", "2023_2026")
        for period, part in ic.groupby("period"):
            results[f"a_return_ic__{col}__{period}"] = ic_stats(part)
        top = obs.dropna(subset=[col, "ret_exc_lead1m"]).copy()
        rows = []
        for eom, part in top.groupby("eom"):
            if len(part) < 40:
                continue
            q = pd.qcut(part[col], 5, labels=False, duplicates="drop")
            if q.nunique() < 5:
                continue
            hi = part.loc[q == q.max(), "ret_exc_lead1m"].mean()
            lo = part.loc[q == q.min(), "ret_exc_lead1m"].mean()
            rows.append({"eom": eom, "spread": hi - lo})
        if rows:
            results[f"a_top_minus_bottom__{col}"] = ic_stats(
                pd.DataFrame(rows).rename(columns={"spread": "ic"})
            )

    # (b) crowding interaction: the tilt should be stronger where the short
    # side is already crowded (high days-to-cover).
    crowding = load_crowding()
    merged = obs.merge(crowding, on=["permno", "eom"], how="left")
    sub = merged.dropna(subset=["foot_traffic_3m_yoy", "days_to_cover", "ret_exc_lead1m"]).copy()
    sub["traffic_x_crowding"] = sub.foot_traffic_3m_yoy * np.sign(sub.days_to_cover - sub.days_to_cover.median())
    results["b_interaction_ic"] = ic_stats(monthly_ic(sub, "traffic_x_crowding", "ret_exc_lead1m"))
    results["b_sample_rows"] = int(len(sub))
    hi = sub.loc[sub.days_to_cover.ge(sub.days_to_cover.quantile(0.75))]
    lo = sub.loc[sub.days_to_cover.le(sub.days_to_cover.quantile(0.25))]
    results["b_ic_high_crowding"] = ic_stats(monthly_ic(hi, "foot_traffic_3m_yoy", "ret_exc_lead1m"))
    results["b_ic_low_crowding"] = ic_stats(monthly_ic(lo, "foot_traffic_3m_yoy", "ret_exc_lead1m"))

    # (c) long + index-hedge comparator (screen)
    sp = load_sp500()
    for run in ("tfm_daily_tilt_squeeze_175", "risk_book_graph_gate"):
        path = ROOT / "final_v2" / "artifacts" / run / "monthly.json"
        if path.exists():
            monthly = pd.DataFrame(json.loads(path.read_text()))
            results[f"c_comparator__{run}"] = hedge_comparator(monthly, sp)

    (OUT / "r3_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
