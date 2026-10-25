"""R2 — demand growth predicts earnings surprise; drift pays (registered 2026-10-25).

Test (a) demand growth ranks the next reported I/B/E/S surprise beyond revision
breadth; (b) post-report drift concentrates in demand-conditioned names within
surprise quintiles; (c) demand x surprise-sign interaction. Retrospective replay
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

DEMAND = [
    "credit_card_spend_3m_yoy",
    "credit_card_spend_12m_yoy",
    "foot_traffic_3m_yoy",
    "foot_traffic_12m_yoy",
]


def monthly_ic(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    rows = []
    for eom, part in frame.groupby("eom"):
        part = part.dropna(subset=[score, target])
        if len(part) < 20:
            continue
        rows.append(
            {"eom": eom, "n": len(part), "ic": part[score].rank().corr(part[target].rank())}
        )
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


def next_report_surprise(obs: pd.DataFrame) -> pd.DataFrame:
    """First I/B/E/S report after formation, with its realized surprise."""
    surp = external.read("ibes_surprise")
    surp["anndats"] = pd.to_datetime(surp.anndats)
    surp = surp.dropna(subset=["anndats", "suescore"])
    link = external.read("ibes_link").dropna(subset=["PERMNO"])
    link = link.loc[link.SCORE <= 2].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    surp = surp.merge(link[["TICKER", "permno", "sdate", "edate"]], on="TICKER", how="inner")
    surp = surp.loc[surp.anndats.ge(surp.sdate) & surp.anndats.le(surp.edate)]
    surp = surp.sort_values("anndats").drop_duplicates(["permno", "anndats", "MEASURE"])

    rows = []
    keep = obs.dropna(subset=["eom"]).copy()
    keep["eom"] = pd.to_datetime(keep.eom)
    by_permno = {p: g for p, g in surp.groupby("permno")}
    for permno, part in keep.groupby("permno"):
        reports = by_permno.get(permno)
        if reports is None:
            continue
        dates = reports.anndats.to_numpy()
        for idx, eom in zip(part.index, part.eom.to_numpy()):
            pos = np.searchsorted(dates, np.datetime64(eom), side="right")
            if pos >= len(dates):
                continue
            rep = reports.iloc[pos]
            rows.append(
                {
                    "idx": idx,
                    "report_date": rep.anndats,
                    "suescore": rep.suescore,
                    "surpmean": rep.surpmean,
                    "actual": rep.actual,
                }
            )
    out = pd.DataFrame(rows).set_index("idx")
    return keep.join(out)


def main() -> None:
    obs = pd.read_parquet(OBS)
    obs["eom"] = pd.to_datetime(obs.eom)
    data = next_report_surprise(obs)
    results: dict = {
        "sample": {
            "rows_total": int(len(obs)),
            "rows_with_next_report": int(data.suescore.notna().sum()),
            "permno": int(data.permno.nunique()),
            "window": f"{data.eom.min():%Y-%m} .. {data.eom.max():%Y-%m}",
        }
    }

    # (a) demand growth -> next reported surprise
    for col in DEMAND:
        ic = monthly_ic(data, col, "suescore")
        results[f"a_surprise_ic__{col}"] = ic_stats(ic)
        ic["period"] = np.where(ic.eom < pd.Timestamp("2023-01-01"), "2019_2022", "2023_2026")
        for period, part in ic.groupby("period"):
            results[f"a_surprise_ic__{col}__{period}"] = ic_stats(part)

    # (a-control) does demand add anything over revision breadth? within-month
    # rank regression of suescore on demand rank + NUMUP/NUMDOWN is not available
    # at this grain; instead report the same-period correlation of demand rank
    # with the panel's own niq_su (analyst-independent surprise proxy).
    if "niq_su" in data.columns:
        control = data.dropna(subset=["niq_su"])
        rows = []
        for col in DEMAND[:2]:
            for eom, part in control.groupby("eom"):
                part = part.dropna(subset=[col])
                if len(part) < 20:
                    continue
                rows.append({"eom": eom, "ic": part[col].rank().corr(part.niq_su.rank())})
        if rows:
            results["a_control__demand_vs_panel_niq_su"] = ic_stats(pd.DataFrame(rows))

    # (b) PEAD spread: within surprise quintiles, demand-conditioned drift
    for col in ("credit_card_spend_12m_yoy", "foot_traffic_3m_yoy"):
        rows = []
        sub = data.dropna(subset=[col, "suescore", "pead"])
        for eom, part in sub.groupby("eom"):
            if len(part) < 40:
                continue
            try:
                part = part.assign(
                    q_s=pd.qcut(part.suescore, 5, labels=False, duplicates="drop"),
                    q_d=pd.qcut(part[col], 5, labels=False, duplicates="drop"),
                )
            except ValueError:
                continue
            highs, lows = [], []
            for _, grp in part.groupby("q_s"):
                if grp.q_d.nunique() < 2:
                    continue
                highs.append(grp.loc[grp.q_d == grp.q_d.max(), "pead"].mean())
                lows.append(grp.loc[grp.q_d == grp.q_d.min(), "pead"].mean())
            if highs and lows:
                rows.append({"eom": eom, "spread": np.nanmean(highs) - np.nanmean(lows)})
        spread = pd.DataFrame(rows)
        stats = ic_stats(spread.rename(columns={"spread": "ic"})) if len(spread) else {"months": 0}
        results[f"b_pead_spread__{col}"] = stats

    # (c) tradable interaction: demand growth x consensus revision direction
    # known at formation (registered form: "demand growth x negative revision
    # direction"). Revision direction is NUMUP - NUMDOWN on the last consensus
    # snapshot on or before formation; nothing after eom enters the feature.
    cons = external.read("ibes_consensus", columns=["TICKER", "STATPERS", "NUMUP", "NUMDOWN"])
    cons["STATPERS"] = pd.to_datetime(cons.STATPERS)
    link = external.read("ibes_link").dropna(subset=["PERMNO"])
    link = link.loc[link.SCORE <= 2].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    cons = cons.merge(link[["TICKER", "permno", "sdate", "edate"]], on="TICKER", how="inner")
    cons = cons.loc[cons.STATPERS.ge(cons.sdate) & cons.STATPERS.le(cons.edate)]
    cons = cons.sort_values("STATPERS")

    rows = []
    by_permno = {p: g for p, g in cons.groupby("permno")}
    keep = data.dropna(subset=["eom", "credit_card_spend_12m_yoy"]).copy()
    for permno, part in keep.groupby("permno"):
        snaps = by_permno.get(permno)
        if snaps is None:
            continue
        dates = snaps.STATPERS.to_numpy()
        for idx, eom in zip(part.index, part.eom.to_numpy()):
            pos = np.searchsorted(dates, np.datetime64(eom), side="right") - 1
            if pos < 0:
                continue
            snap = snaps.iloc[pos]
            rows.append({"idx": idx, "rev_dir": np.sign(snap.NUMUP - snap.NUMDOWN)})
    rev = pd.DataFrame(rows).set_index("idx")
    inter = keep.join(rev).dropna(subset=["rev_dir", "ret_exc_lead1m"])
    inter["interact"] = inter.credit_card_spend_12m_yoy * inter.rev_dir
    results["c_interaction_ic"] = ic_stats(monthly_ic(inter, "interact", "ret_exc_lead1m"))
    results["c_interaction_sample"] = {"rows": int(len(inter))}
    # flagged in the register: the earlier draft used realized report surprise
    # (not known at formation); that variant is recorded as contaminated.
    sub = data.dropna(subset=["credit_card_spend_12m_yoy", "suescore", "ret_exc_lead1m"]).copy()
    sub["interact"] = sub.credit_card_spend_12m_yoy * np.sign(sub.suescore)
    results["c_CONTAMINATED_realized_surprise_sign"] = ic_stats(
        monthly_ic(sub, "interact", "ret_exc_lead1m")
    )

    (OUT / "r2_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
