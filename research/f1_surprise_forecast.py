"""F1 — formation-known surprise forecast with card growth (registered 2026-10-25).

(a) partial IC of card growth for the next reported surprise beyond the panel's
own formation-known surprise characters and consensus revision breadth;
(b) walk-forward ridge predicted surprise (formation-known features only) on
next-month excess returns vs a baseline forecast. Register:
docs/review/research_register_20261025.md.
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

RIDGE = 10.0


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


def next_report_surprise(obs: pd.DataFrame) -> pd.DataFrame:
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
    by_permno = {p: g for p, g in surp.groupby("permno")}
    rows = []
    keep = obs.dropna(subset=["eom"]).copy()
    keep["eom"] = pd.to_datetime(keep.eom)
    for permno, part in keep.groupby("permno"):
        reports = by_permno.get(permno)
        if reports is None:
            continue
        dates = reports.anndats.to_numpy()
        for idx, eom in zip(part.index, part.eom.to_numpy()):
            pos = np.searchsorted(dates, np.datetime64(eom), side="right")
            if pos >= len(dates):
                continue
            rows.append({"idx": idx, "suescore": reports.iloc[pos].suescore})
    return keep.join(pd.DataFrame(rows).set_index("idx"))


def formation_features(data: pd.DataFrame) -> pd.DataFrame:
    """Everything a forecaster could use at formation; nothing after eom."""
    chars = pd.read_parquet(
        PANEL,
        columns=["permno", "eom", "niq_su", "saleq_su", "ret_12_1", "saleq_gr1"],
    )
    chars["eom"] = pd.to_datetime(chars.eom)
    data = data.merge(chars, on=["permno", "eom"], how="left")

    cons = external.read("ibes_consensus", columns=["TICKER", "STATPERS", "NUMEST", "NUMUP", "NUMDOWN", "MEANEST"])
    cons["STATPERS"] = pd.to_datetime(cons.STATPERS)
    link = external.read("ibes_link").dropna(subset=["PERMNO"])
    link = link.loc[link.SCORE <= 2].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    cons = cons.merge(link[["TICKER", "permno", "sdate", "edate"]], on="TICKER", how="inner")
    cons = cons.loc[cons.STATPERS.ge(cons.sdate) & cons.STATPERS.le(cons.edate)]
    cons = cons.sort_values("STATPERS")
    cons["rev_breadth"] = (cons.NUMUP - cons.NUMDOWN) / cons.NUMEST.clip(lower=1)

    rows = []
    by_permno = {p: g for p, g in cons.groupby("permno")}
    for permno, part in data.groupby("permno"):
        snaps = by_permno.get(permno)
        if snaps is None:
            continue
        dates = snaps.STATPERS.to_numpy()
        for idx, eom in zip(part.index, part.eom.to_numpy()):
            pos = np.searchsorted(dates, np.datetime64(eom), side="right") - 1
            if pos < 0:
                continue
            snap = snaps.iloc[pos]
            prior = snaps.iloc[max(0, pos - 3)]
            rows.append(
                {
                    "idx": idx,
                    "rev_breadth": snap.rev_breadth,
                    "rev_mom3": float(snap.MEANEST - prior.MEANEST)
                    if np.isfinite(snap.MEANEST) and np.isfinite(prior.MEANEST)
                    else np.nan,
                }
            )
    return data.join(pd.DataFrame(rows).set_index("idx"))


def walk_forward_ridge(data: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Expanding-window ridge on cross-sectional ranks; strictly past months."""
    data = data.copy()
    for col in features + [target]:
        data[f"r_{col}"] = data.groupby("eom")[col].rank(pct=True)
    rcols = [f"r_{c}" for c in features]
    months = sorted(data.eom.unique())
    out = []
    for month in months:
        train = data.loc[data.eom < month]
        test = data.loc[data.eom.eq(month)]
        if len(months) and month <= months[11]:
            continue
        tr = train.dropna(subset=rcols + [f"r_{target}"])
        te = test.dropna(subset=rcols)
        if len(tr) < 500 or len(te) < 20:
            continue
        x = tr[rcols].to_numpy()
        y = tr[f"r_{target}"].to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        xt_x = x.T @ x + RIDGE * np.eye(x.shape[1])
        beta = np.linalg.solve(xt_x, x.T @ y)
        xt = np.column_stack([np.ones(len(te)), te[rcols].to_numpy()])
        te = te.copy()
        te["pred"] = xt @ beta
        out.append(te[["eom", "permno", "pred", "ret_exc_lead1m", "pead", f"r_{target}"]])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    obs = pd.read_parquet(OBS)
    obs["eom"] = pd.to_datetime(obs.eom)
    data = next_report_surprise(obs)
    data = formation_features(data)

    results: dict = {
        "sample": {
            "rows_with_surprise": int(data.suescore.notna().sum()),
            "window": f"{data.eom.min():%Y-%m} .. {data.eom.max():%Y-%m}",
        }
    }

    # (a) partial IC of card growth beyond formation-known surprise proxies
    sub = data.dropna(
        subset=["credit_card_spend_3m_yoy", "suescore", "niq_su", "saleq_su", "rev_breadth"]
    ).copy()
    for col in ("credit_card_spend_3m_yoy", "niq_su", "saleq_su", "rev_breadth"):
        sub[f"r_{col}"] = sub.groupby("eom")[col].rank(pct=True)
    sub["r_target"] = sub.groupby("eom").suescore.rank(pct=True)
    rows = []
    for eom, part in sub.groupby("eom"):
        if len(part) < 40:
            continue
        x = np.column_stack(
            [
                np.ones(len(part)),
                part.r_credit_card_spend_3m_yoy.to_numpy(),
                part.r_niq_su.to_numpy(),
                part.r_saleq_su.to_numpy(),
                part.r_rev_breadth.to_numpy(),
            ]
        )
        y = part.r_target.to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        rows.append(
            {
                "eom": eom,
                "beta_card": beta[1],
                "partial_ic": float(
                    np.corrcoef(part.r_credit_card_spend_3m_yoy.rank().to_numpy(), pd.Series(resid).rank().to_numpy())[0, 1]
                ),
            }
        )
    partial = pd.DataFrame(rows)
    mean, sd = partial.partial_ic.mean(), partial.partial_ic.std(ddof=1)
    results["a_partial_ic_card"] = {
        "months": int(len(partial)),
        "mean_partial_ic": float(mean),
        "t_stat": float(mean / (sd / np.sqrt(len(partial)))) if sd > 0 else None,
        "mean_beta_card": float(partial.beta_card.mean()),
    }

    # (b) walk-forward predicted surprise -> returns, full vs baseline
    full_feats = [
        "credit_card_spend_3m_yoy",
        "credit_card_spend_12m_yoy",
        "niq_su",
        "saleq_su",
        "rev_breadth",
        "rev_mom3",
        "ret_12_1",
    ]
    base_feats = ["niq_su", "saleq_su", "rev_breadth"]
    for label, feats in (("full", full_feats), ("baseline", base_feats)):
        pred = walk_forward_ridge(data, feats, "suescore")
        ic = monthly_ic(pred, "pred", "ret_exc_lead1m")
        results[f"b_{label}__return_ic"] = ic_stats(ic)
        ic2 = monthly_ic(pred, "pred", "pead")
        results[f"b_{label}__pead_ic"] = ic_stats(ic2)
        if label == "full":
            full = pred
        else:
            base = pred

    paired = full.merge(base, on=["eom", "permno"], suffixes=("_f", "_b"))
    rows = []
    for eom, part in paired.groupby("eom"):
        if len(part) < 20:
            continue
        rows.append(
            {
                "eom": eom,
                "ic_f": part.pred_f.rank().corr(part.ret_exc_lead1m_f.rank()),
                "ic_b": part.pred_b.rank().corr(part.ret_exc_lead1m_b.rank()),
            }
        )
    diff = pd.DataFrame(rows)
    diff["d"] = diff.ic_f - diff.ic_b
    mean, sd = diff.d.mean(), diff.d.std(ddof=1)
    results["b_paired_full_minus_baseline"] = {
        "months": int(len(diff)),
        "mean_ic_diff": float(mean),
        "t_stat": float(mean / (sd / np.sqrt(len(diff)))) if sd > 0 else None,
    }

    (OUT / "f1_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
