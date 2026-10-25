"""F2 — 8-K corpus survivorship test (registered 2026-10-25, brief section 6.3).

(a) named 2017-2020 delisters vs survivors in the corpus; (b) covered vs
uncovered average returns in our universe. Register:
docs/review/research_register_20261025.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
OUT.mkdir(parents=True, exist_ok=True)

FILINGS = ROOT / "data" / "raw" / "final" / "8k_20150101_20260831_identified.parquet"
PANEL = ROOT / "data" / "raw" / "final" / "chars_final_with_names.parquet"

NAMED_DELISTERS = {
    "Sears": ["SHLD"],
    "Bon-Ton": ["BONT"],
    "Pier 1": ["PIR"],
    "GNC": ["GNC"],
    "Ascena": ["ASNA"],
    "J.C. Penney": ["JCP"],
}


def main() -> None:
    filings = pd.read_parquet(FILINGS, columns=["permno", "filing_date", "is_amendment"])
    filings = filings.loc[~filings.is_amendment.fillna(False) & filings.permno.notna()]
    filings["permno"] = filings.permno.astype(int)
    filings["filing_date"] = pd.to_datetime(filings.filing_date)
    filings["year"] = filings.filing_date.dt.year
    covered = filings.groupby(["permno", "year"]).size().rename("n_filings").reset_index()

    panel = pd.read_parquet(
        PANEL, columns=["permno", "eom", "ticker", "company_name", "ret_exc_lead1m", "me"]
    )
    panel["eom"] = pd.to_datetime(panel.eom)
    panel = panel.drop_duplicates(["permno", "eom"])

    last = panel.groupby("permno").eom.max().rename("last_eom")
    first = panel.groupby("permno").eom.min().rename("first_eom")
    life = pd.concat([first, last], axis=1)
    life["exited_2017_2020"] = life.last_eom.between("2017-01-01", "2020-12-31")
    life["survivor"] = life.last_eom.ge("2026-07-01")

    corpus_names = set(filings.permno.unique())
    any_ever = set(covered.loc[covered.year.between(2015, 2026), "permno"].unique())

    results: dict = {
        "sample": {
            "filings": int(len(filings)),
            "permno_in_corpus": int(len(corpus_names)),
            "permno_in_panel": int(len(life)),
            "exited_2017_2020": int(life.exited_2017_2020.sum()),
            "survivors": int(life.survivor.sum()),
        }
    }

    def resolve_rate(mask: pd.Series) -> dict:
        permnos = life.index[mask]
        hit = sum(1 for p in permnos if p in any_ever)
        return {"n": int(len(permnos)), "in_corpus": hit, "share": float(hit / len(permnos)) if len(permnos) else None}

    results["a_named_delisters"] = {}
    names = panel.dropna(subset=["ticker"]).drop_duplicates("permno").set_index("permno")
    for label, tickers in NAMED_DELISTERS.items():
        found = names[names.ticker.astype(str).str.upper().isin(tickers)]
        entry = {"tickers_found": int(len(found)), "in_corpus": 0, "permnos": []}
        for permno, row in found.iterrows():
            entry["permnos"].append(int(permno))
            if permno in any_ever:
                entry["in_corpus"] += 1
            entry.setdefault("last_eom", {})[int(permno)] = str(life.loc[permno, "last_eom"])
            entry.setdefault("filings_by_year", {})[int(permno)] = {
                int(y): int(n)
                for y, n in covered.loc[covered.permno.eq(permno)].set_index("year").n_filings.items()
            }
        results["a_named_delisters"][label] = entry

    results["a_exited_vs_survivors"] = {
        "exited_2017_2020": resolve_rate(life.exited_2017_2020),
        "survivors": resolve_rate(life.survivor),
        "all_panel": resolve_rate(pd.Series(True, index=life.index)),
    }

    # (b) covered vs uncovered returns: any 8-K in the trailing 12 months
    ev = filings[["permno", "filing_date"]].copy()
    ev["filing_month"] = ev.filing_date.dt.to_period("M").dt.to_timestamp("M")
    keys = panel[["permno", "eom"]].copy()
    keys["lag12"] = keys.eom - pd.DateOffset(months=12)
    merged = keys.merge(ev, on="permno", how="left")
    merged = merged.loc[merged.filing_month.between(merged.lag12, merged.eom)]
    flagged = merged.groupby(["permno", "eom"]).size().rename("n_filings_12m").reset_index()
    data = panel.merge(flagged, on=["permno", "eom"], how="left")
    data["n_filings_12m"] = data.n_filings_12m.fillna(0)
    data["covered"] = data.n_filings_12m.gt(0)

    rows = []
    for eom, part in data.groupby("eom"):
        cov = part.loc[part.covered, "ret_exc_lead1m"].dropna()
        unc = part.loc[~part.covered, "ret_exc_lead1m"].dropna()
        if len(cov) < 10 or len(unc) < 10:
            continue
        rows.append(
            {
                "eom": eom,
                "n_cov": len(cov),
                "n_unc": len(unc),
                "diff": cov.mean() - unc.mean(),
            }
        )
    diff = pd.DataFrame(rows)
    mean, sd = diff["diff"].mean(), diff["diff"].std(ddof=1)
    results["b_covered_minus_uncovered_return"] = {
        "months": int(len(diff)),
        "mean_monthly_diff": float(mean),
        "t_stat": float(mean / (sd / np.sqrt(len(diff)))) if sd > 0 else None,
        "share_covered": float(data.covered.mean()),
    }

    # size/tilt of covered names (is coverage return-tilted via size?)
    results["b_coverage_by_size"] = {
        str(k): float(v)
        for k, v in data.groupby(pd.qcut(data.me.rank(pct=True), 5, duplicates="drop"))["covered"].mean().items()
    }

    (OUT / "f2_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
