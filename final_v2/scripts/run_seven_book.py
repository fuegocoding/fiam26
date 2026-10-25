"""Seven-sleeve run: five deterministic ports + two trained sleeves.

Trains xgb_rank (walk-forward on 147 chars) and filing_events (walk-forward
on EDGAR event counts with dated cik links), caches the forecasts, blends
all seven as families, and runs the engine end to end.

Usage: python final_v2/scripts/run_seven_book.py [--reuse-forecasts]
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.allocation import (  # noqa: E402
    Candidate,
    ConstructionEngine,
    FamilyEstimate,
    FamilyIntegrator,
    InternalPolicy,
    OfficialLimits,
)
from pod_platform.data.competition import CharsPanelLoader  # noqa: E402
from pod_platform.reporting.compliance import ActualHolding, ComplianceEngine  # noqa: E402
from pod_platform.reporting.tables import RequiredTable  # noqa: E402
from pod_platform.risk.overlay import PANEL_COLUMNS as OVERLAY_COLUMNS  # noqa: E402
from pod_platform.risk.overlay import protect  # noqa: E402
from pod_platform.signals import (  # noqa: E402
    PortedSignals,
    build_filing_event_features,
    build_filing_events,
    build_xgb_rank,
    scaled_rank,
)

REPO = ROOT.parent
PANEL = REPO / "final" / "data" / "chars_final_with_names.parquet"
FACTORS = REPO / "final" / "data" / "factor_char_list.csv"
EXTERNAL = REPO / "final" / "external_data" / "external.parquet"
FILINGS = REPO / "final" / "data" / "8k_20150101_20260831_identified.parquet"
CACHE = ROOT / "artifacts" / "seven_book" / "forecasts"

TEST_START = dt.date(2021, 1, 31)
TEST_END = dt.date(2026, 8, 31)
TEST_YEARS = list(range(2021, 2027))
LAST_TARGET = pd.Timestamp("2026-08-31")
COST_BP_PER_SIDE = 20.0
BORROW_ANNUAL = 0.05
USE_RISK_OVERLAY = "--risk-overlay" in sys.argv
USE_GAP = "--with-gap" in sys.argv
BLEND_RENORM = "--blend-renorm" in sys.argv
OUT = ROOT / "artifacts" / (
    ("risk_book_renorm" if BLEND_RENORM else "risk_book") if USE_RISK_OVERLAY else "seven_book"
)
GAP_WORK = ROOT / "artifacts" / "valuation_work"
FAMILIES = (
    "inventory_retail",
    "oplev_software",
    "runway_micro",
    "e7_surprise_breadth",
    "conf_revision_gated",
    "xgb_rank",
    "filing_events",
) + (("gap_micro",) if USE_GAP else ())


def load_tb3ms() -> dict[str, float]:
    ext = pd.read_parquet(EXTERNAL, columns=["table", "date", "series", "value"])
    fred = ext[ext["table"] == "fred"].copy()
    fred["date"] = pd.to_datetime(fred["date"])
    tb = fred[fred["series"] == "TB3MS"].set_index("date")["value"]
    return {f"{d.year}-{d.month:02d}": float(v) for d, v in tb.items()}


def get_forecasts(panel_full, factor_columns, reuse: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    xgb_path = CACHE / "xgb_rank.parquet"
    filing_path = CACHE / "filing_events.parquet"
    audits_path = CACHE / "fold_audits.json"

    audits: dict = {}
    if reuse and xgb_path.exists() and filing_path.exists():
        xgb_fc = pd.read_parquet(xgb_path)
        filing_fc = pd.read_parquet(filing_path)
        if audits_path.exists():
            audits = json.loads(audits_path.read_text(encoding="utf-8"))
        return xgb_fc, filing_fc, audits

    t0 = time.time()
    print("training xgb_rank (24 fits)...", flush=True)
    xgb_fc, xgb_audits = build_xgb_rank(
        panel_full,
        factor_columns,
        test_years=TEST_YEARS,
        last_target=LAST_TARGET,
        on_fold=lambda a: print(
            f"  {a.year}: train={a.train_rows:,} val={a.validation_rows:,} test={a.test_rows:,} iters={a.best_iterations}",
            flush=True,
        ),
    )
    xgb_fc.to_parquet(xgb_path, index=False)
    print(f"xgb_rank done in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    print("building filing-event features...", flush=True)
    ext = pd.read_parquet(EXTERNAL, columns=["table", "form_type", "date_filed", "cik"])
    filings_index = ext[ext["table"] == "edgar"][["form_type", "date_filed", "cik"]].copy()
    filings_index["date_filed"] = pd.to_datetime(filings_index["date_filed"])
    corpus = pd.read_parquet(FILINGS, columns=["permno", "cik", "filing_date", "is_amendment"])
    corpus = corpus[corpus["is_amendment"] == False]  # noqa: E712
    links = corpus.dropna(subset=["permno", "cik"]).copy()
    links = links.groupby(["cik", "permno"]).size().reset_index(name="n")
    links["link_date"] = pd.Timestamp("2015-01-01")
    months = pd.DatetimeIndex(
        sorted({pd.Timestamp(m) + pd.offsets.MonthEnd(0) for m in panel_full["eom"].unique()})
    )
    events = build_filing_event_features(filings_index, links[["cik", "permno", "link_date"]], months)
    print(f"filing features: {len(events):,} rows x {events.shape[1] - 2} cols in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    print("training filing_events (24 fits)...", flush=True)
    filing_fc, filing_audits = build_filing_events(
        panel_full,
        events,
        test_years=TEST_YEARS,
        last_target=LAST_TARGET,
        on_fold=lambda a: print(
            f"  {a.year}: train={a.train_rows:,} val={a.validation_rows:,} test={a.test_rows:,} iters={a.best_iterations}",
            flush=True,
        ),
    )
    filing_fc.to_parquet(filing_path, index=False)
    print(f"filing_events done in {time.time() - t0:.0f}s", flush=True)

    audits = {
        "xgb_rank": [a.__dict__ for a in xgb_audits],
        "filing_events": [a.__dict__ for a in filing_audits],
    }
    audits_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    return xgb_fc, filing_fc, audits


def get_gap_map():
    """Trains the valuation family once, returns {(sid, date): gap score}."""
    import json as _json

    from pod_platform.signals import valuation as val
    from pod_platform.signals.ibes_state import set_external_path

    set_external_path(str(EXTERNAL))
    GAP_WORK.mkdir(parents=True, exist_ok=True)
    out = val.directory(GAP_WORK)
    if not (out / val.SELECTION_FILE).exists():
        print("training valuation family (gap_micro) - long run...", flush=True)
        val.build(PANEL, GAP_WORK, device="cpu")
    selection = _json.loads((out / val.SELECTION_FILE).read_text())
    xs = pd.read_parquet(out / val.XS_FILE)
    pooled = pd.read_parquet(out / val.POOLED_FILE)
    gap_map = {}
    for year_s, fold in selection["folds"].items():
        year = int(year_s)
        model = fold["best_pricing_model"]
        print(f"  gap_micro {year}: selected {model} (mdae {min(fold['validation_mdae'].values()):.3f})", flush=True)
        if model in val.POOLED_MODELS or model == "M9":
            needed = [c for c in val.ENSEMBLE if c not in val.POOLED_MODELS]
            block = pooled[pooled.fold_year == year][["permno", "eom", "M6", "M8"]].merge(
                xs[["permno", "eom", "y", *needed]], on=["permno", "eom"], how="left"
            )
            if model == "M9":
                block["M9"] = block[list(val.ENSEMBLE)].mean(axis=1, skipna=False)
        else:
            block = xs
        block = block.dropna(subset=[model, "y"])
        for row in block.itertuples():
            gap_map[(str(int(row.permno)), row.eom.date())] = float(getattr(row, model) - row.y)
    return gap_map


def main() -> None:
    reuse = "--reuse-forecasts" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    loader = CharsPanelLoader(PANEL, FACTORS)
    panel = loader.load()
    panel["eom"] = pd.to_datetime(panel["eom"])
    signals = PortedSignals(EXTERNAL)
    tb3ms = load_tb3ms()

    factor_columns = pd.read_csv(FACTORS)["variable"].tolist()
    panel_columns = list(
        dict.fromkeys(["permno", "eom", "ret_exc_lead1m", "ret_exc", "rmax1_21d", "size_grp", "gics", *factor_columns])
    )
    panel_full = pd.read_parquet(PANEL, columns=panel_columns)
    panel_full["eom"] = pd.to_datetime(panel_full["eom"])
    overlay_panel = panel_full[OVERLAY_COLUMNS].copy()
    overlay_panel["eom"] = overlay_panel["eom"].dt.to_period("M")
    overlay_panel["permno"] = overlay_panel["permno"].astype(int).astype(str)

    xgb_fc, filing_fc, audits = get_forecasts(panel_full, factor_columns, reuse)
    gap_map = get_gap_map() if USE_GAP else {}
    xgb_map = {
        (str(int(r.permno)), r.eom.date()): float(r.pred_xgb_rank) for r in xgb_fc.itertuples()
    }
    filing_map = {
        (str(int(r.permno)), r.eom.date()): float(r.pred_filing_events) for r in filing_fc.itertuples()
    }

    integrator = FamilyIntegrator(FAMILIES, neutral_uncertainty=0.1)
    engine = ConstructionEngine(OfficialLimits(), InternalPolicy(target_positions=300))
    compliance = ComplianceEngine(OfficialLimits())

    months = sorted({m for m in panel["eom"].dt.date.unique() if TEST_START <= m <= TEST_END})
    print(f"formation months: {len(months)} ({months[0]} .. {months[-1]})")

    nav = 1_000_000_000.0
    nav_path = [{"month": "start", "nav": nav}]
    monthly = []
    last_holdings = []
    prev_weights: dict[str, float] = {}
    coverage_log = []
    score_history: list[pd.DataFrame] = []
    overlay_log = []

    for month_date in months:
        month_ts = pd.Timestamp(month_date)
        month_df, _ = loader.formation_month(
            dt.datetime(month_date.year, month_date.month, month_date.day, tzinfo=dt.timezone.utc)
        )
        if len(month_df) < 250:
            continue

        results = {
            "inventory_retail": signals.inventory_retail(month_df),
            "oplev_software": signals.oplev_software(month_df),
            "runway_micro": signals.runway_micro(month_df),
            "e7_surprise_breadth": signals.e7_surprise_breadth(month_df, month_ts),
            "conf_revision_gated": signals.conf_revision_gated(month_df, month_ts),
        }
        scaled = {name: scaled_rank(res) for name, res in results.items()}
        coverage = {name: int(res.eligible.fillna(False).sum()) for name, res in results.items()}

        merged = month_df.copy()
        merged["sid"] = merged["permno"].astype(int).astype(str)
        keys = [(sid, month_date) for sid in merged["sid"]]
        merged["xgb_rank"] = [xgb_map.get(k) for k in keys]
        merged["filing_events"] = [filing_map.get(k) for k in keys]
        if USE_GAP:
            merged["gap_micro"] = [gap_map.get(k) for k in keys]
            micro = merged["size_grp"].eq("micro") & merged["gap_micro"].notna()
            ranked = merged.loc[micro, "gap_micro"].rank(pct=True)
            scaled["gap_micro"] = ((ranked - 0.5) * (12 ** 0.5)).reindex(merged.index).fillna(0.0)
            coverage["gap_micro"] = int(micro.sum())
        coverage["xgb_rank"] = int(merged["xgb_rank"].notna().sum())
        coverage["filing_events"] = int(merged["filing_events"].notna().sum())
        coverage_log.append({"month": str(month_date), "coverage": coverage})

        for name in ("xgb_rank", "filing_events"):
            series = merged[name]
            ranked = series.rank(pct=True)
            scaled[name] = ((ranked - 0.5) * (12 ** 0.5)).where(series.notna()).fillna(0.0)

        candidates = []
        candidates_alpha = {}
        for sid, idx in zip(merged["sid"], merged.index):
            fams = []
            for name in FAMILIES:
                value = float(scaled[name].get(idx, 0.0))
                if value != 0.0:
                    fams.append(FamilyEstimate(name, sid, value, 0.1))
            if not fams:
                continue
            alpha = sum(f.expected_excess_return for f in fams) / (
                len(fams) if BLEND_RENORM else len(FAMILIES)
            )
            candidates_alpha[sid] = alpha
            row = merged.loc[idx]
            beta_value = row.get("beta_60m")
            candidates.append(
                Candidate(
                    security_id=sid,
                    alpha=alpha,
                    market_cap_usd=float(row["me_usd"]),
                    tradable=True,
                    incumbent=sid in prev_weights,
                    sector=str(row["gics"])[:2] if row["gics"] else "unknown",
                    beta=float(beta_value) if beta_value == beta_value and beta_value is not None else 1.0,
                )
            )
        if len(candidates) < 250:
            continue

        book = engine.build(
            candidates,
            run_id=f"seven_{month_date}",
            decision_at=dt.datetime(month_date.year, month_date.month, month_date.day, tzinfo=dt.timezone.utc),
            forecast_publication_ids=[f"fc_{month_date}"],
            current_book_snapshot_id=f"snap_{month_date}",
            nav_snapshot_id=f"nav_{month_date}",
            price_snapshot_id=f"px_{month_date}",
            risk_envelope_version="envelope_v1",
            covariance_version="cov_v1",
            cost_model_version="cost_v1",
            attribution_method="family_equal_contribution",
        )

        weights = {p.security_id: float(p.target_weight) for p in book.positions}

        ret_month = (month_ts + pd.offsets.MonthEnd(1)).to_period("M")
        for sid, idx in zip(merged["sid"], merged.index):
            score_history.append(
                {
                    "return_month": ret_month,
                    "permno": sid,
                    "final_score": float(candidates_alpha.get(sid, 0.0)),
                    "prc": float(merged.loc[idx, "prc"]),
                }
            )

        if USE_RISK_OVERLAY and weights:
            scores_df = pd.DataFrame(score_history)
            overlay_result = protect(weights, ret_month, overlay_panel, scores_df)
            weights = overlay_result.weights
            overlay_log.append({"month": str(month_date), **overlay_result.summary})

        turnover = 0.5 * sum(
            abs(weights.get(k, 0.0) - prev_weights.get(k, 0.0))
            for k in set(weights) | set(prev_weights)
        )

        rets = merged.set_index("sid")["ret_exc_lead1m"].to_dict()
        missing_weight = sum(
            abs(w) for k, w in weights.items()
            if rets.get(k) is None or (isinstance(rets.get(k), float) and math.isnan(rets[k]))
        )
        resolved_weight = sum(abs(w) for w in weights.values())
        missing_share = missing_weight / resolved_weight if resolved_weight else 1.0
        if missing_share > 0.5:
            continue

        resolved = {
            k: (0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
            for k, v in rets.items()
        }
        month_key = f"{month_date.year}-{month_date.month:02d}"
        rf_m = tb3ms.get(month_key, 0.0) / 1200.0
        bench_m = rf_m + 0.04 / 12.0
        active_ret = sum(w * resolved.get(k, 0.0) for k, w in weights.items())
        short_w = -sum(w for w in weights.values() if w < 0)
        borrow = short_w * BORROW_ANNUAL / 12.0
        cost = turnover * COST_BP_PER_SIDE / 10_000.0 * 2.0
        gross_ret = rf_m + active_ret
        net_ret = gross_ret - borrow - cost

        nav *= 1.0 + net_ret
        nav_path.append({"month": str(month_date), "nav": nav})
        monthly.append(
            {
                "month": str(month_date),
                "gross_return": gross_ret,
                "net_return": net_ret,
                "rf_monthly": rf_m,
                "benchmark_monthly": bench_m,
                "active_return": active_ret,
                "missing_weight_share": missing_share,
                "borrow": borrow,
                "cost": cost,
                "turnover": turnover,
                "names": len(weights),
                "gross_exposure": sum(abs(w) for w in weights.values()),
                "net_exposure": sum(weights.values()),
            }
        )
        prev_weights = weights

        me_map = merged.set_index("sid")["me_usd"].to_dict()
        last_holdings = [
            ActualHolding(
                security_id=p.security_id,
                weight=Decimal(str(p.target_weight)),
                market_cap_usd=Decimal(str(me_map.get(p.security_id, 0.0))),
                formation_market_cap_usd=Decimal(str(me_map.get(p.security_id, 0.0))),
            )
            for p in book.positions
        ]

    net_returns = [m["net_return"] for m in monthly]
    table = RequiredTable(
        ending_value=nav,
        starting_value=1_000_000_000.0,
        elapsed_years=len(monthly) / 12.0,
        periods_per_year=12,
        portfolio_returns=tuple(net_returns),
        benchmark_returns=tuple(m["benchmark_monthly"] for m in monthly),
        risk_free_returns=tuple(m["rf_monthly"] for m in monthly),
        gross_returns=tuple(m["gross_return"] for m in monthly),
        cost_returns=tuple(m["cost"] for m in monthly),
        borrow_returns=tuple(m["borrow"] for m in monthly),
        position_counts=tuple(m["names"] for m in monthly),
        gross_exposures=tuple(m["gross_exposure"] for m in monthly),
        net_exposures=tuple(m["net_exposure"] for m in monthly),
        turnover_per_rebalance=tuple(m["turnover"] for m in monthly),
    ).as_table()

    compliance_report = compliance.check(last_holdings) if last_holdings else None

    (OUT / "nav_path.json").write_text(json.dumps(nav_path, indent=2), encoding="utf-8")
    (OUT / "monthly.json").write_text(json.dumps(monthly, indent=2), encoding="utf-8")
    (OUT / "required_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (OUT / "coverage.json").write_text(json.dumps(coverage_log, indent=2), encoding="utf-8")
    if overlay_log:
        (OUT / "overlay_log.json").write_text(json.dumps(overlay_log, indent=2, default=str), encoding="utf-8")
    if compliance_report is not None:
        (OUT / "compliance.json").write_text(
            json.dumps(compliance_report.as_dict(), indent=2), encoding="utf-8"
        )

    avg_coverage = {
        name: sum(c["coverage"].get(name, 0) for c in coverage_log) / max(len(coverage_log), 1)
        for name in FAMILIES
    }
    print()
    label = "WITH RISK OVERLAY" if USE_RISK_OVERLAY else "no risk overlay"
    blend = "coverage-renormalized (challenger)" if BLEND_RENORM else "family-equal (C2 baseline)"
    print(f"=== seven sleeves (5 deterministic + 2 trained) — {label} | blend: {blend} ===")
    if overlay_log:
        scales = [o["risk_scale"] for o in overlay_log]
        edges = [o["edge_scale"] for o in overlay_log]
        flagged = sum(o["peer_flagged_shorts"] for o in overlay_log)
        capped = sum(o["runup_capped_shorts"] for o in overlay_log)
        print(f"overlay: avg risk_scale={sum(scales)/len(scales):.3f} avg edge_scale={sum(edges)/len(edges):.3f}")
        print(f"overlay: peer-flagged shorts={flagged} runup-capped={capped}")
    print(f"months evaluated        : {len(monthly)}")
    print(f"avg coverage per sleeve : " + ", ".join(f"{k}={v:.0f}" for k, v in avg_coverage.items()))
    print(f"ending NAV              : ${nav:,.0f}")
    print(f"cumulative return       : {table['cumulative_return']:.2%}")
    print(f"CAGR                    : {table['cagr']:.2%}")
    print(f"annualized volatility   : {table['annualized_volatility']:.2%}")
    print(f"max drawdown            : {table['max_drawdown']:.2%}")
    print(f"IR vs cash+4%           : {table['information_ratio']:.2f}")
    print(f"Sharpe                  : {table['sharpe_ratio']:.2f}")
    print(f"turnover per rebalance  : {table['turnover_per_rebalance_avg']:.2%}")
    print(f"annual cost % NAV       : {table['annual_cost_pct_nav']:.2%}")
    if compliance_report is not None:
        print(f"compliance official pass: {compliance_report.official_pass}")
    print(f"artifacts written to    : {OUT}")


if __name__ == "__main__":
    main()
