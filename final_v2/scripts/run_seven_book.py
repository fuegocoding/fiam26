"""Seven-sleeve run: five deterministic ports + two trained sleeves.

Trains xgb_rank (walk-forward on 147 chars) and filing_events (walk-forward
on EDGAR event counts with dated cik links), caches the forecasts, blends
all seven as families, and runs the engine end to end.

Hardened replay controls (pod_platform.orchestration.replay):
* sources need APPROVED formation-eligible cards (--source-cards FILE) or an
  explicit --research-only run, which is labeled and can never claim freeze;
* every period is compliance-checked (formation target, post-risk, post-trade
  actual, period-end drifted) with pre and post weights recorded;
* no missing-return zero-fill: a held name without an outcome stops the run
  unless dated liquidation evidence is supplied (--liquidations FILE);
* turnover/cost use drifted actual holdings on the real AccountingLedger;
* output goes to a fresh directory (--out-dir) or needs --overwrite.

Monthly inputs only: no daily path, quotes or liquidity realism is claimed.

Usage: python final_v2/scripts/run_seven_book.py (--source-cards FILE | --research-only)
           [--reuse-forecasts] [--out-dir DIR] [--overwrite] [--liquidations FILE]
           [--allow-terminal-unevaluated] [--no-hash-inputs]
           [--trial-count N] [--evaluation-access-count N]
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.accounting.ledger import BorrowSchedule  # noqa: E402
from pod_platform.allocation import (  # noqa: E402
    Candidate,
    ConstructionEngine,
    FamilyEstimate,
    FamilyIntegrator,
    InternalPolicy,
    OfficialLimits,
)
from pod_platform.contracts.base import canonical_digest  # noqa: E402
from pod_platform.contracts.errors import ErrorCode, PodPlatformError  # noqa: E402
from pod_platform.data.competition import CharsPanelLoader  # noqa: E402
from pod_platform.orchestration.replay import (  # noqa: E402
    MONTHLY_LIMITATIONS,
    ArtifactWriter,
    CoreSourceLineage,
    LiquidationEvidence,
    MonthlyAccountingReplay,
    PeriodComplianceLog,
    SourceEligibilityGate,
    account_period,
    assess_freeze,
    environment_lock_hash,
    fail,
    legacy_source_uses,
    load_source_cards,
    utc_close,
    validate_contiguous_periods,
)
from pod_platform.reporting.dossier import HoldingsRow, holdings_csv, returns_csv  # noqa: E402
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
from pod_platform.signals.short_interest import build_short_interest  # noqa: E402

REPO = ROOT.parent
PANEL = REPO / "data" / "chars_final_with_names.parquet"
FACTORS = REPO / "data" / "factor_char_list.csv"
EXTERNAL = REPO / "external_data" / "external.parquet"
FILINGS = REPO / "data" / "8k_20150101_20260831_identified.parquet"
CACHE = ROOT / "artifacts" / "seven_book" / "forecasts"

TEST_START = dt.date(2021, 1, 31)
TEST_END = dt.date(2026, 8, 31)
TEST_YEARS = list(range(2021, 2027))
LAST_TARGET = pd.Timestamp("2026-08-31")
COST_BP_PER_SIDE = 20.0
BORROW_ANNUAL = 0.05
START_NAV = Decimal("1000000000")
USE_RISK_OVERLAY = "--risk-overlay" in sys.argv
USE_GAP = "--with-gap" in sys.argv
BLEND_RENORM = "--blend-renorm" in sys.argv
USE_GRAPH = "--with-graph" in sys.argv
USE_DEMAND = "--with-demand" in sys.argv
GATE_E7_DEMAND = "--gate-e7-demand" in sys.argv
RESEARCH_ONLY = "--research-only" in sys.argv
OVERWRITE = "--overwrite" in sys.argv
ALLOW_TERMINAL = "--allow-terminal-unevaluated" in sys.argv
HASH_INPUTS = "--no-hash-inputs" not in sys.argv
_out_name = "seven_book"
if USE_RISK_OVERLAY:
    _out_name = "risk_book" + ("_renorm" if BLEND_RENORM else "") + ("_graph" if USE_GRAPH else "") + ("_demand" if USE_DEMAND else "") + ("_gate" if GATE_E7_DEMAND else "")


def _arg_value(flag: str):
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 >= len(sys.argv):
            raise SystemExit(f"{flag} needs a value")
        return sys.argv[index + 1]
    return None


SOURCE_CARDS = _arg_value("--source-cards")
LIQUIDATIONS = _arg_value("--liquidations")
TRIAL_COUNT = int(_arg_value("--trial-count") or 0)
EVAL_ACCESS_COUNT = int(_arg_value("--evaluation-access-count") or 0)
# Never default onto an existing artifact directory; a non-empty target needs --overwrite.
OUT = Path(_arg_value("--out-dir") or (ROOT / "artifacts" / f"{_out_name}_hardened"))
GRAPH_WORK = ROOT / "artifacts" / "graph_work"
GAP_WORK = ROOT / "artifacts" / "valuation_work"
FAMILIES = (
    "inventory_retail",
    "oplev_software",
    "runway_micro",
    "e7_surprise_breadth",
    "conf_revision_gated",
    "xgb_rank",
    "filing_events",
) + (("gap_micro",) if USE_GAP else ()) + (("demand",) if USE_DEMAND else ())


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
    if (xgb_path.exists() or filing_path.exists()) and not OVERWRITE:
        fail(
            ErrorCode.SCHEMA_MISMATCH,
            f"forecast cache already exists at {CACHE}",
            "Existing forecast artifacts are preserved: pass --reuse-forecasts or an explicit --overwrite.",
        )

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


def get_demand_map():
    """Combined demand signal: {(TICKER, formation date): scaled rank}.

    Equal rank-mean of card spend 12m YoY and foot traffic 3m YoY (the
    strongest belief features), formed at month+1 (T+3 lag), ranked
    cross-sectionally per formation month.
    """
    card = pd.read_parquet(ROOT / "artifacts" / "demand_card.parquet") if (ROOT / "artifacts" / "demand_card.parquet").exists() else None
    if card is None:
        card = pd.read_parquet(REPO / "data" / "carbon" / "raw" / "framework" / "card_panel_unified.parquet")
        ft = pd.read_parquet(REPO / "data" / "carbon" / "raw" / "framework" / "foot_traffic_panel.parquet")
        card = card[["ticker", "date", "credit_card_spend_12m_yoy"]].merge(
            ft[["ticker", "date", "foot_traffic_3m_yoy"]], on=["ticker", "date"], how="outer"
        )
        card["feature_month"] = pd.to_datetime(card["date"]) + pd.offsets.MonthEnd(1)
        out = ROOT / "artifacts" / "demand_card.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        card.to_parquet(out, index=False)
    g = card.dropna(subset=["feature_month"]).copy()
    g["combo"] = (
        g["credit_card_spend_12m_yoy"].rank(pct=True).fillna(0.5)
        + g["foot_traffic_3m_yoy"].rank(pct=True).fillna(0.5)
    )
    # rank within formation month across tickers
    g["scaled"] = (g.groupby("feature_month")["combo"].rank(pct=True) - 0.5) * (12 ** 0.5)
    return {
        (str(r.ticker).upper(), r.feature_month.date()): float(r.scaled)
        for r in g.dropna(subset=["scaled"]).itertuples()
    }


def get_graph_map():
    """3-seed ensemble of graph EPS forecasts: {(sid, formation date): pred}."""
    import glob as _glob

    files = sorted(_glob.glob(str(GRAPH_WORK / "graph_attention_*_seed*.parquet")))
    if not files:
        raise FileNotFoundError("graph forecasts not found - run scripts/train_graph.py first")
    frames = [pd.read_parquet(f) for f in files]
    allf = pd.concat(frames, ignore_index=True)
    ens = allf.groupby(["permno", "eom"], as_index=False)["pred_eps"].mean()
    return {(str(int(r.permno)), r.eom.date()): float(r.pred_eps) for r in ens.itertuples()}


def gate_sources() -> tuple:
    """Source eligibility gate + explicit lineage; runs before any heavy work."""
    if not SOURCE_CARDS and not RESEARCH_ONLY:
        fail(
            ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE,
            "no source cards supplied",
            "Pass --source-cards FILE for a formation run, or --research-only for a labeled exploratory run that cannot claim freeze.",
        )
    cards = load_source_cards(Path(SOURCE_CARDS)) if SOURCE_CARDS else {}
    components = list(FAMILIES) + ["short_interest", "risk_free_benchmark"]
    if GATE_E7_DEMAND:
        components += ["gate_e7_demand"]
    if USE_GRAPH:
        components += ["graph_tilt"]
    gate = SourceEligibilityGate(research_only=RESEARCH_ONLY).evaluate(legacy_source_uses(components, cards))
    lineage = CoreSourceLineage()
    for source_id, path, role in (
        ("core_chars_panel", PANEL, "core"),
        ("core_factor_list", FACTORS, "core"),
        ("external_parquet", EXTERNAL, "external"),
        ("filings_8k", FILINGS, "external"),
    ):
        if path.exists():
            lineage.add(source_id, path, role=role, hash_content=HASH_INPUTS)
    return gate, lineage


def load_liquidations() -> dict:
    """{formation date: {security_id: LiquidationEvidence}} from explicit dated evidence only."""
    if not LIQUIDATIONS:
        return {}
    out: dict = {}
    for record in json.loads(Path(LIQUIDATIONS).read_text(encoding="utf-8")):
        record = dict(record)
        day = dt.date.fromisoformat(record.pop("period_formation_date"))
        evidence = LiquidationEvidence.model_validate(record)
        out.setdefault(day, {})[evidence.security_id] = evidence
    return out


def main() -> None:
    reuse = "--reuse-forecasts" in sys.argv
    gate, lineage = gate_sources()  # fails closed before any training or heavy IO
    print(f"source policy: {gate.run_label}; blocked sources: {sorted(gate.blocked) or 'none'}", flush=True)
    writer = ArtifactWriter(OUT, overwrite=OVERWRITE)
    compliance = PeriodComplianceLog(OfficialLimits())
    monthly: list[dict] = []
    try:
        run(reuse, gate, lineage, writer, compliance, monthly)
    except Exception as error:  # persist evidence for a fail-closed stop, then re-raise
        code = error.response.error.code.value if isinstance(error, PodPlatformError) else type(error).__name__
        writer.write_json(
            "failure.json",
            {
                "status": "FAILED_CLOSED",
                "error": code,
                "summary": str(error),
                "periods_completed": len(monthly),
                "frozen": False,
                "run_label": gate.run_label,
            },
        )
        writer.write_json("compliance.partial.json", compliance.as_dict())
        writer.write_json("monthly.partial.json", monthly)
        raise


def run(reuse, gate, lineage, writer, compliance, monthly) -> None:
    loader = CharsPanelLoader(PANEL, FACTORS)
    panel = loader.load()
    panel["eom"] = pd.to_datetime(panel["eom"])
    signals = PortedSignals(EXTERNAL)
    tb3ms = load_tb3ms()
    liquidations_by_day = load_liquidations()

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
    graph_map = get_graph_map() if USE_GRAPH else {}
    demand_map = get_demand_map() if (USE_DEMAND or GATE_E7_DEMAND) else {}
    if USE_DEMAND or GATE_E7_DEMAND:
        print(f"demand scores: {len(demand_map):,}")

    si_table = build_short_interest(str(EXTERNAL), str(PANEL))
    si_map = {
        (str(int(r.permno)), r.return_month.date()): (
            float("nan") if pd.isna(r.short_pct) else float(r.short_pct)
        )
        for r in si_table.itertuples()
    }
    print(f"short-interest rows: {len(si_table):,}")
    xgb_map = {
        (str(int(r.permno)), r.eom.date()): float(r.pred_xgb_rank) for r in xgb_fc.itertuples()
    }
    filing_map = {
        (str(int(r.permno)), r.eom.date()): float(r.pred_filing_events) for r in filing_fc.itertuples()
    }

    integrator = FamilyIntegrator(FAMILIES, neutral_uncertainty=0.1)
    engine = ConstructionEngine(OfficialLimits(), InternalPolicy(target_positions=300))

    months = sorted({m for m in panel["eom"].dt.date.unique() if TEST_START <= m <= TEST_END})
    print(f"formation months: {len(months)} ({months[0]} .. {months[-1]})")

    # Every formation month must have realized outcomes; none is skipped silently.
    labelled = panel_full.groupby("eom")["ret_exc_lead1m"].apply(lambda s: int(s.notna().sum()))
    unlabelled = [m for m in months if int(labelled.get(pd.Timestamp(m), 0)) == 0]
    excluded_terminal: list[str] = []
    if unlabelled:
        if ALLOW_TERMINAL and unlabelled == [months[-1]]:
            excluded_terminal = [str(months[-1])]
            months = months[:-1]
            print(f"explicitly excluding unevaluable terminal formation month {excluded_terminal[0]}", flush=True)
        else:
            fail(
                ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                f"no realized outcomes for formation months: {', '.join(str(m) for m in unlabelled)}",
                "Months without outcomes are never skipped silently; shorten TEST_END or pass --allow-terminal-unevaluated for the final month only.",
            )
    validate_contiguous_periods([(m, (pd.Timestamp(m) + pd.offsets.MonthEnd(1)).date()) for m in months])

    replay = MonthlyAccountingReplay(
        START_NAV,
        borrow_schedule=BorrowSchedule(
            schedule_id="legacy_flat_borrow",
            large_cap_threshold_usd=Decimal("0"),
            large_cap_annual_rate=Decimal(str(BORROW_ANNUAL)),
            other_annual_rate=Decimal(str(BORROW_ANNUAL)),
        ),
    )
    nav_path = [{"month": "start", "nav": float(replay.nav())}]
    holdings_rows: list[list] = []
    caps: dict[str, Decimal] = {}
    identities: dict[str, tuple[str, str]] = {}
    coverage_log = []
    score_history: list[pd.DataFrame] = []
    overlay_log = []
    last_drifted: dict = {}

    for month_date in months:
        month_ts = pd.Timestamp(month_date)
        month_df, _ = loader.formation_month(
            dt.datetime(month_date.year, month_date.month, month_date.day, tzinfo=dt.timezone.utc)
        )
        if len(month_df) < 250:
            fail(
                ErrorCode.INSUFFICIENT_HISTORY,
                f"formation universe too small at {month_date}: {len(month_df)} rows",
                "A thin month is reported, never skipped.",
            )

        ret_date = (month_ts + pd.offsets.MonthEnd(1)).date()
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
        if merged["ret_exc_lead1m"].notna().sum() == 0:
            fail(
                ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                f"no forward labels for the return month after {month_date}",
                "A formation month without outcomes is never skipped.",
            )
        keys = [(sid, month_date) for sid in merged["sid"]]
        merged["xgb_rank"] = [xgb_map.get(k) for k in keys]
        merged["filing_events"] = [filing_map.get(k) for k in keys]
        if USE_DEMAND:
            merged["demand"] = [
                demand_map.get((str(t).upper(), month_date), float("nan"))
                for t in merged["ticker"]
            ]
            dv = merged["demand"]
            scaled["demand"] = dv.where(dv.notna()).fillna(0.0)
            coverage["demand"] = int(dv.notna().sum())
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

        if GATE_E7_DEMAND:
            e7s = scaled["e7_surprise_breadth"]
            gate_dropped = 0
            for idx in merged.index:
                e7 = float(e7s.get(idx, 0.0))
                d = demand_map.get((str(merged.loc[idx, "ticker"]).upper(), month_date))
                if e7 != 0.0 and d is not None and d != 0.0 and (e7 > 0) != (d > 0):
                    e7s[idx] = 0.0
                    gate_dropped += 1
            coverage["e7_gate_dropped"] = gate_dropped

        raw_alpha = {}
        for sid, idx in zip(merged["sid"], merged.index):
            fams = []
            for name in FAMILIES:
                value = float(scaled[name].get(idx, 0.0))
                if value != 0.0:
                    fams.append(FamilyEstimate(name, sid, value, 0.1))
            if not fams:
                continue
            raw_alpha[sid] = sum(f.expected_excess_return for f in fams) / (
                len(fams) if BLEND_RENORM else len(FAMILIES)
            )

        candidates_alpha = {}
        if USE_GRAPH and raw_alpha:
            sr = pd.Series(raw_alpha)
            sr_rank = sr.rank(pct=True)
            gv = pd.Series({sid: graph_map.get((sid, month_date)) for sid in raw_alpha})
            covered = gv.notna()
            tilted = sr_rank.copy()
            if covered.any():
                gr_rank = gv[covered].rank(pct=True)
                tilted[covered] = 0.95 * sr_rank[covered] + 0.05 * gr_rank
            candidates_alpha = (tilted - 0.5).to_dict()
        else:
            candidates_alpha = dict(raw_alpha)

        # Incumbents are the drifted actual holdings, not raw prior target weights.
        held_before = replay.held()
        candidates = []
        for sid, idx in zip(merged["sid"], merged.index):
            if sid not in candidates_alpha:
                continue
            alpha = candidates_alpha[sid]
            row = merged.loc[idx]
            beta_value = row.get("beta_60m")
            si_value = si_map.get((sid, ret_date))
            candidates.append(
                Candidate(
                    security_id=sid,
                    alpha=alpha,
                    market_cap_usd=float(row["me_usd"]),
                    tradable=True,
                    incumbent=sid in held_before,
                    sector=str(row["gics"])[:2] if row["gics"] else "unknown",
                    beta=float(beta_value) if beta_value == beta_value and beta_value is not None else 1.0,
                    short_interest_pct=si_value if si_value is not None else float("nan"),
                )
            )
        if len(candidates) < 250:
            fail(
                ErrorCode.INSUFFICIENT_HISTORY,
                f"too few candidates at {month_date}: {len(candidates)}",
                "A thin candidate set is reported, never skipped.",
            )

        try:
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
        except PodPlatformError:
            neg = sum(1 for c in candidates if c.alpha < 0)
            si_blocked = sum(1 for c in candidates if c.alpha < 0 and c.short_interest_pct == c.short_interest_pct and c.short_interest_pct >= 0.15)
            print(f"construction failed at {month_date}: candidates={len(candidates)} neg_alpha={neg} si_blocked={si_blocked}", flush=True)
            raise

        construction_weights = {p.security_id: p.target_weight for p in book.positions}
        post_risk_weights = {k: float(v) for k, v in construction_weights.items()}

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

        if USE_RISK_OVERLAY and post_risk_weights:
            scores_df = pd.DataFrame(score_history)
            overlay_result = protect(post_risk_weights, ret_month, overlay_panel, scores_df)
            post_risk_weights = dict(overlay_result.weights)
            overlay_log.append({"month": str(month_date), **overlay_result.summary})

        # Caps for every name ever held or targeted (finite only; compliance fails closed otherwise).
        for sid, cap in zip(merged["sid"], merged["me_usd"]):
            if cap == cap and cap is not None and float(cap) not in (float("inf"), float("-inf")):
                caps[sid] = Decimal(str(float(cap)))

        month_key = f"{month_date.year}-{month_date.month:02d}"
        if month_key not in tb3ms:
            fail(
                ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                f"TB3MS risk-free rate missing for {month_key}",
                "The risk-free input is never zero-filled.",
            )
        rf_m = tb3ms[month_key] / 1200.0
        bench_m = rf_m + 0.04 / 12.0

        accounted = account_period(
            replay,
            compliance,
            period=str(month_date),
            formation_at=utc_close(month_date),
            end_at=utc_close(ret_date),
            construction_weights=construction_weights,
            post_risk_weights=post_risk_weights,
            outcomes=merged.set_index("sid")["ret_exc_lead1m"].to_dict(),
            liquidations=liquidations_by_day.get(month_date),
            rf_monthly=rf_m,
            cost_bp_per_side=COST_BP_PER_SIDE,
            market_caps=caps,
        )
        trade, result = accounted.trade, accounted.result
        last_drifted = result.drifted_weights
        nav_path.append({"month": str(month_date), "nav": float(result.nav_end)})
        monthly.append(
            {
                "month": str(month_date),
                "gross_return": result.gross_return,
                "net_return": result.net_return,
                "rf_monthly": rf_m,
                "benchmark_monthly": bench_m,
                "active_return": result.active_return,
                "borrow": result.borrow_return,
                "cost": result.cost_return,
                "turnover": trade.turnover_one_way,
                "names": result.names,
                "gross_exposure": result.gross_exposure,
                "net_exposure": result.net_exposure,
                "liquidated": result.liquidated,
                "nav_residuals": result.reconciliation,
            }
        )
        for sid, weight in sorted(trade.post_weights.items()):
            holdings_rows.append([str(month_date), sid, str(weight), "long" if weight > 0 else "short"])
        id_rows = merged.set_index("sid")
        for sid in trade.post_weights:
            row = id_rows.loc[sid]
            ticker = row.get("ticker")
            name = row.get("comp_name", row.get("name", ""))
            identities[sid] = (str(ticker) if ticker == ticker and ticker is not None else sid, str(name) if name == name and name is not None else "", str(month_date))

    net_returns = [m["net_return"] for m in monthly]
    table = RequiredTable(
        ending_value=float(replay.nav()),
        starting_value=float(START_NAV),
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

    final_rows = []
    for sid, weight in sorted(last_drifted.items()):
        ticker, name, ref = identities[sid]
        final_rows.append(
            HoldingsRow(sid, ticker, name, PANEL.name, ref, float(replay.ledger.positions[sid].quantity), float(weight), "long" if weight > 0 else "short")
        )

    config = {
        "families": list(FAMILIES),
        "test_start": str(TEST_START),
        "test_end": str(TEST_END),
        "cost_bp_per_side": COST_BP_PER_SIDE,
        "borrow_annual": BORROW_ANNUAL,
        "risk_overlay": USE_RISK_OVERLAY,
        "blend_renorm": BLEND_RENORM,
        "with_graph": USE_GRAPH,
        "with_gap": USE_GAP,
        "with_demand": USE_DEMAND,
        "gate_e7_demand": GATE_E7_DEMAND,
        "target_positions": 300,
        "research_only": RESEARCH_ONLY,
        "excluded_terminal_months": excluded_terminal,
    }
    writer.write_json("nav_path.json", nav_path)
    writer.write_json("monthly.json", monthly)
    writer.write_json("required_table.json", table)
    writer.write_json("coverage.json", coverage_log)
    if overlay_log:
        writer.write_json("overlay_log.json", overlay_log)
    writer.write_json("compliance.json", compliance.as_dict())
    writer.write_text("returns_export.csv", returns_csv([(m["month"], m["net_return"]) for m in monthly]))
    writer.write_text("holdings_export.csv", holdings_csv(final_rows))
    lines = ["period,security_id,post_trade_weight,side"] + [",".join(r) for r in holdings_rows]
    writer.write_text("holdings_by_period.csv", "\n".join(lines) + "\n")
    writer.write_json("lineage.json", lineage.as_dict())

    assessment = assess_freeze(
        manifest=None,  # the legacy path declares no security-master/graph versions or certificates
        outputs=dict(writer.payloads),
        gate=gate,
        certificate_count=0,
        deployed_forecast_count=len(FAMILIES),
        accounting_reconciled=True,  # advance() raises on any NAV residual; reaching here is measured
        compliance=compliance,
        trial_count=TRIAL_COUNT,
        evaluation_access_count=EVAL_ACCESS_COUNT,
        frozen_at=utc_close(months[-1]),
    )
    writer.write_json(
        "run_status.json",
        {
            "labels": {
                "run_kind": "replay",
                "evidence_class": "legacy_monthly_replay",
                "trial_count_declared": TRIAL_COUNT,
                "evaluation_access_count_declared": EVAL_ACCESS_COUNT,
            },
            "freeze": assessment.as_dict(),
            "source_policy": gate.as_dict(),
            "config_hash": canonical_digest(config),
            "config": config,
            "environment_lock_hash": environment_lock_hash(),
            "python": sys.version.split()[0],
            "input_hashed": HASH_INPUTS,
            "periods": len(monthly),
            "excluded_terminal_months": excluded_terminal,
            "internal_warnings": compliance.internal_warnings(),
            "liquidations_supplied": bool(liquidations_by_day),
            "limitations": list(MONTHLY_LIMITATIONS),
            "llm_called": False,
        },
    )
    writer.write_json("artifact_hashes.json", dict(sorted(writer.hashes.items())))

    avg_coverage = {
        name: sum(c["coverage"].get(name, 0) for c in coverage_log) / max(len(coverage_log), 1)
        for name in FAMILIES
    }
    print()
    label = "WITH RISK OVERLAY" if USE_RISK_OVERLAY else "no risk overlay"
    blend = "coverage-renormalized (challenger)" if BLEND_RENORM else "family-equal (C2 baseline)"
    print(f"=== seven sleeves (5 deterministic + 2 trained) — {label} | blend: {blend} ===")
    print(f"run label               : {gate.run_label}  freeze: {assessment.status}")
    for reason in assessment.reasons:
        print(f"  not frozen because    : {reason}")
    if overlay_log:
        scales = [o["risk_scale"] for o in overlay_log]
        edges = [o["edge_scale"] for o in overlay_log]
        flagged = sum(o["peer_flagged_shorts"] for o in overlay_log)
        capped = sum(o["runup_capped_shorts"] for o in overlay_log)
        print(f"overlay: avg risk_scale={sum(scales)/len(scales):.3f} avg edge_scale={sum(edges)/len(edges):.3f}")
        print(f"overlay: peer-flagged shorts={flagged} runup-capped={capped}")
    print(f"months evaluated        : {len(monthly)}")
    print(f"avg coverage per sleeve : " + ", ".join(f"{k}={v:.0f}" for k, v in avg_coverage.items()))
    print(f"ending NAV              : ${float(replay.nav()):,.0f}")
    print(f"cumulative return       : {table['cumulative_return']:.2%}")
    print(f"CAGR                    : {table['cagr']:.2%}")
    print(f"annualized volatility   : {table['annualized_volatility']:.2%}")
    print(f"max drawdown            : {table['max_drawdown']:.2%}")
    print(f"IR vs cash+4%           : {table['information_ratio']:.2f}")
    print(f"Sharpe                  : {table['sharpe_ratio']:.2f}")
    print(f"turnover per rebalance  : {table['turnover_per_rebalance_avg']:.2%}")
    print(f"annual cost % NAV       : {table['annual_cost_pct_nav']:.2%}")
    print(f"compliance official pass: {compliance.all_official_pass} ({len(compliance.records)} period-stage checks)")
    print(f"artifacts written to    : {OUT}")


if __name__ == "__main__":
    main()
