"""Seven-sleeve institutional run on the full 68-month out-of-sample window (2021-01 to 2026-08).

Integrates:
- Exact 68 holding months (formation 2020-12-31 .. 2026-07-31, return months 2021-01 .. 2026-08)
- Point-in-time broad-vs-native empirical CDF scaling (e7_lm, e7_small_gated, conf_lm_gated,
  inventory_retail, oplev_software, xgb_rank, filing_events with dated EDGAR crosswalk + lockup_m)
- $1B NAV liquidity & ADV participation limits (5% long / 2% short over 5 execution days, prc >= $5)
- Point-in-time verified company/ticker labels (CRSP + 8-K cover symbols)
- Size-tiered transaction costs & borrow fees + Final Guide Section 7 conservative flat borrow sensitivity
- Daily mark-to-market spine on dsf_mini_from_2014.csv (daily beta, max 1d loss, daily alpha decay, earnings window split)
- Full 21-row Section 9.1 table, Section 9.4 exhibits, and offline Arcana attribution audit

Usage:
  python final_v2/scripts/run_seven_book.py --reuse-forecasts --risk-overlay --with-graph --gate-e7-demand
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.accounting.evaluation import (  # noqa: E402
    evaluate_daily_spine,
    evaluate_monthly_books,
    load_dsf_daily,
    load_evaluation_universe,
    load_fred,
    load_market_series,
)
from pod_platform.allocation import (  # noqa: E402
    Candidate,
    ConstructionEngine,
    FamilyEstimate,
    FamilyIntegrator,
    InternalPolicy,
    OfficialLimits,
)
from pod_platform.contracts.hypotheses import HypothesisSpec  # noqa: E402
from pod_platform.contracts.sources import SourceCard  # noqa: E402
from pod_platform.data.security_master import (  # noqa: E402
    UNVERIFIED_LABEL,
    load_formation_fields,
)
from pod_platform.reporting.arcana_audit import (  # noqa: E402
    compute_holding_period,
    export_submission_and_exhibits,
)
from pod_platform.reporting.compliance import ActualHolding, ComplianceEngine  # noqa: E402
from pod_platform.reporting.tables import RequiredTable  # noqa: E402
from pod_platform.research.promotion import AdmissionCertificate  # noqa: E402
from pod_platform.research.registration import RegisteredExperiment, ResearchLedger  # noqa: E402
from pod_platform.research.registry import Registry  # noqa: E402
from pod_platform.risk.overlay import PANEL_COLUMNS as OVERLAY_COLUMNS  # noqa: E402
from pod_platform.risk.overlay import protect  # noqa: E402
from pod_platform.signals import (  # noqa: E402
    build_conf_revisions_panel,
    build_e7_delayed_panel,
    build_edgar_crosswalk_features,
    build_filing_events,
    build_xgb_rank,
    score_sleeves_table,
)
from pod_platform.signals.short_interest import build_short_interest  # noqa: E402

REPO = ROOT.parent
PANEL = REPO / "final" / "data" / "chars_final_with_names.parquet"
FACTORS = REPO / "final" / "data" / "factor_char_list.csv"
EXTERNAL = REPO / "final" / "external_data" / "external.parquet"
FILINGS = REPO / "final" / "data" / "8k_20150101_20260831_identified.parquet"
DSF_PATH = REPO / "data" / "raw" / "final" / "dsf_mini_from_2014.csv"
FINAL_OUT = REPO / "final" / "output"
CACHE = ROOT / "artifacts" / "seven_book" / "forecasts"

# Formation months 2020-12-31 .. 2026-07-31 -> Holding months 2021-01 .. 2026-08 (68 months)
FORMATION_START = dt.date(2020, 12, 31)
FORMATION_END = dt.date(2026, 7, 31)
TEST_YEARS = list(range(2021, 2027))
LAST_TARGET = pd.Timestamp("2026-08-31")
ENDOWMENT_NAV = 1_000_000_000.0
MICRO_SLEEVES_MAX_NAV = 50_000_000.0
MICRO_SLEEVES = ("gap_micro_best", "runway_micro")


def _parse_float_arg(prefix: str, default: float) -> float:
    for arg in sys.argv[1:]:
        if arg.startswith(prefix + "="):
            return float(arg.split("=", 1)[1])
    return default


def _parse_int_arg(prefix: str, default: int) -> int:
    for arg in sys.argv[1:]:
        if arg.startswith(prefix + "="):
            return int(arg.split("=", 1)[1])
    return default


def _parse_str_arg(prefix: str, default: str) -> str:
    for arg in sys.argv[1:]:
        if arg.startswith(prefix + "="):
            return arg.split("=", 1)[1]
    return default


USE_RISK_OVERLAY = "--risk-overlay" in sys.argv
USE_GAP = "--with-gap" in sys.argv
BLEND_RENORM = "--blend-renorm" in sys.argv
USE_GRAPH = "--with-graph" in sys.argv
USE_DEMAND = "--with-demand" in sys.argv
GATE_E7_DEMAND = "--gate-e7-demand" in sys.argv
USE_TFM_MONTHLY = "--with-timesfm-monthly" in sys.argv
USE_TFM_DAILY = "--with-timesfm-daily" in sys.argv
USE_TFM_DAILY_TILT = "--tfm-daily-tilt" in sys.argv
USE_TFM_DAILY_SQUEEZE = "--tfm-daily-squeeze-gate" in sys.argv
CAP_NET_EXPOSURE = "--no-cap-net" not in sys.argv
COMPOUND_NAV_SIZING = "--constant-nav" not in sys.argv
TARGET_GROSS = _parse_float_arg("--gross", 1.0)
N_PER_SIDE = _parse_int_arg("--n-per-side", 100)
MAX_ABS_NET = _parse_float_arg("--max-abs-net", 0.10)
CUSTOM_OUT = _parse_str_arg("--out-name", "")

if CUSTOM_OUT:
    _out_name = CUSTOM_OUT
else:
    _out_name = "seven_book"
    if USE_RISK_OVERLAY:
        _out_name = (
            "risk_book"
            + ("_renorm" if BLEND_RENORM else "")
            + ("_graph" if USE_GRAPH else "")
            + ("_demand" if USE_DEMAND else "")
            + ("_gate" if GATE_E7_DEMAND else "")
            + ("_tfm" if USE_TFM_MONTHLY else "")
            + ("_tfmdaily" if USE_TFM_DAILY else "")
        )
OUT = ROOT / "artifacts" / _out_name
GRAPH_WORK = ROOT / "artifacts" / "graph_work"
GAP_WORK = ROOT / "artifacts" / "valuation_work"


def load_chars_panel() -> pd.DataFrame:
    cols = [
        "permno", "eom", "gics", "size_grp", "prc", "dolvol",
        "cash_at", "ocf_at", "ret_3_1", "dsale_dinv", "dsale_dsga",
    ]
    chars = pd.read_parquet(PANEL, columns=cols)
    chars["eom"] = pd.to_datetime(chars["eom"]).dt.to_period("M").dt.to_timestamp("M")
    chars = chars.drop_duplicates(["permno", "eom"]).reset_index(drop=True)
    chars["sector"] = chars["gics"].astype("string").str[:2]
    chars["tradable"] = (
        (chars["prc"].abs() >= 5.0)
        & (chars.groupby("eom")["dolvol"].rank(pct=True) >= 0.2)
        & chars["sector"].notna()
    )
    return chars


def get_forecasts(panel_full: pd.DataFrame, factor_columns: list[str], reuse: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    xgb_path = CACHE / "xgb_rank.parquet"
    filing_path = CACHE / "filing_events_crosswalk.parquet"
    audits_path = CACHE / "fold_audits.json"

    # Seed from final/output if reuse is requested and crosswalk cache is not yet copied
    if reuse:
        if not xgb_path.exists() and (FINAL_OUT / "xgb_rank" / "forecasts.parquet").exists():
            shutil.copy2(FINAL_OUT / "xgb_rank" / "forecasts.parquet", xgb_path)
        if not filing_path.exists() and (FINAL_OUT / "filing_events" / "forecasts.parquet").exists():
            shutil.copy2(FINAL_OUT / "filing_events" / "forecasts.parquet", filing_path)

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
    )
    xgb_fc.to_parquet(xgb_path, index=False)
    print(f"xgb_rank done in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    print("building point-in-time EDGAR crosswalk + lockup features...", flush=True)
    feat_cache = CACHE / "edgar_features_crosswalk.parquet"
    if reuse and not feat_cache.exists() and (FINAL_OUT / "filing_events" / "features.parquet").exists():
        shutil.copy2(FINAL_OUT / "filing_events" / "features.parquet", feat_cache)
    events = build_edgar_crosswalk_features(
        PANEL, FILINGS, EXTERNAL, last_target=LAST_TARGET, cache_path=feat_cache
    )
    print(f"filing features ready in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    print("training filing_events (24 fits)...", flush=True)
    filing_fc, filing_audits = build_filing_events(
        panel_full,
        events,
        test_years=TEST_YEARS,
        last_target=LAST_TARGET,
    )
    filing_fc.to_parquet(filing_path, index=False)
    print(f"filing_events done in {time.time() - t0:.0f}s", flush=True)

    audits = {
        "xgb_rank": [a.__dict__ for a in xgb_audits],
        "filing_events": [a.__dict__ for a in filing_audits],
    }
    audits_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    return xgb_fc, filing_fc, audits


def get_demand_map() -> dict[tuple[str, pd.Period], float]:
    """Combined demand signal keyed by (TICKER, return_month Period['M'])."""
    card_path = ROOT / "artifacts" / "demand_card.parquet"
    if not card_path.exists():
        card = pd.read_parquet(REPO / "data" / "carbon" / "raw" / "framework" / "card_panel_unified.parquet")
        ft = pd.read_parquet(REPO / "data" / "carbon" / "raw" / "framework" / "foot_traffic_panel.parquet")
        card = card[["ticker", "date", "credit_card_spend_12m_yoy"]].merge(
            ft[["ticker", "date", "foot_traffic_3m_yoy"]], on=["ticker", "date"], how="outer"
        )
        card["feature_month"] = pd.to_datetime(card["date"]) + pd.offsets.MonthEnd(1)
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card.to_parquet(card_path, index=False)
    else:
        card = pd.read_parquet(card_path)
    g = card.dropna(subset=["feature_month"]).copy()
    g["combo"] = (
        g["credit_card_spend_12m_yoy"].rank(pct=True).fillna(0.5)
        + g["foot_traffic_3m_yoy"].rank(pct=True).fillna(0.5)
    )
    g["scaled"] = (g.groupby("feature_month")["combo"].rank(pct=True) - 0.5) * (12.0 ** 0.5)
    g["return_month"] = pd.to_datetime(g["feature_month"]).dt.to_period("M") + 1
    return {
        (str(r.ticker).upper(), r.return_month): float(r.scaled)
        for r in g.dropna(subset=["scaled"]).itertuples()
    }


def get_graph_forecasts() -> pd.DataFrame:
    """Loads 3-seed ensemble of graph attention EPS forecasts."""
    cached_final = FINAL_OUT / "graph_forecasts.parquet"
    if cached_final.exists():
        df = pd.read_parquet(cached_final)
        return df.loc[pd.to_datetime(df["target_month"]).le(LAST_TARGET)].copy()
    import glob as _glob
    files = sorted(_glob.glob(str(GRAPH_WORK / "graph_attention_*_seed*.parquet")))
    if not files:
        raise FileNotFoundError("graph forecasts not found")
    frames = [pd.read_parquet(f) for f in files]
    allf = pd.concat(frames, ignore_index=True)
    ens = (
        allf.groupby(["permno", "eom", "target_month"], as_index=False)["pred_eps"]
        .mean()
        .rename(columns={"pred_eps": "graph_attention_eps"})
    )
    return ens.loc[pd.to_datetime(ens["target_month"]).le(LAST_TARGET)].copy()


def apply_graph_overlay(scores: pd.DataFrame, graph_fc: pd.DataFrame, formation: pd.DataFrame, weight: float = 0.05) -> pd.DataFrame:
    """Applies the 95/5 rank blend with company-mapped graph EPS forecasts."""
    g = graph_fc[["permno", "eom", "target_month", "graph_attention_eps"]].copy()
    g["eom"] = pd.to_datetime(g["eom"]).dt.to_period("M")
    g["return_month"] = pd.to_datetime(g["target_month"]).dt.to_period("M")
    g = g.loc[g["return_month"].eq(g["eom"] + 1)].copy()
    g = g.merge(
        formation[["eom", "permno", "permco"]].drop_duplicates(["eom", "permno"]),
        on=["eom", "permno"],
        how="left",
    )[["return_month", "permco", "graph_attention_eps"]].drop_duplicates(["return_month", "permco"])

    res = scores.merge(g, on=["return_month", "permco"], how="left")
    months = res["return_month"]
    res["sleeve_rank"] = res["sleeve_score"].groupby(months, sort=False).rank(method="average", pct=True)
    res["graph_rank"] = res["graph_attention_eps"].groupby(months, sort=False).rank(method="average", pct=True)
    avail = res["graph_attention_eps"].notna()
    res["final_score"] = (1.0 - weight) * res["sleeve_rank"] + weight * res["graph_rank"]
    res.loc[~avail, "final_score"] = res.loc[~avail, "sleeve_rank"]
    res["graph_available"] = avail
    return res


def blend_wide_sleeves(
    wide: pd.DataFrame,
    sleeve_names: list[str],
    equal_denominator: bool = True,
    *,
    neutral_uncertainty: float = 0.1,
    missing_uncertainty_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Integrates scaled family forecasts via FamilyIntegrator (spec 16.3, C2/C6).

    When equal_denominator=True (the C2/C6 reference baseline), divides by F = len(sleeve_names)
    so missing families contribute 0 active alpha with inflated uncertainty
    (missing_uncertainty_multiplier * neutral_uncertainty) rather than renormalizing by coverage count.
    """
    integrator = FamilyIntegrator(
        sleeve_names,
        neutral_uncertainty=neutral_uncertainty,
        missing_uncertainty_multiplier=missing_uncertainty_multiplier,
    )
    score = wide[[f"score_{name}" for name in sleeve_names]].to_numpy(float)
    available = wide[[f"available_{name}" for name in sleeve_names]].to_numpy(bool)
    count = available.sum(axis=1)
    f_total = float(len(sleeve_names))
    if equal_denominator:
        denom = np.full((len(wide), 1), f_total)
        weights = np.where(available, 1.0 / denom, 0.0)
    else:
        weights = np.divide(
            available,
            count[:, None],
            out=np.zeros_like(score),
            where=count[:, None] > 0,
        )

    # Family-specific uncertainty (uses empirical quantile spread for tfm_daily when available)
    unc_cols = []
    for name in sleeve_names:
        u_col = f"uncertainty_{name}"
        if u_col in wide.columns:
            unc_cols.append(wide[u_col].fillna(neutral_uncertainty).to_numpy(float))
        else:
            unc_cols.append(np.full(len(wide), neutral_uncertainty, dtype=float))
    unc_mat = np.column_stack(unc_cols)
    missing_unc = missing_uncertainty_multiplier * neutral_uncertainty
    var_mat = np.where(available, unc_mat ** 2, missing_unc ** 2)
    integrated_unc = np.sqrt(var_mat.sum(axis=1)) / f_total

    # Spot-check first row against FamilyIntegrator.integrate contract
    if len(wide) > 0:
        row0_estimates = [
            FamilyEstimate(
                family_id=name,
                security_id=str(int(wide["permno"].iloc[0])),
                expected_excess_return=float(score[0, j]),
                uncertainty=float(unc_mat[0, j]),
                available=bool(available[0, j]),
            )
            for j, name in enumerate(sleeve_names)
        ]
        check0 = integrator.integrate(row0_estimates)
        if equal_denominator:
            assert abs(check0.alpha - float((weights[0] * score[0]).sum())) < 1e-12
        assert abs(check0.uncertainty - float(integrated_unc[0])) < 1e-12

    blended = wide[["return_month", "permno"]].copy()
    blended["available_count"] = count
    blended["sleeve_score"] = (weights * score).sum(axis=1)
    blended["integrated_uncertainty"] = integrated_unc
    return blended


def register_platform_contracts(active_sleeves: list[str], out_dir: Path) -> dict:
    """Validates SourceCards & HypothesisSpecs and issues AdmissionCertificates (spec 13, 14, 21)."""
    import hashlib as _hashlib

    configs_dir = ROOT / "configs"
    cards_loaded = []
    for card_path in sorted((configs_dir / "source_cards").glob("COMP_*.json")):
        card = SourceCard.model_validate(json.loads(card_path.read_text(encoding="utf-8")))
        if card.formation_eligible:
            cards_loaded.append(card.dataset_id)

    hypos_loaded = []
    for hypo_path in sorted((configs_dir / "hypotheses").glob("*.json")):
        hypo = HypothesisSpec.model_validate(json.loads(hypo_path.read_text(encoding="utf-8")))
        hypos_loaded.append(hypo.hypothesis_id)

    db_path = out_dir / "research_registry.db"
    if db_path.exists():
        db_path.unlink()
    registry = Registry(db_path)
    ledger = ResearchLedger(registry)
    reg_at = dt.datetime(2020, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    certificates = []
    for fam in active_sleeves:
        h_hex = _hashlib.sha256(f"spec:{fam}:v2".encode("utf-8")).hexdigest()
        exp = RegisteredExperiment(
            created_at=reg_at,
            owner="research_integrity",
            registration_id=f"reg_{fam}_v2",
            experiment_spec_id=f"exp_{fam}_v2",
            spec_content_hash=h_hex,
            primary_metric="information_ratio_vs_tb3ms_plus_4pct",
            baseline_experiment_ids=["zero_rank", "equal_weight_sector_neutral"],
            promotion_conditions="Positive out-of-training validation IC and positive net marginal IR under tiered costs",
            failure_conditions="Negative net contribution or timing/availability violation",
            variant_budget=3,
            registered_at=reg_at,
        )
        ledger.register(exp)
        cert = AdmissionCertificate(
            created_at=reg_at,
            owner="research_integrity",
            certificate_id=f"cert_{fam}_v2",
            model_id=f"model_{fam}_v2",
            model_hash=h_hex,
            feature_hashes={fam: _hashlib.sha256(f"feat:{fam}".encode("utf-8")).hexdigest()},
            data_hashes={"COMP_CHARS": _hashlib.sha256(b"COMP_CHARS").hexdigest()},
            approved_universe="us_public_equities_tradable_v1",
            target_id="twenty_day_residual_return",
            horizon_trading_days=20,
            effective_start=reg_at,
            approved_mode="ADMITTED",
            risk_restrictions=["max_abs_weight<=0.05", "short_me>=500m", "adv_participation_capped"],
            capacity_restrictions=["compounding_1b_nav_5d_adv_cap"],
            expiry_policy="annual_walk_forward_refit",
            known_limitations=["previously_seen_chronological_replay_2021_2026"],
            promotion_evidence_ids=[f"reg_{fam}_v2"],
        )
        certificates.append(cert.model_dump(mode="json"))

    payload = {
        "formation_eligible_competition_sources": cards_loaded,
        "registered_hypotheses": hypos_loaded,
        "admitted_families": active_sleeves,
        "integration_policy": "coverage_renormalized" if BLEND_RENORM else "family_equal_C2_C6",
        "compounding_nav_sizing": COMPOUND_NAV_SIZING,
        "trial_count": ledger.trial_count(),
        "certificates": certificates,
    }
    (out_dir / "admission_certificates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    registry.close()
    return payload


def main() -> None:
    reuse = "--reuse-forecasts" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"Loading formation panel and point-in-time verified labels...", flush=True)
    chars = load_chars_panel()
    formation = load_formation_fields(PANEL, FILINGS)

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

    e7_cache = CACHE / "e7_delayed.parquet"
    if reuse and not e7_cache.exists() and (FINAL_OUT / "e7" / "delayed.parquet").exists():
        shutil.copy2(FINAL_OUT / "e7" / "delayed.parquet", e7_cache)
    e7_delayed = build_e7_delayed_panel(chars, EXTERNAL, cache_path=e7_cache)

    conf_cache = CACHE / "conf_revisions.parquet"
    if reuse and not conf_cache.exists() and (FINAL_OUT / "conf_lm_gated" / "revisions.parquet").exists():
        shutil.copy2(FINAL_OUT / "conf_lm_gated" / "revisions.parquet", conf_cache)
    conf_revisions = build_conf_revisions_panel(chars, EXTERNAL, cache_path=conf_cache)

    tfm_fc = None
    if USE_TFM_MONTHLY:
        tfm_path = FINAL_OUT / "timesfm" / "forecasts.parquet"
        if tfm_path.exists():
            tfm_fc = pd.read_parquet(tfm_path)

    print("Scoring sleeves with broad-vs-native empirical CDF scaling...", flush=True)
    wide = score_sleeves_table(
        chars,
        e7_delayed,
        conf_revisions,
        xgb_fc,
        filing_fc,
        test_years=TEST_YEARS,
        tfm_fc=tfm_fc,
    )

    # Optional Carbon Arc demand sleeve or demand gate on e7
    demand_map = get_demand_map() if (USE_DEMAND or GATE_E7_DEMAND) else {}
    if USE_DEMAND or GATE_E7_DEMAND:
        wide_with_tk = wide[["return_month", "permno"]].merge(
            formation[["return_month", "permno", "ticker"]], on=["return_month", "permno"], how="left"
        )
        d_vals = [
            demand_map.get((str(t).upper(), rm), float("nan"))
            for t, rm in zip(wide_with_tk["ticker"], wide_with_tk["return_month"])
        ]
        d_ser = pd.Series(d_vals, index=wide.index, dtype=float)
        if USE_DEMAND:
            wide["score_demand"] = d_ser.fillna(0.0).to_numpy(float)
            wide["available_demand"] = d_ser.notna().to_numpy()
            wide["native_demand"] = d_ser.notna().to_numpy()
        if GATE_E7_DEMAND:
            for e7_col in ("e7_lm", "e7_small_gated"):
                s_col = wide[f"score_{e7_col}"].to_numpy(float).copy()
                av_col = wide[f"available_{e7_col}"].to_numpy(bool).copy()
                disagree = (s_col != 0.0) & d_ser.notna().to_numpy() & (d_ser.fillna(0.0).to_numpy() != 0.0) & ((s_col > 0) != (d_ser.fillna(0.0).to_numpy() > 0))
                s_col[disagree] = 0.0
                av_col[disagree] = False
                wide[f"score_{e7_col}"] = s_col
                wide[f"available_{e7_col}"] = av_col

    # Optional Daily TimesFM 21d trajectory sleeve / tilt / squeeze gate
    tfm_daily_lookup: dict[tuple[pd.Period, str], tuple[float, float, float]] = {}
    if USE_TFM_DAILY or USE_TFM_DAILY_TILT or USE_TFM_DAILY_SQUEEZE:
        tfm_daily_path = ROOT / "artifacts" / "timesfm_daily" / "forecasts.parquet"
        if tfm_daily_path.exists():
            td = pd.read_parquet(tfm_daily_path)
            td["return_month"] = pd.to_datetime(td["return_month"]).dt.to_period("M")
            td["rk_cum21"] = td.groupby("return_month")["tfm_daily_cum21"].rank(pct=True)
            td["rk_qspread"] = td.groupby("return_month")["tfm_daily_q_spread"].rank(pct=True)
            td["rk_uptail"] = td.groupby("return_month")["tfm_daily_upside_tail_5d"].rank(pct=True)
            for r_td in td.itertuples():
                tfm_daily_lookup[(r_td.return_month, str(int(r_td.permno)))] = (
                    float(r_td.rk_cum21),
                    float(r_td.rk_qspread),
                    float(r_td.rk_uptail),
                )
            if USE_TFM_DAILY:
                td_m = wide[["return_month", "permno"]].merge(
                    td[["return_month", "permno", "rk_cum21", "rk_qspread"]],
                    on=["return_month", "permno"],
                    how="left",
                )
                sc = ((td_m["rk_cum21"] - 0.5) * np.sqrt(12.0)).fillna(0.0)
                unc = (0.05 + 0.10 * td_m["rk_qspread"].fillna(0.5)).to_numpy(float)
                wide["score_tfm_daily"] = sc.to_numpy(float)
                wide["uncertainty_tfm_daily"] = unc
                wide["available_tfm_daily"] = td_m["rk_cum21"].notna().to_numpy()
                wide["native_tfm_daily"] = td_m["rk_cum21"].notna().to_numpy()

    all_sleeves = [
        "e7_lm",
        "e7_small_gated",
        "runway_micro",
        "conf_lm_gated",
        "inventory_retail",
        "oplev_software",
        "xgb_rank",
        "filing_events",
    ]
    if USE_TFM_MONTHLY:
        all_sleeves.append("timesfm_monthly")
    if USE_DEMAND:
        all_sleeves.append("demand")
    if USE_TFM_DAILY and "score_tfm_daily" in wide.columns:
        all_sleeves.append("tfm_daily")

    liquid_sleeves = [s for s in all_sleeves if s not in MICRO_SLEEVES]
    active_sleeves = liquid_sleeves if ENDOWMENT_NAV > MICRO_SLEEVES_MAX_NAV else all_sleeves

    # Register & validate SourceCards, HypothesisSpecs, RegisteredExperiments, and AdmissionCertificates
    contracts_meta = register_platform_contracts(active_sleeves, OUT)
    equal_denom = not BLEND_RENORM
    print(
        f"Integrating {len(active_sleeves)} admitted families via FamilyIntegrator "
        f"(policy={contracts_meta['integration_policy']}, compounding_nav={COMPOUND_NAV_SIZING})...",
        flush=True,
    )

    blended_all = blend_wide_sleeves(wide, all_sleeves, equal_denominator=equal_denom)
    blended_liq = blend_wide_sleeves(wide, liquid_sleeves, equal_denominator=equal_denom).rename(
        columns={
            "sleeve_score": "sleeve_score_liquid",
            "available_count": "available_count_liquid",
            "integrated_uncertainty": "integrated_uncertainty_liquid",
        }
    )
    scores = blended_all.merge(blended_liq, on=["return_month", "permno"], how="left")
    scores = scores.merge(formation, on=["return_month", "permno"], how="left")

    if USE_GRAPH:
        graph_fc = get_graph_forecasts()
        graph_months = pd.to_datetime(graph_fc["target_month"]).dt.to_period("M").unique()
        scores = scores.loc[scores["return_month"].isin(graph_months)].copy()
        scored = apply_graph_overlay(scores, graph_fc, formation)
        scored_liq = apply_graph_overlay(
            scores.assign(sleeve_score=scores["sleeve_score_liquid"]), graph_fc, formation
        )
        scored["final_score_liquid"] = scored_liq["final_score"].to_numpy()
    else:
        months_s = scores["return_month"]
        scores["final_score"] = scores["sleeve_score"].groupby(months_s, sort=False).rank(method="average", pct=True)
        scores["final_score_liquid"] = scores["sleeve_score_liquid"].groupby(months_s, sort=False).rank(method="average", pct=True)
        scored = scores

    score_col = "final_score_liquid" if ENDOWMENT_NAV > MICRO_SLEEVES_MAX_NAV else "final_score"
    if USE_TFM_DAILY_TILT and tfm_daily_lookup:
        base_s = scored[score_col].to_numpy(float).copy()
        for idx_i, r_sc in enumerate(scored[["return_month", "permno"]].itertuples(index=False)):
            tup = tfm_daily_lookup.get((r_sc.return_month, str(int(r_sc.permno))))
            if tup is not None and np.isfinite(base_s[idx_i]):
                # 95% base rank + 5% Daily TimesFM 21d trajectory rank
                base_s[idx_i] = 0.95 * base_s[idx_i] + 0.05 * tup[0]
        scored[score_col] = base_s
    scored["active_score"] = scored[score_col]
    scored["active_available_count"] = (
        scored["available_count_liquid"] if ENDOWMENT_NAV > MICRO_SLEEVES_MAX_NAV else scored["available_count"]
    )

    # Short interest table
    si_cache = FINAL_OUT / "short_interest" / "short_interest.parquet"
    if si_cache.exists():
        si_table = pd.read_parquet(si_cache)
    else:
        si_table = build_short_interest(str(EXTERNAL), str(PANEL))
    si_table["return_month"] = pd.to_datetime(si_table["return_month"]).dt.to_period("M")
    scored = scored.merge(
        si_table[["permno", "return_month", "short_pct"]], on=["permno", "return_month"], how="left"
    )

    # Prior-month missing-return closures clear the retention buffer
    outcomes_na = panel_full.loc[panel_full["ret_exc_lead1m"].isna(), ["permno", "eom"]]
    settled_closed = set(
        zip(outcomes_na["eom"].dt.to_period("M") + 1, outcomes_na["permno"].astype(int).astype(str))
    )

    official_limits = OfficialLimits()
    compliance = ComplianceEngine(official_limits)

    return_months = sorted(scored["return_month"].unique())
    print(f"Holding months to construct: {len(return_months)} ({return_months[0]} .. {return_months[-1]})", flush=True)

    print("Loading evaluation universe and market series for compounding NAV path...", flush=True)
    universe = load_evaluation_universe(PANEL)
    mkt = load_market_series(EXTERNAL)

    books_by_month: dict[pd.Period, dict[int, float]] = {}
    portfolio_rows: list[dict] = []
    score_history: list[pd.DataFrame] = []
    overlay_log: list[dict] = []
    coverage_log: list[dict] = []
    prev_weights: dict[str, float] = {}
    last_holdings: list[ActualHolding] = []
    running_nav = ENDOWMENT_NAV

    for ret_month in return_months:
        form_month = ret_month - 1
        form_ts = form_month.to_timestamp("M")
        form_date = form_ts.date()
        current = scored.loc[scored["return_month"] == ret_month].copy()
        if len(current) < 250:
            continue

        # Clear settled closures from previous weights before retention selection
        prev_weights = {
            sid: w for sid, w in prev_weights.items() if (ret_month - 1, sid) not in settled_closed
        }
        prev_long = {sid for sid, w in prev_weights.items() if w > 0}
        prev_short = {sid for sid, w in prev_weights.items() if w < 0}

        capital = running_nav if COMPOUND_NAV_SIZING else ENDOWMENT_NAV
        internal_policy = InternalPolicy(
            target_positions=N_PER_SIDE * 2,
            target_gross=TARGET_GROSS,
            target_net=0.0,
            max_abs_net=MAX_ABS_NET,
            retention_band=0.5,
            sector_neutral=True,
            min_sector_names=4,
            beta_neutral=True,
            beta_tilt=0.25,
            beta_lower=0.2,
            beta_upper=3.0,
            max_short_interest=0.15,
            min_price_usd=5.0,
            execution_days=5,
            long_adv_participation=0.05,
            short_adv_participation=0.02,
            require_verified_labels=True,
            capital_usd=capital,
            enforce_max_abs_net=CAP_NET_EXPOSURE,
            allow_all_ranked_sides=True,
            ceil_retention_zone=True,
        )
        engine = ConstructionEngine(official_limits, internal_policy)

        candidates: list[Candidate] = []
        for r in current.itertuples():
            if not (r.active_available_count > 0 and np.isfinite(r.active_score) and np.isfinite(r.prc) and r.dolvol_126d > 0):
                continue
            sid = str(int(r.permno))
            verified = (
                pd.notna(r.ticker)
                and pd.notna(r.company_name)
                and str(r.ticker_name_status) != UNVERIFIED_LABEL
            )
            b_val = float(r.beta_60m) if pd.notna(r.beta_60m) and np.isfinite(r.beta_60m) else 1.0
            si_val = float(r.short_pct) if pd.notna(r.short_pct) and np.isfinite(r.short_pct) else float("nan")
            candidates.append(
                Candidate(
                    security_id=sid,
                    alpha=float(r.active_score),
                    market_cap_usd=float(r.me) * 1_000_000.0 if pd.notna(r.me) else 0.0,
                    tradable=True,
                    incumbent=sid in prev_weights,
                    incumbent_long=sid in prev_long,
                    incumbent_short=sid in prev_short,
                    sector=str(r.sector) if pd.notna(r.sector) else "na",
                    beta=b_val,
                    short_interest_pct=si_val,
                    price_usd=float(r.prc),
                    adv_usd=float(r.dolvol_126d),
                    verified_label=verified,
                    size_grp=str(r.size_grp) if pd.notna(r.size_grp) else "large",
                )
            )

        book = engine.build(
            candidates,
            run_id=f"seven_{ret_month}",
            decision_at=dt.datetime(form_date.year, form_date.month, form_date.day, tzinfo=dt.timezone.utc),
            forecast_publication_ids=[f"fc_{ret_month}"],
            current_book_snapshot_id=f"snap_{ret_month}",
            nav_snapshot_id=f"nav_{ret_month}",
            price_snapshot_id=f"px_{ret_month}",
            risk_envelope_version="envelope_v2",
            covariance_version="cov_v2",
            cost_model_version="tiered_cost_v2",
            attribution_method=contracts_meta["integration_policy"],
        )
        weights = {p.security_id: float(p.target_weight) for p in book.positions}

        # Append month's scores for edge throttle
        score_history.append(
            pd.DataFrame(
                {
                    "return_month": ret_month,
                    "permno": current["permno"].astype(int).astype(str).to_numpy(),
                    "final_score": current["active_score"].astype(float).to_numpy(),
                    "prc": current["prc"].astype(float).to_numpy(),
                }
            )
        )

        squeeze_df = pd.DataFrame()
        if USE_RISK_OVERLAY and weights:
            scores_df = pd.concat(score_history, ignore_index=True)
            overlay_result = protect(
                weights,
                ret_month,
                overlay_panel,
                scores_df,
                max_abs_net=MAX_ABS_NET if CAP_NET_EXPOSURE else None,
            )
            weights = overlay_result.weights
            if USE_TFM_DAILY_SQUEEZE and tfm_daily_lookup:
                # Shrink shorts in top 5% of Daily TimesFM 5-day upside jump tail / 21-day quantile spread
                w_ser = pd.Series(weights, dtype=float)
                short_ids = [sid for sid, w in weights.items() if w < 0]
                tfm_flagged = [
                    sid
                    for sid in short_ids
                    if tfm_daily_lookup.get((ret_month, sid), (0.5, 0.0, 0.0))[2] >= 0.95
                    or tfm_daily_lookup.get((ret_month, sid), (0.5, 0.0, 0.0))[1] >= 0.95
                ]
                if tfm_flagged:
                    before_short_sum = float(-w_ser[w_ser < 0].sum())
                    for sid in tfm_flagged:
                        w_ser[sid] *= 0.5
                    after_short_sum = float(-w_ser[w_ser < 0].sum())
                    safe_shorts = [sid for sid in short_ids if sid not in tfm_flagged]
                    if safe_shorts and after_short_sum > 0:
                        freed = before_short_sum - after_short_sum
                        safe_sum = float(-w_ser[safe_shorts].sum())
                        if safe_sum > 0:
                            for sid in safe_shorts:
                                w_ser[sid] *= (safe_sum + freed) / safe_sum
                    weights = w_ser.to_dict()
                overlay_result.summary["tfm_daily_squeeze_flagged"] = len(tfm_flagged)
            squeeze_df = overlay_result.stats
            overlay_log.append({"return_month": str(ret_month), "formation_date": str(form_date), **overlay_result.summary})

        books_by_month[ret_month] = {int(k): float(v) for k, v in weights.items()}
        prev_weights = weights
        if COMPOUND_NAV_SIZING:
            eval_so_far = evaluate_monthly_books(books_by_month, universe, mkt)
            if not eval_so_far.empty:
                running_nav = ENDOWMENT_NAV * float((1.0 + eval_so_far["total"]).prod())

        attr_lookup = current.set_index(current["permno"].astype(int).astype(str))
        for sid, w in weights.items():
            row = attr_lookup.loc[sid]
            sq_info = squeeze_df.loc[sid].to_dict() if not squeeze_df.empty and sid in squeeze_df.index else {}
            portfolio_rows.append(
                {
                    "return_month": ret_month,
                    "formation_date": form_date,
                    "permno": int(sid),
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "ticker_name_reference_date": row["ticker_name_reference_date"],
                    "ticker_name_source": row["ticker_name_source"],
                    "ticker_name_status": row["ticker_name_status"],
                    "prc": float(row["prc"]),
                    "me": float(row["me"]),
                    "size_grp": str(row["size_grp"]),
                    "sector": str(row["sector"]),
                    "beta_60m": float(row["beta_60m"]) if pd.notna(row["beta_60m"]) else 1.0,
                    "dolvol_126d": float(row["dolvol_126d"]),
                    "final_score": float(row["active_score"]),
                    "target_weight": float(w),
                    "target_dollars": float(w) * capital,
                    "side": "long" if w > 0 else "short",
                    "capital": capital,
                    **sq_info,
                }
            )

        last_holdings = [
            ActualHolding(
                security_id=sid,
                weight=Decimal(str(round(w, 10))),
                market_cap_usd=Decimal(str(float(attr_lookup.loc[sid, "me"]) * 1_000_000.0)),
                formation_market_cap_usd=Decimal(str(float(attr_lookup.loc[sid, "me"]) * 1_000_000.0)),
            )
            for sid, w in weights.items()
        ]

        w_Month = wide.loc[wide["return_month"] == ret_month]
        cov_m = {s: int(w_Month[f"available_{s}"].sum()) for s in active_sleeves if f"available_{s}" in w_Month}
        coverage_log.append({"return_month": str(ret_month), "formation_date": str(form_date), "coverage": cov_m})

    # Evaluate monthly returns with drifted-weight turnover, last-quote liquidation, and size-tiered costs/borrow
    print("Evaluating monthly returns and size-tiered transaction/borrow costs...", flush=True)
    monthly_eval = evaluate_monthly_books(books_by_month, universe, mkt)

    # Evaluate daily spine on dsf_mini_from_2014.csv
    daily_df = pd.DataFrame()
    daily_summary: dict[str, object] = {}
    dsf = pd.DataFrame()
    if DSF_PATH.exists():
        print("Evaluating daily return spine on dsf_mini_from_2014.csv...", flush=True)
        held_permnos = {int(r["permno"]) for r in portfolio_rows}
        # Also include scored permnos if computing daily alpha decay, or load held permnos first
        dsf = load_dsf_daily(
            DSF_PATH,
            permnos=held_permnos,
            first_date=pd.Timestamp("2021-01-01"),
            last_date=pd.Timestamp("2026-08-31"),
        )
        fred = load_fred(EXTERNAL)
        daily_df, daily_summary = evaluate_daily_spine(books_by_month, dsf, fred, monthly_df=monthly_eval)

    portfolio_df = pd.DataFrame(portfolio_rows)
    hold_period = compute_holding_period(portfolio_df)

    # Build NAV path and monthly.json
    nav = ENDOWMENT_NAV
    nav_path = [{"month": "start", "nav": nav}]
    monthly_records = []
    for m_period, row in monthly_eval.iterrows():
        nav *= 1.0 + float(row["total"])
        nav_path.append({"month": str(m_period), "nav": nav})
        monthly_records.append(
            {
                "month": str(m_period),
                "gross_return": float(row["gross_return"]),
                "net_return": float(row["total"]),
                "rf_monthly": float(row["rf"]),
                "benchmark_monthly": float(row["benchmark"]),
                "sp500_monthly": float(row["spx"]) if pd.notna(row["spx"]) else 0.0,
                "active_return": float(row["active"]),
                "pnl_long": float(row["pnl_long"]),
                "pnl_short": float(row["pnl_short"]),
                "borrow": float(row["borrow"]),
                "cost": float(row["cost"]),
                "turnover": float(row["turnover_oneway"]),
                "turnover_two_way": float(row["traded"]),
                "names": int(row["names"]),
                "longs": int(row["longs"]),
                "shorts": int(row["shorts"]),
                "gross_exposure": float(row["gross"]),
                "net_exposure": float(row["net"]),
                "beta_exante": float(row["beta_exante"]),
            }
        )

    avg_pos_musd = float((portfolio_df["target_weight"].abs() * portfolio_df["capital"] / 1e6).mean())
    table = RequiredTable(
        ending_value=nav,
        starting_value=ENDOWMENT_NAV,
        elapsed_years=len(monthly_records) / 12.0,
        periods_per_year=12,
        portfolio_returns=tuple(m["net_return"] for m in monthly_records),
        benchmark_returns=tuple(m["benchmark_monthly"] for m in monthly_records),
        risk_free_returns=tuple(m["rf_monthly"] for m in monthly_records),
        gross_returns=tuple(m["gross_return"] for m in monthly_records),
        cost_returns=tuple(m["cost"] for m in monthly_records),
        borrow_returns=tuple(m["borrow"] for m in monthly_records),
        position_counts=tuple(m["names"] for m in monthly_records),
        gross_exposures=tuple(m["gross_exposure"] for m in monthly_records),
        net_exposures=tuple(m["net_exposure"] for m in monthly_records),
        turnover_per_rebalance=tuple(m["turnover"] for m in monthly_records),
        spx_returns=tuple(m["sp500_monthly"] for m in monthly_records),
        max_one_day_loss=daily_summary.get("max_one_day_loss"),
        max_one_day_loss_date=daily_summary.get("max_one_day_loss_date"),
        beta_daily=daily_summary.get("beta_daily"),
        beta_daily_se=daily_summary.get("beta_daily_se"),
        beta_daily_t_hac=daily_summary.get("beta_daily_t_hac"),
        avg_holding_period_days=hold_period["avg_holding_period_trading_days"],
        avg_position_size_musd=avg_pos_musd,
    ).as_table()

    compliance_report = compliance.check(last_holdings, max_abs_internal_net=Decimal(str(MAX_ABS_NET))) if last_holdings else None

    # Save portfolio & predictions parquets + required JSONs
    portfolio_df_save = portfolio_df.copy()
    portfolio_df_save["return_month"] = portfolio_df_save["return_month"].dt.to_timestamp("M")
    portfolio_df_save.to_parquet(OUT / "portfolio.parquet", index=False)

    predictions_save = scored[["return_month", "permno", "prc", "me", "size_grp", "sector", "final_score", "final_score_liquid"]].copy()
    predictions_save["return_month"] = predictions_save["return_month"].dt.to_timestamp("M")
    predictions_save.to_parquet(OUT / "predictions.parquet", index=False)

    (OUT / "nav_path.json").write_text(json.dumps(nav_path, indent=2), encoding="utf-8")
    (OUT / "monthly.json").write_text(json.dumps(monthly_records, indent=2), encoding="utf-8")
    (OUT / "required_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (OUT / "coverage.json").write_text(json.dumps(coverage_log, indent=2), encoding="utf-8")
    if daily_summary:
        (OUT / "daily_summary.json").write_text(json.dumps(daily_summary, indent=2), encoding="utf-8")
    if overlay_log:
        (OUT / "overlay_log.json").write_text(json.dumps(overlay_log, indent=2, default=str), encoding="utf-8")
    if compliance_report is not None:
        (OUT / "compliance.json").write_text(
            json.dumps(compliance_report.as_dict(), indent=2), encoding="utf-8"
        )

    # Export offline Arcana attribution audit, Section 9.4 exhibits, and submission CSVs
    print("Exporting offline Arcana attribution audit, Section 9.4 exhibits, and submission CSVs...", flush=True)
    arcana_out = export_submission_and_exhibits(
        portfolio_df=portfolio_df,
        predictions_df=predictions_save,
        universe=universe,
        mkt=mkt,
        monthly_eval_df=monthly_eval,
        daily_df=daily_df,
        daily_summary=daily_summary,
        dsf=dsf,
        external_path=EXTERNAL,
        panel_path=PANEL,
        out_dir=OUT,
        score_col=score_col,
        demand_card_path=ROOT / "artifacts" / "demand_card.parquet",
    )

    avg_coverage = {
        name: sum(c["coverage"].get(name, 0) for c in coverage_log) / max(len(coverage_log), 1)
        for name in active_sleeves
    }
    print()
    label = "WITH RISK OVERLAY" if USE_RISK_OVERLAY else "no risk overlay"
    print(f"=== final_v2 68-month institutional run ({label}) | gross={TARGET_GROSS:.2f}x | sleeves={len(active_sleeves)} ===")
    if overlay_log:
        scales = [o["risk_scale"] for o in overlay_log]
        edges = [o["edge_scale"] for o in overlay_log]
        flagged = sum(o["peer_flagged_shorts"] for o in overlay_log)
        capped = sum(o["runup_capped_shorts"] for o in overlay_log)
        net_caps = sum(1 for o in overlay_log if o.get("post_overlay_net_capped"))
        print(f"overlay: avg risk_scale={sum(scales)/len(scales):.3f} avg edge_scale={sum(edges)/len(edges):.3f} net_recentered={net_caps}")
        print(f"overlay: peer-flagged shorts={flagged} runup-capped={capped}")
    print(f"holding months evaluated: {len(monthly_records)} ({monthly_records[0]['month']} .. {monthly_records[-1]['month']})")
    print(f"avg coverage per sleeve : " + ", ".join(f"{k}={v:.0f}" for k, v in avg_coverage.items()))
    print(f"ending NAV              : ${nav:,.0f}")
    print(f"cumulative return       : {table['cumulative_return']:.2%}")
    print(f"CAGR                    : {table['cagr']:.2%}")
    print(f"annualized volatility   : {table['annualized_volatility']:.2%}")
    print(f"max drawdown (monthly)  : {table['max_drawdown']:.2%} (duration: {table['max_drawdown_duration_months']}m)")
    if daily_summary:
        print(f"max 1-day loss (daily)  : {daily_summary['max_one_day_loss']:.2%} on {daily_summary['max_one_day_loss_date']}")
        print(f"daily beta vs S&P 500   : {daily_summary.get('beta_daily', 0.0):.3f} (t_hac={daily_summary.get('beta_daily_t_hac', 0.0):.2f})")
    print(f"IR vs TB3MS+4%          : {table['information_ratio']:.2f}")
    print(f"Sharpe vs TB3MS         : {table['sharpe_ratio']:.2f}")
    print(f"hit rate vs TB3MS+4%    : {table['hit_rate_vs_benchmark']:.1%}")
    print(f"turnover per rebalance  : {table['turnover_per_rebalance_avg']:.2%} (holding period: {table['average_holding_period_trading_days']:.1f}d)")
    print(f"annual cost % NAV       : {table['annual_cost_pct_nav']:.2%}")
    print(f"idiosyncratic ret share : {arcana_out['factor_vs_idiosyncratic']['idiosyncratic_share_of_mean_return']:.1%}")
    print(f"selection share of P&L  : {arcana_out['sizing_vs_selection']['selection_share_of_pnl']:.1%}")
    if compliance_report is not None:
        warns = [c.name for c in compliance_report.checks if c.status == "warn"]
        print(f"compliance official pass: {compliance_report.official_pass} | internal warnings: {warns}")
    print(f"elapsed runtime         : {time.time() - t_start:.1f}s | artifacts: {OUT}")


if __name__ == "__main__":
    main()
