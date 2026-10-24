"""Ported-signal run: the five deterministic sleeves through the platform.

Reproduces the old combiner's equal-weight-over-covering-sleeves blend, then
runs the full engine: family integration, construction, costs, evaluation.
Usage: python final_v2/scripts/run_ported_book.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
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
from pod_platform.signals import PortedSignals, scaled_rank  # noqa: E402

REPO = ROOT.parent
PANEL = REPO / "final" / "data" / "chars_final_with_names.parquet"
FACTORS = REPO / "final" / "data" / "factor_char_list.csv"
EXTERNAL = REPO / "final" / "external_data" / "external.parquet"
OUT = ROOT / "artifacts" / "ported_book"

TEST_START = dt.date(2021, 1, 31)
TEST_END = dt.date(2026, 8, 31)
COST_BP_PER_SIDE = 20.0
BORROW_ANNUAL = 0.05
FAMILIES = (
    "inventory_retail",
    "oplev_software",
    "runway_micro",
    "e7_surprise_breadth",
    "conf_revision_gated",
)


def load_tb3ms() -> dict[str, float]:
    ext = pd.read_parquet(EXTERNAL, columns=["table", "date", "series", "value"])
    fred = ext[ext["table"] == "fred"].copy()
    fred["date"] = pd.to_datetime(fred["date"])
    tb = fred[fred["series"] == "TB3MS"].set_index("date")["value"]
    return {f"{d.year}-{d.month:02d}": float(v) for d, v in tb.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loader = CharsPanelLoader(PANEL, FACTORS)
    panel = loader.load()
    panel["eom"] = pd.to_datetime(panel["eom"])
    signals = PortedSignals(EXTERNAL)
    tb3ms = load_tb3ms()
    print(f"panel rows: {len(panel):,}")

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
    total_turnover = 0.0
    coverage_log = []

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

        coverage = {
            name: int((res.eligible.fillna(False)).sum()) for name, res in results.items()
        }
        coverage_log.append({"month": str(month_date), "coverage": coverage})

        estimates: dict[str, list[FamilyEstimate]] = {}
        candidates = []
        merged = month_df.copy()
        merged["sid"] = merged["permno"].astype(int).astype(str)
        for sid, idx in zip(merged["sid"], merged.index):
            fams = []
            for name in FAMILIES:
                value = float(scaled[name].get(idx, 0.0))
                if value != 0.0:
                    fams.append(FamilyEstimate(name, sid, value, 0.1))
            if not fams:
                continue
            estimates[sid] = fams
            alpha = sum(f.expected_excess_return for f in fams) / len(FAMILIES)
            row = merged.loc[idx]
            candidates.append(
                Candidate(
                    security_id=sid,
                    alpha=alpha,
                    market_cap_usd=float(row["me_usd"]),
                    tradable=True,
                    incumbent=sid in prev_weights,
                    sector=str(row["gics"])[:2] if row["gics"] else "unknown",
                )
            )
        if len(candidates) < 250:
            continue

        book = engine.build(
            candidates,
            run_id=f"ported_{month_date}",
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
        turnover = 0.5 * sum(
            abs(weights.get(k, 0.0) - prev_weights.get(k, 0.0))
            for k in set(weights) | set(prev_weights)
        )
        total_turnover += turnover

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
    if compliance_report is not None:
        (OUT / "compliance.json").write_text(
            json.dumps(compliance_report.as_dict(), indent=2), encoding="utf-8"
        )

    avg_coverage = {
        name: sum(c["coverage"][name] for c in coverage_log) / max(len(coverage_log), 1)
        for name in FAMILIES
    }
    print()
    print("=== ported sleeves: 5 deterministic signals through the engine ===")
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
