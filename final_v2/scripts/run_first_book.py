"""First real run: competition chars panel through the platform end to end.

Loads the approved competition panel (source card COMP_chars_panel), computes
two registered charter signals from real columns (Charter 4 momentum ret_12_1,
Charter 2 net issuance chcsho_12m/eqnpo_12m), integrates families, constructs
a mandate-compliant book monthly, evaluates with real returns and frozen
costs, and writes artifacts: NAV path, books, compliance, required table.

Usage: python final_v2/scripts/run_first_book.py
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

from pod_platform.accounting.returns import benchmark_monthly  # noqa: E402
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

REPO = ROOT.parent
PANEL = REPO / "final" / "data" / "chars_final_with_names.parquet"
FACTORS = REPO / "final" / "data" / "factor_char_list.csv"
EXTERNAL = REPO / "final" / "external_data" / "external.parquet"
OUT = ROOT / "artifacts" / "first_book"

TEST_START = dt.datetime(2021, 1, 31, tzinfo=dt.timezone.utc)
TEST_END = dt.datetime(2026, 8, 31, tzinfo=dt.timezone.utc)
COST_BP_PER_SIDE = 20.0
BORROW_ANNUAL = 0.05
FAMILIES = ("momentum", "capital_allocation")


def load_tb3ms() -> dict[str, float]:
    ext = pd.read_parquet(EXTERNAL, columns=["table", "date", "series", "value"])
    fred = ext[ext["table"] == "fred"].copy()
    fred["date"] = pd.to_datetime(fred["date"])
    tb = fred[fred["series"] == "TB3MS"].set_index("date")["value"]
    return {f"{d.year}-{d.month:02d}": float(v) for d, v in tb.items()}


def _rank_scale(series):
    ranked = series.rank(pct=True)
    return (ranked - 0.5) * (12 ** 0.5)


def month_candidates(month):
    df = month.dropna(subset=["ret_12_1", "chcsho_12m", "eqnpo_12m", "me_usd"]).copy()
    df["mom"] = _rank_scale(df["ret_12_1"])
    df["iss"] = _rank_scale(-df["chcsho_12m"] + df["eqnpo_12m"])
    candidates = []
    estimates = {}
    for row in df.itertuples():
        alpha = 0.5 * float(row.mom) + 0.5 * float(row.iss)
        candidates.append(
            Candidate(
                security_id=str(int(row.permno)),
                alpha=alpha,
                market_cap_usd=float(row.me_usd),
                tradable=True,
                incumbent=False,
                sector=str(row.gics)[:2] if row.gics else "unknown",
            )
        )
        estimates[str(int(row.permno))] = [
            FamilyEstimate("momentum", str(int(row.permno)), float(row.mom), 0.1),
            FamilyEstimate("capital_allocation", str(int(row.permno)), float(row.iss), 0.1),
        ]
    return candidates, estimates, df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loader = CharsPanelLoader(PANEL, FACTORS)
    panel = loader.load()
    tb3ms = load_tb3ms()
    print(f"panel rows: {len(panel):,}  units: {loader.units}")
    print(f"TB3MS months loaded: {len(tb3ms)}")

    integrator = FamilyIntegrator(FAMILIES, neutral_uncertainty=0.1)
    engine = ConstructionEngine(OfficialLimits(), InternalPolicy(target_positions=300))
    compliance = ComplianceEngine(OfficialLimits())

    months = sorted(
        {m for m in panel["eom"].tolist() if TEST_START.date() <= m.date() <= TEST_END.date()}
    )
    print(f"formation months: {len(months)} ({months[0].date()} .. {months[-1].date()})")

    nav = 1_000_000_000.0
    nav_path = [{"month": "start", "nav": nav}]
    monthly = []
    last_holdings: list[ActualHolding] = []
    total_turnover = 0.0
    prev_weights: dict[str, float] = {}
    materialization = None

    for month_ts in months:
        formation_at = dt.datetime(month_ts.year, month_ts.month, month_ts.day, tzinfo=dt.timezone.utc)
        month_df, materialization = loader.formation_month(formation_at)
        candidates, estimates, merged = month_candidates(month_df)
        if len(candidates) < 250:
            continue

        book = engine.build(
            candidates,
            run_id=f"first_{month_ts.date()}",
            decision_at=formation_at,
            forecast_publication_ids=[f"fc_{month_ts.date()}"],
            current_book_snapshot_id=f"snap_{month_ts.date()}",
            nav_snapshot_id=f"nav_{month_ts.date()}",
            price_snapshot_id=f"px_{month_ts.date()}",
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

        rets = merged.set_index(merged["permno"].astype(int).astype(str))["ret_exc_lead1m"].to_dict()

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
        month_key = f"{month_ts.year}-{month_ts.month:02d}"
        rf_m = tb3ms.get(month_key, 0.0) / 1200.0
        bench_m = rf_m + 0.04 / 12.0
        active_ret = sum(w * resolved.get(k, 0.0) for k, w in weights.items())
        short_w = -sum(w for w in weights.values() if w < 0)
        borrow = short_w * BORROW_ANNUAL / 12.0
        cost = turnover * COST_BP_PER_SIDE / 10_000.0 * 2.0
        gross_ret = rf_m + active_ret
        net_ret = gross_ret - borrow - cost

        nav *= 1.0 + net_ret
        nav_path.append({"month": str(month_ts.date()), "nav": nav})
        monthly.append(
            {
                "month": str(month_ts.date()),
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

        last_holdings = [
            ActualHolding(
                security_id=p.security_id,
                weight=Decimal(str(p.target_weight)),
                market_cap_usd=Decimal(str(next(c.market_cap_usd for c in candidates if c.security_id == p.security_id))),
                formation_market_cap_usd=Decimal(str(next(c.market_cap_usd for c in candidates if c.security_id == p.security_id))),
            )
            for p in book.positions
        ]

    net_returns = [m["net_return"] for m in monthly]
    bench = [m["benchmark_monthly"] for m in monthly]
    rf = [m["rf_monthly"] for m in monthly]

    table = RequiredTable(
        ending_value=nav,
        starting_value=1_000_000_000.0,
        elapsed_years=len(monthly) / 12.0,
        periods_per_year=12,
        portfolio_returns=tuple(net_returns),
        benchmark_returns=tuple(bench),
        risk_free_returns=tuple(rf),
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
    if compliance_report is not None:
        (OUT / "compliance.json").write_text(
            json.dumps(compliance_report.as_dict(), indent=2), encoding="utf-8"
        )

    print()
    print("=== first real run: charter signals on the competition panel ===")
    print(f"months evaluated        : {len(monthly)} (of {len(months)} formation months)")
    skipped = len(months) - len(monthly)
    if skipped:
        print(f"months skipped          : {skipped} (label coverage < 50% of book weight)")
    max_missing = max((m["missing_weight_share"] for m in monthly), default=0.0)
    print(f"max missing-weight share: {max_missing:.2%} (missing returns not zero-filled silently; see run notes)")
    print(f"avg TB3MS (annualized)  : {sum(m['rf_monthly'] for m in monthly) / len(monthly) * 12:.2%}")
    print(f"ending NAV              : ${nav:,.0f}")
    print(f"cumulative return       : {table['cumulative_return']:.2%}")
    print(f"CAGR                    : {table['cagr']:.2%}")
    print(f"annualized volatility   : {table['annualized_volatility']:.2%}")
    print(f"max drawdown            : {table['max_drawdown']:.2%}")
    print(f"IR vs cash+4%           : {table['information_ratio']:.2f}")
    print(f"Sharpe                  : {table['sharpe_ratio']:.2f}")
    print(f"avg names / month       : {table['position_counts']['average']:.0f}")
    print(f"avg gross / net         : {table['gross_exposure']['average']:.2f} / {table['net_exposure']['average']:.2f}")
    print(f"turnover per rebalance  : {table['turnover_per_rebalance_avg']:.2%}")
    print(f"annual cost % NAV       : {table['annual_cost_pct_nav']:.2%}")
    if compliance_report is not None:
        print(f"compliance official pass: {compliance_report.official_pass}")
    if materialization is not None:
        print(f"materialization checks  : {materialization.assertions}")
    print(f"artifacts written to    : {OUT}")


if __name__ == "__main__":
    main()
