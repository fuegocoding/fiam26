"""Deterministic planted-signal fixtures. Never investment evidence."""

from __future__ import annotations

import csv
import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path

from ..contracts.sources import (
    AssessmentStatus,
    CardStatus,
    Screen,
    ScreenItem,
    SourceCard,
)
from ..research.evaluation import ForecastOutcome
from ..strategies.base import Charter
from .common import SYNTHETIC_LABEL, UTC, atomic_write_json


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 2608
    names: int = 40
    design_dates: int = 8
    check_dates: int = 4


def build_world(config: SyntheticConfig = SyntheticConfig()):
    if config.names < 30 or config.design_dates < 8 or config.check_dates < 4:
        raise ValueError(
            "fixture requires >=30 names, 8 design and 4 check dates"
        )
    rng = random.Random(config.seed)
    registered = dt.datetime(2014, 1, 1, tzinfo=UTC)
    dates = [
        registered + dt.timedelta(days=32 * (i + 1))
        for i in range(config.design_dates + config.check_dates)
    ]
    series = []
    for date in dates:
        rows = []
        for n in range(config.names):
            signal, baseline = rng.gauss(0, 1), rng.gauss(0, 1)
            target = signal + rng.gauss(0, 0.3)
            rows.append(
                ForecastOutcome(
                    security_id=f"S{n:03}",
                    formation_at=date,
                    feature_available_at=date - dt.timedelta(days=1),
                    label_resolves_at=date + dt.timedelta(days=28),
                    prediction=signal,
                    baseline_prediction=baseline,
                    target=target,
                    net_forward_return=0.02 * target,
                )
            )
        series.append(rows)
    item = ScreenItem(
        status=AssessmentStatus.ESTABLISHED,
        evidence_id="synthetic_generator_v1",
    )
    card = SourceCard(
        owner="data",
        created_at=registered,
        source_id="synthetic_signal",
        dataset_id="SYNTHETIC_FIXTURE",
        topic=SYNTHETIC_LABEL,
        snapshot_id=f"synthetic_{config.seed}",
        status=CardStatus.APPROVED,
        access_confirmed=True,
        formation_eligible=True,
        entity_grain=["security"],
        screen=Screen(
            history=item, metric_lag=item, vintage=item, coverage=item
        ),
    )
    return {
        "label": SYNTHETIC_LABEL,
        "config": config.__dict__,
        "registered_at": registered,
        "evaluation_asof": dates[-1] + dt.timedelta(days=29),
        "card": card,
        "design": [o for rows in series[: config.design_dates] for o in rows],
        "check": [o for rows in series[config.design_dates :] for o in rows],
        "charters": [c.value for c in Charter],
    }


def write_fixture(world, published, certificates, out_dir, **kwargs):
    """Write the operator fixture schema consumed by ControlledReplay.

    Only explicitly published/admitted strategy forecasts are accepted. If none
    pass challenge, the caller must separately authorize a test baseline book.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not published or not certificates:
        raise ValueError(
            "no certified published forecasts: cannot construct a trading fixture"
        )
    atomic_write_json(
        root / "config.json",
        {
            "run_id": "reference_demo_book",
            "run_kind": "replay",
            "starting_nav": "1000000",
            "certificate_policy": "test",
            "universe_id": "synthetic_universe",
            "official_limits": {
                "min_positions": 4,
                "max_positions": 50,
                "max_gross": 2.0,
                "min_net": -0.5,
                "max_net": 0.5,
                "max_abs_name_weight": 0.3,
                "min_short_market_cap_usd": 0,
            },
            "internal_policy": {
                "target_positions": 8,
                "sector_neutral": False,
                "beta_neutral": False,
            },
            "admitted_families": sorted({f.strategy_id for f in published}),
            "neutral_uncertainty": 0.1,
            "cost_model": {
                "model_id": "synthetic_cost_v1",
                "half_spread_by_size_bp": {"large": 2.0},
                "eta": 0.1,
                "explicit_fee_bp": 0.5,
            },
            "execution_policy": {
                "policy_id": "synthetic_exec_v1",
                "max_daily_participation": 0.2,
                "min_resize_fraction": 0.0,
                "no_trade_band": 0.0005,
                "min_economic_edge_bp": 0.0,
                "order_ttl_days": 5,
            },
            "fill_window_days": 5,
            "core_sources": ["synthetic_signal"],
            "risk": {
                "policy_version": "synthetic_risk_v1",
                "max_gross": 2.0,
                "max_abs_net": 0.5,
                "max_name_weight": 0.3,
                "min_names": 4,
            },
        },
    )
    atomic_write_json(
        root / "source_cards.json", [world["card"].model_dump(mode="json")]
    )
    atomic_write_json(
        root / "forecasts.json", [f.model_dump(mode="json") for f in published]
    )
    atomic_write_json(
        root / "certificates.json",
        [c.model_dump(mode="json") for c in certificates],
    )
    # Distinct future fixture dates: challenge outcomes cannot leak into deployment.
    formations = sorted({f.earliest_formation_at.date() for f in published})
    with (root / "periods.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["formation_date", "period_end", "tb3ms_annual_percent"]
        )
        for i, date in enumerate(formations):
            end = (
                formations[i + 1]
                if i + 1 < len(formations)
                else date + dt.timedelta(days=30)
            )
            writer.writerow([date, end, "3.0"])
    by_date = {
        date: {
            f.security_id: f
            for f in published
            if f.earliest_formation_at.date() == date
        }
        for date in formations
    }
    with (root / "market.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "formation_date",
                "security_id",
                "ticker",
                "name",
                "label_source",
                "label_reference_date",
                "market_cap_usd",
                "beta",
                "sector",
                "tradable",
                "price",
                "adv_dollars",
                "daily_volatility",
                "size_tier",
                "outcome_excess_return",
            ]
        )
        for i, date in enumerate(formations):
            for n, (sid, f) in enumerate(sorted(by_date[date].items())):
                # First selected long loses: a real failed trade inside this labelled fixture.
                expected = float(f.return_forecast.expected_excess_return)
                outcome = (
                    -0.12 if sid == "S000" else expected + ((n % 3) - 1) * 0.003
                )
                writer.writerow(
                    [
                        date,
                        sid,
                        sid,
                        f"Synthetic {sid}",
                        SYNTHETIC_LABEL,
                        date,
                        "1000000000",
                        1,
                        "synthetic",
                        "true",
                        100,
                        50000000,
                        0.02,
                        "large",
                        outcome,
                    ]
                )
    atomic_write_json(
        root / "lineage.json",
        {
            "security_master_version": "synthetic_master_v1",
            "economic_graph_version": "synthetic_graph_v1",
            "calendar_policy": "synthetic_30_day",
            "timezone_policy": "utc_close_2100",
            "data_access_policy_version": "synthetic_v1",
            "evaluation_policy_version": "fixture_challenge_v1",
            "trial_count": 16,
            "evaluation_access_count": 0,
        },
    )
    return root
