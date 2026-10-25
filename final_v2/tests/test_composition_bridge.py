"""Actual fixed-graph production, sealed check and scoped forecast publication."""

import datetime as dt
import random

import pytest

from pod_platform.application.artifacts import ArtifactIndex
from pod_platform.application.integrity import IntegrityPolicy, IntegrityService
from pod_platform.application.namespaces import RoleContext
from pod_platform.application.publication import PublicationService
from pod_platform.contracts.sources import SourceCard
from pod_platform.papers import PaperService, PaperStore
from pod_platform.research.builtins.fixtures import fixture_manifest
from pod_platform.research.builtins.timeutil import (
    month_index,
    month_end_known_at,
)
from pod_platform.research.challenge_bridge import CheckAuthority
from pod_platform.research.composition import (
    CompositionService,
    observation_digest,
)
from pod_platform.research.composition_bridge import CompositionBridge
from pod_platform.research.evaluation import RegisteredEvaluator
from pod_platform.research.features import FeatureObservation
from pod_platform.research.registration import ResearchLedger
from pod_platform.research.registry import Registry

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)


@pytest.fixture
def world(tmp_path):
    first = month_index(dt.date(2015, 1, 1))
    rng = random.Random(81)
    rows = []
    previous = {str(i): rng.gauss(0, 1) for i in range(40)}
    for month in range(first, first + 19):
        for security in previous:
            traffic, spend = rng.gauss(0, 1), rng.gauss(0, 1)
            for source, metric, value in [
                ("traffic", "visits", traffic),
                ("spend", "spend", spend),
                ("earnings", "margin", previous[security]),
                (
                    "returns",
                    "return",
                    0.02 * previous[security] + 0.003 * rng.gauss(0, 1),
                ),
            ]:
                day = month_end_known_at(month)
                rows.append(
                    FeatureObservation(
                        observation_id=f"{source}_{security}_{month}",
                        security_id=security,
                        source_id=source,
                        metric=metric,
                        period_end=day,
                        available_at=day,
                        version_known_at=day,
                        value=value,
                    )
                )
            previous[security] = traffic * spend
    cards = []
    for source in ["traffic", "spend", "earnings", "returns"]:
        card = fixture_manifest("unused.csv")["datasets"]["fixture_panel"][
            "source_card"
        ]
        card.update(source_id=source, dataset_id=source)
        cards.append(card)
    service = CompositionService(
        tmp_path / "compositions.sqlite",
        papers=PaperService(
            PaperStore(tmp_path / "papers.sqlite"), clock=lambda: NOW
        ),
        namespace="pod_fundamentals",
        clock=lambda: NOW,
        code_digest="a" * 64,
    )
    service.register_grant(
        dict(
            grant_id="retail",
            namespace="pod_fundamentals",
            code_digest="a" * 64,
            source_metrics={
                "traffic": ["visits"],
                "spend": ["spend"],
                "earnings": ["margin"],
                "returns": ["return"],
            },
            source_cards=cards,
            targets={
                "next_margin": dict(
                    source_id="earnings", metric="margin", horizon_months=1
                )
            },
            formation_dates=[
                month_end_known_at(m).isoformat()
                for m in range(first, first + 8)
            ],
            protected_from=dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc),
            security_ids=list(previous),
        )
    )
    spec = dict(
        composition_name="discount_margin",
        grant_id="retail",
        reasoning_mode="own_reasoning",
        question="Does the traffic and paid-demand interaction predict future margin?",
        mechanism="Discount-supported demand may change profit per transaction.",
        own_reasoning="The combination distinguishes composition effects from organic demand.",
        dependencies=[
            dict(dependency_id=s, role="core", source_id=s, metric=m)
            for s, m in [
                ("traffic", "visits"),
                ("spend", "spend"),
                ("earnings", "margin"),
            ]
        ],
        target_id="next_margin",
        nodes=[
            dict(
                node_id=n,
                operation="input",
                recipe=dict(
                    feature_id=n, source_id=s, metric=m, operation="latest"
                ),
            )
            for n, s, m in [
                ("traffic", "traffic", "visits"),
                ("spend", "spend", "spend"),
                ("persistence", "earnings", "margin"),
            ]
        ]
        + [
            dict(
                node_id="combined",
                operation="product",
                inputs=["traffic", "spend"],
            )
        ],
        outputs=["combined"],
        baselines=["persistence"],
        causal_edges=[
            (
                "traffic",
                "margin",
                "promotions may jointly explain volume and profit",
            )
        ],
        applicability="Only entities with matched observed traffic and spend.",
        competing_explanations=[
            "Panel composition rather than economics changed."
        ],
        falsification=[
            "No independent increment over individual measurements or persistence."
        ],
    )
    plan = service.submit(spec)
    design = [o for o in rows if o.period_end.date() <= dt.date(2015, 9, 30)]
    check = [o for o in rows if o.period_end.date() >= dt.date(2016, 1, 1)]
    snapshots = lambda data: {
        s: observation_digest([o for o in data if o.source_id == s])
        for s in ["traffic", "spend", "earnings", "returns"]
    }
    service.bind_inputs(plan["composition_id"], snapshots(design))
    service.run(plan["composition_id"], lambda: design)
    registry = Registry(tmp_path / "registry.sqlite")
    policy = IntegrityPolicy(policy_id="composed_fixture", frozen_at=NOW)
    integrity = IntegrityService(
        registry=registry,
        ledger=ResearchLedger(registry),
        evaluator=RegisteredEvaluator(tmp_path / "evaluation.sqlite"),
        policy=policy,
        source_cards=[
            SourceCard.model_validate(c)
            for c in service.get_plan(plan["composition_id"])["grant"][
                "source_cards"
            ]
        ],
        clock=lambda: NOW,
        evidence_seal=CheckAuthority().evidence_seal(),
        allow_unbound_fixture=False,
    )
    ctx = RoleContext(
        "integrity", "integrity", "composed_run", NOW, ("design", "check")
    )
    bridge = CompositionBridge(service, integrity, ctx=ctx, clock=lambda: NOW)
    review = dict(
        composition_id=plan["composition_id"],
        output_node="combined",
        baseline_node="persistence",
        mechanism_family="retail_margin",
        return_source_id="returns",
        return_metric="return",
        design_end=dt.date(2015, 9, 30),
        check_dates=[
            month_end_known_at(m) for m in range(first + 12, first + 18)
        ],
        design_snapshots=snapshots(design),
        check_snapshots=snapshots(check),
        short_eligible_security_ids=list(previous),
        cost_bps_one_way=1,
        cost_source="Explicit synthetic fixture cost assumption.",
        certificate_scope="fixture",
    )
    return service, bridge, review, design, check, integrity, registry, tmp_path


def test_exact_composition_reaches_sealed_check_and_scoped_publication(world):
    service, bridge, review, design, check, integrity, registry, tmp_path = (
        world
    )
    frozen = bridge.freeze(review)
    result = bridge.challenge(
        frozen["candidate_id"],
        design_loader=lambda: design,
        check_loader=lambda: check,
    )
    assert result["status"] == "completed", result
    assert result["admitted"], result
    assert result["decision"]["scope"] == "fixture"
    assert result["return_calibration"]["slope"] > 0
    again = bridge.challenge(
        frozen["candidate_id"],
        design_loader=lambda: pytest.fail("no repeated design"),
        check_loader=lambda: pytest.fail("no second check"),
    )
    assert again == result
    cutoff = month_end_known_at(month_index(dt.date(2026, 9, 1)))
    current = [
        o.model_copy(
            update={
                "observation_id": "current_" + o.observation_id,
                "period_end": cutoff,
                "available_at": cutoff,
                "version_known_at": cutoff,
            }
        )
        for o in design
        if o.source_id != "returns"
        and o.period_end.date() == dt.date(2015, 8, 31)
    ]
    publication = PublicationService(
        tmp_path / "publications.sqlite", registry=registry
    )
    artifacts = ArtifactIndex(tmp_path / "artifacts.sqlite", clock=lambda: NOW)
    strategy = RoleContext("strategy", "pod_fundamentals", "composed_run", NOW)
    batch = bridge.publish(
        frozen["candidate_id"],
        information_cutoff=cutoff,
        observations=current,
        publication=publication,
        artifacts=artifacts,
        published_at=NOW,
        ctx=strategy,
    )
    assert batch.forecast_count == 40 and batch.scope == "fixture"
    forecasts = publication.export_forecasts(strategy)
    assert all(
        f.target.unit == "centered_rank"
        and f.return_forecast.unit == "decimal_return"
        for f in forecasts
    )
    assert all(f.return_forecast.horizon_trading_days == 21 for f in forecasts)
    assert all(abs(float(f.target.prediction)) <= 0.5 for f in forecasts)
    future = current[0].model_copy(
        update={
            "observation_id": "future",
            "available_at": NOW + dt.timedelta(days=20),
        }
    )
    with pytest.raises(ValueError, match="future outcomes"):
        bridge.publish(
            frozen["candidate_id"],
            information_cutoff=cutoff,
            observations=current + [future],
            publication=publication,
            artifacts=artifacts,
            published_at=NOW,
            ctx=strategy,
        )


def test_failed_check_is_retained_and_does_not_reread(world):
    _, bridge, review, design, check, _, _, _ = world
    frozen = bridge.freeze(review)
    changed = [check[0].model_copy(update={"value": 999})] + check[1:]
    failed = bridge.challenge(
        frozen["candidate_id"],
        design_loader=lambda: design,
        check_loader=lambda: changed,
    )
    assert (
        failed["status"] == "failed"
        and failed["check_opened"]
        and not failed["admitted"]
    )
    assert (
        bridge.challenge(
            frozen["candidate_id"],
            design_loader=lambda: pytest.fail("no rerun"),
            check_loader=lambda: pytest.fail("no rerun"),
        )
        == failed
    )


def test_failed_design_does_not_open_protected_data(world):
    service, bridge, review, design, _, _, _, _ = world
    # Selecting the same transform as candidate and baseline cannot add evidence.
    plan = service.get_plan(review["composition_id"])
    revised = {
        **plan["spec"],
        "composition_name": "no_increment",
        "outputs": ["persistence"],
        "baselines": ["persistence", "traffic"],
    }
    registered = service.submit(revised)
    frozen = bridge.freeze(
        {
            **review,
            "composition_id": registered["composition_id"],
            "output_node": "persistence",
            "baseline_node": "persistence",
        }
    )
    result = bridge.challenge(
        frozen["candidate_id"],
        design_loader=lambda: design,
        check_loader=lambda: pytest.fail(
            "rejected design cannot read protected data"
        ),
    )
    assert result["status"] == "design_rejected"
    assert not result["admitted"] and not result["check_opened"]
