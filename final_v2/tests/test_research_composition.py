"""Multiple papers, original reasoning and alt measurements share a native test."""

import datetime as dt
import json
import random
import sqlite3

import pytest

from pod_platform.papers import PaperService, PaperStore
from pod_platform.research.builtins.fixtures import fixture_manifest
from pod_platform.research.builtins.timeutil import (
    month_index,
    month_end_known_at,
)
from pod_platform.research.composition import (
    CompositionService,
    observation_digest,
)
from pod_platform.research.features import FeatureObservation

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)


@pytest.fixture
def env(tmp_path):
    papers = PaperService(
        PaperStore(tmp_path / "papers.sqlite"), clock=lambda: NOW
    )
    selections = []
    for index, text in enumerate(
        [
            "Traffic and paid demand can disagree about margins.",
            "A conditional interaction distinguishes promotion from adoption.",
        ]
    ):
        path = tmp_path / f"paper{index}.txt"
        path.write_text(text)
        card = papers.ingest_local(
            path,
            title=f"Fixture paper {index}",
            first_public_date=dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc),
            version_date=dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc),
        )
        card = papers.add_claim(card.paper_id, kind="mechanism", quote=text)
        selections.append(
            dict(
                namespace="pod_fundamentals",
                component_kind="mechanism",
                component_name=text,
                paper_id=card.paper_id,
                version_key=card.version_key,
                claim_ids=[card.claims[0].claim_id],
                source="full_text",
            )
        )
    first = month_index(dt.date(2015, 1, 1))
    randomizer = random.Random(42)
    rows = []
    previous = {str(i): randomizer.gauss(0, 1) for i in range(20)}
    for month in range(first, first + 7):
        for security in previous:
            traffic, spend = randomizer.gauss(0, 1), randomizer.gauss(0, 1)
            for source, metric, value in [
                ("traffic", "visits", traffic),
                ("spend", "spend", spend),
                ("earnings", "margin", previous[security]),
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
    for source in ["traffic", "spend", "earnings"]:
        card = fixture_manifest("unused.csv")["datasets"]["fixture_panel"][
            "source_card"
        ]
        card.update(
            source_id=source,
            dataset_id=source,
            formation_eligible=False,
            status="SAMPLED",
        )
        cards.append(card)
    grant = dict(
        grant_id="retail",
        namespace="pod_fundamentals",
        code_digest="a" * 64,
        source_metrics={
            "traffic": ["visits"],
            "spend": ["spend"],
            "earnings": ["margin"],
        },
        source_cards=cards,
        targets={
            "next_margin": dict(
                source_id="earnings", metric="margin", horizon_months=1
            )
        },
        formation_dates=[
            month_end_known_at(m).isoformat() for m in range(first, first + 6)
        ],
        security_ids=list(previous),
    )
    spec = dict(
        composition_name="margin_disagreement",
        grant_id="retail",
        reasoning_mode="cross_source_composition",
        question="Do traffic and spend jointly predict next native margins?",
        mechanism="Promotions can purchase demand and alter margins differently from organic adoption.",
        own_reasoning="Testing the interaction can falsify a volume-only interpretation.",
        paper_components=selections,
        dependencies=[
            dict(
                dependency_id=s,
                source_id=s,
                metric=m,
                role="core",
                request_id="request_" + s,
            )
            for s, m in [
                ("traffic", "visits"),
                ("spend", "spend"),
                ("earnings", "margin"),
            ]
        ]
        + [
            dict(
                dependency_id="promotions",
                source_id="promotions",
                metric="discount",
                role="optional_control",
                missing_reason="not available",
            )
        ],
        target_id="next_margin",
        nodes=[
            dict(
                node_id="traffic",
                operation="input",
                recipe=dict(
                    feature_id="visits",
                    source_id="traffic",
                    metric="visits",
                    operation="latest",
                ),
            ),
            dict(
                node_id="spend",
                operation="input",
                recipe=dict(
                    feature_id="spend",
                    source_id="spend",
                    metric="spend",
                    operation="latest",
                ),
            ),
            dict(
                node_id="persistence",
                operation="input",
                recipe=dict(
                    feature_id="margin",
                    source_id="earnings",
                    metric="margin",
                    operation="latest",
                ),
            ),
            dict(
                node_id="interaction",
                operation="product",
                inputs=["traffic", "spend"],
            ),
        ],
        outputs=["interaction"],
        baselines=["persistence"],
        causal_edges=[
            (
                "traffic",
                "next_margin",
                "volume and discounts jointly affect margins",
            )
        ],
        applicability="Comparable retail businesses with directly observed traffic and spend.",
        competing_explanations=[
            "Source panel composition changed instead of company economics."
        ],
        falsification=[
            "Interaction does not improve on either component or the persistence baseline."
        ],
    )
    service = CompositionService(
        tmp_path / "compositions.sqlite",
        papers=papers,
        namespace="pod_fundamentals",
        clock=lambda: NOW,
        code_digest="a" * 64,
        dependency_lookup=lambda ns, rid: {
            "namespace": ns,
            "request_id": rid,
            "status": "data_ready",
        },
    )
    service.register_grant(grant)
    return service, spec, grant, rows


def bind(service, composition_id, rows):
    return service.bind_inputs(
        composition_id,
        {
            s: observation_digest([o for o in rows if o.source_id == s])
            for s in ["traffic", "spend", "earnings"]
        },
    )


def test_multiple_papers_plus_own_reasoning_measure_real_interaction(env):
    service, spec, grant, rows = env
    plan = service.submit(spec)
    assert len(plan["evidence"]) == 2
    assert service.submit(spec) == plan
    bind(service, plan["composition_id"], rows)

    def loader():
        with sqlite3.connect(service.path) as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM compositions WHERE kind='trial'"
                ).fetchone()[0]
                == 1
            )
        return rows

    result = service.run(plan["composition_id"], loader)
    assert result["status"] == "measured"
    interaction = next(
        r for r in result["comparisons"] if r["candidate"] == "interaction"
    )
    assert all(
        d["candidate_ic"] == pytest.approx(1) for d in interaction["dates"]
    )
    assert {"traffic", "spend", "interaction"} <= {
        r["candidate"] for r in result["comparisons"]
    }
    assert result["data_dependencies"]["omitted_controls"] == ["promotions"]
    assert (
        not result["admitted"]
        and result["orders"] == result["protected_check_accesses"] == 0
    )
    assert (
        service.run(
            plan["composition_id"],
            lambda: pytest.fail("repeat must not read data"),
        )
        == result
    )
    frozen = service.integrated_plan(plan["composition_id"])
    assert frozen["spec"]["nodes"] == spec["nodes"] or len(
        frozen["spec"]["nodes"]
    ) == len(spec["nodes"])
    assert (
        frozen["required_gates"]
        and frozen["input_snapshots"]
        and not frozen["admitted"]
    )


def test_missing_core_waits_without_spending_trial_optional_does_not_block(env):
    service, spec, _, rows = env
    plan = service.submit(spec)
    waiting = service.run(
        plan["composition_id"],
        lambda: pytest.fail("missing core cannot read rows"),
    )
    assert waiting["status"] == "waiting_data"
    assert waiting["missing_core"] == ["traffic", "spend", "earnings"]
    bind(service, plan["composition_id"], rows)
    assert (
        service.dependency_status(plan["composition_id"])["status"] == "ready"
    )


def test_new_hypothesis_needs_no_papers_and_no_full_replication(env):
    service, spec, _, rows = env
    spec.update(paper_components=[], reasoning_mode="own_reasoning")
    plan = service.submit(spec)
    bind(service, plan["composition_id"], rows)
    assert (
        service.run(plan["composition_id"], lambda: rows)["status"]
        == "measured"
    )


def test_asof_predictions_ignore_future_inputs_and_revisions(env):
    service, spec, grant, rows = env
    plan = service.submit(spec)
    day = dt.datetime.fromisoformat(grant["formation_dates"][0])
    before = service.predict(plan["composition_id"], rows, day)
    changed = [
        o.model_copy(update={"value": 999999}) if o.period_end > day else o
        for o in rows
    ]
    original = next(
        o for o in rows if o.source_id == "traffic" and o.period_end == day
    )
    changed.append(
        original.model_copy(
            update={
                "observation_id": "later_revision",
                "version_known_at": NOW,
                "value": 999999,
            }
        )
    )
    assert service.predict(plan["composition_id"], changed, day) == before


def test_bound_snapshot_change_fails_and_remains_counted(env):
    service, spec, _, rows = env
    plan = service.submit(spec)
    bind(service, plan["composition_id"], rows)
    changed = [rows[0].model_copy(update={"value": 999})] + rows[1:]
    result = service.run(plan["composition_id"], lambda: changed)
    assert (
        result["status"] == "failed" and "snapshot changed" in result["error"]
    )
    assert service.run(plan["composition_id"], lambda: rows) == result
    with sqlite3.connect(service.path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM compositions WHERE kind='trial'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "change",
    [
        {"outputs": ["unknown"]},
        {"variant_budget": 1},
        {"target_id": "invented_sentiment"},
        {"paper_components": [], "own_reasoning": None},
        {"nodes": [dict(node_id="bad", operation="exec", inputs=["traffic"])]},
    ],
)
def test_ungranted_incomplete_unsafe_or_unbounded_spec_is_rejected(env, change):
    service, spec, _, _ = env
    with pytest.raises(ValueError):
        service.submit({**spec, **change})


def test_core_cannot_be_downgraded_to_optional_to_run_unrelated_proxy(env):
    service, spec, _, _ = env
    spec["dependencies"][0]["role"] = "optional_control"
    with pytest.raises(ValueError, match="core dependency"):
        service.submit(spec)
