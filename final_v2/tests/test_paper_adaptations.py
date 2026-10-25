"""Paper -> adaptation -> registration -> real builtin executor via the real queue worker, plus adversarial rejections.

Everything is synthetic: a local text file as the 'paper' and the builtin fixture panel (random numbers). The executor really
runs; the resulting metrics prove only that the machinery works and say nothing about any strategy or about LLM skill.
"""

import datetime as dt
import math

import pytest

from pod_platform.agents.durable_cache import DurableExtractionCache
from pod_platform.agents.provider import DeterministicMockProvider
from pod_platform.papers import PaperService, PaperStore
from pod_platform.papers.adaptations import AdaptationRejected, PaperAdapter
from pod_platform.pods import ExecutorRegistry, PodResearchRunner, pod_spec
from pod_platform.pods.runner import PodWorker
from pod_platform.research.builtins import register_executors, write_fixture_dataset
from pod_platform.research.builtins.catalog import RECIPES
from pod_platform.research.jobs import ResearchQueue
from pod_platform.research.memory import PodMemory
from pod_platform.research.registration import ResearchLedger
from pod_platform.research.registry import Registry

UTC = dt.timezone.utc
ASOF = dt.datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
NOW = ASOF + dt.timedelta(hours=1)
NS = "pod_valuation"
RECIPE = "valuation_residual_return_v1"
TEXT = ("Cheap stocks relative to fundamentals earn higher subsequent returns.\n"
        "The mechanism is slow-moving capital that corrects mispricing over several months.\n")


class Env:
    def __init__(self, tmp):
        self.tmp = tmp
        self.executors = ExecutorRegistry()
        self.ctx = register_executors(self.executors, manifest_path=write_fixture_dataset(tmp / "data"), state_dir=tmp / "state",
                                      clock=lambda: NOW, research_asof=ASOF)
        self.papers = PaperService(PaperStore(tmp / "papers.sqlite"), clock=lambda: NOW)
        self.registry = Registry(tmp / "registry.sqlite")
        self.memory = PodMemory(tmp / "mem", NS)
        self.adapter = self.make_adapter()
        self.card = self.paper("paper.txt", dt.datetime(2020, 1, 1, tzinfo=UTC))
        self.claims = {}
        for kind, quote in (("economic_claim", TEXT.split("\n")[0]), ("mechanism", TEXT.split("\n")[1])):
            card = self.papers.add_claim(self.card.paper_id, kind=kind, quote=quote)
            self.claims[kind] = next(c.claim_id for c in card.claims if c.kind == kind)

    def make_adapter(self, ledger=None):
        return PaperAdapter(self.tmp / "adapt.sqlite", papers=self.papers, registry=self.registry, ledger=ledger or ResearchLedger(self.registry),
                            memory=self.memory, context=self.ctx, clock=lambda: NOW)

    def paper(self, name, version_date, text=TEXT):
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return self.papers.ingest_local(path, title=f"Value spreads {name}", first_public_date=version_date, version_date=version_date)

    def review(self, **over):
        base = {
            "namespace": NS, "paper_id": self.card.paper_id, "version_key": self.card.version_key, "label": "MECHANISM_TRANSFER",
            "claim_ids": list(self.claims.values()),
            "deviations": ["Monthly panel replaces the paper's daily sample; sector peers are GICS-prefix based."],
            "comparison_plan": "Compare the adapted recipe with sector-relative book-to-market on design-partition months.",
            "baseline": "Sector-relative book-to-market rank", "recipe_id": RECIPE, "recipe_digest": RECIPES[RECIPE].digest(),
            "dataset_id": "fixture_panel", "method_id": "ridge", "feature_set": "core", "ridge_lambda": 0.05,
            "economic_effect": "Cheap stocks earn higher residual returns over three months.", "declared_sign": "positive",
            "falsification": ["Rank IC does not beat the sector-relative value baseline on design."],
            "selected_by": "implementation", "reviewed_by": "reviewer_1", "reviewed_at": ASOF.isoformat(),
        }
        base.update(over)
        return {k: v for k, v in base.items() if v is not None}


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def test_registers_before_execution_then_real_worker_runs_real_executor(env):
    rec = env.adapter.adapt(env.review())
    rid = rec["registration_id"]
    assert rec["status"] == "registered_not_executed" and rec["admitted"] is False and rec["fixture_data"] is True
    assert env.registry.count("hypotheses") == 1 and env.registry.count("experiment_specs") == 1
    assert env.registry.get("experiment_specs", rec["experiment_spec_id"])["identity"]["hypothesis_id"] == rec["hypothesis_id"]
    assert env.ctx.report(rid, "adapt_v1") is None and not env.memory.evaluated_results()  # nothing ran, no data was read
    assert [s["stage"] for s in env.adapter.lineage(rec["adaptation_id"])] == ["source", "method", "hypothesis", "experiment"]
    assert env.adapter.adapt(env.review()) == rec and env.registry.count("hypotheses") == 1  # idempotent
    queue = ResearchQueue(env.tmp / "q.sqlite", clock=lambda: NOW)
    job = env.adapter.submit_job(rec["adaptation_id"], queue)
    assert env.adapter.submit_job(rec["adaptation_id"], queue).job_id == job.job_id
    assert job.kind == "experiment" and job.payload == {"draft_entry_id": rec["draft_entry_id"], "variants": ["adapt_v1"]}
    runner = PodResearchRunner(pod_spec(NS), DeterministicMockProvider(lambda request: "{}"), DurableExtractionCache(env.tmp / "cache.sqlite"), env.memory)
    outcome = PodWorker(queue, {NS: runner}, env.executors).run_once()
    assert outcome.state == "succeeded" and outcome.result["admitted"] is False
    result = env.adapter.record_result(rec["adaptation_id"])
    chain = {s["stage"]: s for s in result["lineage"]}
    assert list(chain) == ["source", "method", "hypothesis", "experiment", "result"] and result["admitted"] is False
    assert chain["source"]["ref_hash"] == env.papers.get(env.card.paper_id).card_hash
    assert chain["method"]["ref_hash"] == RECIPES[RECIPE].digest() and chain["experiment"]["registration_id"] == rid
    assert chain["result"]["partition"] == "design" and chain["result"]["fixture"] is True and chain["result"]["report_hash"]
    assert all(math.isfinite(v) for v in chain["result"]["metrics"].values()) and "belief_ic_candidate" in chain["result"]["metrics"]
    assert chain["result"]["executor_spec_hash"] == env.ctx.report(rid, "adapt_v1")["spec_hash"]  # bound to the executor's own report
    # retry retains hash bindings and budget: same entry, no second execution, no new registration
    again = env.adapter.execute_design(rec["adaptation_id"], env.executors)
    assert again["lineage"] == result["lineage"] and len(env.memory.evaluated_results()) == 1
    assert env.registry.count("experiment_specs") == 1


def test_restart_with_fresh_process_local_ledger_keeps_registration(env):
    rec = env.adapter.adapt(env.review())
    fresh = env.make_adapter(ledger=ResearchLedger(env.registry))  # new process: the in-memory ledger starts empty
    assert fresh.adapt(env.review()) == rec
    assert fresh.execute_design(rec["adaptation_id"], env.executors)["lineage"][-1]["stage"] == "result"


REJECTIONS = [
    ("incomplete_review", {"deviations": None}),
    ("incomplete_review", {"claim_ids": []}),
    ("incomplete_review", {"selected_by": "model"}),
    ("unreviewed_recipe", {"recipe_digest": "0" * 64}),
    ("unreviewed_recipe", {"recipe_id": "arbitrary_module_recipe"}),
    ("unresolved_source", {"dataset_id": "unknown_panel"}),
    ("ungrounded_claims", {"claim_ids": ["deadbeefdeadbeef"]}),
    ("version_mismatch", {"version_key": "v9"}),
    ("future_time", {"reviewed_at": (NOW + dt.timedelta(days=1)).isoformat()}),
    ("namespace_mismatch", {"namespace": "pod_events"}),
    ("future_version", {"research_mode": "historical", "as_of": dt.datetime(2019, 1, 1, tzinfo=UTC).isoformat()}),
]


@pytest.mark.parametrize("code,override", REJECTIONS)
def test_rejections_leave_no_registration_draft_or_result(env, code, override):
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.adapt(env.review(**override))
    assert exc.value.code == code
    assert env.registry.count("hypotheses") == 0 and env.registry.count("experiment_specs") == 0
    assert env.memory.count() == 0 and env.ctx.report(f"{NS}:x", "adapt_v1") is None


def test_wrong_charter_recipe_future_paper_version_and_missing_full_text_are_rejected(env):
    dyn = RECIPES["dynamics_reversal_liquidity_v1"]
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.adapt(env.review(recipe_id=dyn.recipe_id, recipe_digest=dyn.digest()))
    assert exc.value.code == "unreviewed_recipe"
    future = env.paper("future.txt", NOW + dt.timedelta(days=30), text=TEXT + "Future edition.\n")
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.adapt(env.review(paper_id=future.paper_id, version_key=future.version_key))
    assert exc.value.code == "future_version"
    bare = env.paper("bare.txt", dt.datetime(2020, 6, 1, tzinfo=UTC), text=TEXT + "Bare edition.\n")
    env.papers.store.save_card(bare.model_copy(update={"text_status": "incomplete:no_permitted_full_text_source"}))
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.adapt(env.review(paper_id=bare.paper_id, version_key=bare.version_key))
    assert exc.value.code == "fulltext_absent"
    only_economic = {"claim_ids": [env.claims["economic_claim"]]}  # MECHANISM_TRANSFER needs a grounded mechanism claim too
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.adapt(env.review(**only_economic))
    assert exc.value.code == "ungrounded_claims" and env.registry.count("hypotheses") == 0


def test_claim_text_cannot_be_forged_after_the_fact(env):
    """A tampered stored text makes verify_claims fail, so adaptation refuses rather than trusting the claim."""
    stored = env.papers.get(env.card.paper_id)
    with env.papers.store._db:  # direct tampering with the stored text, bypassing the service
        env.papers.store._db.execute("UPDATE contents SET text=? WHERE content_hash=?", ("Completely different text.", stored.content_hash))
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.adapt(env.review())
    assert exc.value.code in ("ungrounded_claims", "fulltext_absent")


def test_nothing_executes_without_a_matching_registration(env):
    rec = env.adapter.adapt(env.review())
    ran = []
    trap = ExecutorRegistry()
    trap.register(rec["executor_id"], lambda request: ran.append(request) or (_ for _ in ()).throw(AssertionError("must not run")))
    empty_registry = Registry(env.tmp / "other_registry.sqlite")  # same adaptation rows, no registered specification
    stranger = PaperAdapter(env.tmp / "adapt.sqlite", papers=env.papers, registry=empty_registry, ledger=ResearchLedger(empty_registry),
                            memory=env.memory, context=env.ctx, clock=lambda: NOW)
    with pytest.raises(AdaptationRejected) as exc:
        stranger.execute_design(rec["adaptation_id"], trap)
    assert exc.value.code == "unregistered" and not ran
    with pytest.raises(AdaptationRejected):
        stranger.submit_job(rec["adaptation_id"], ResearchQueue(env.tmp / "q2.sqlite", clock=lambda: NOW))
    with pytest.raises(AdaptationRejected) as exc:
        env.adapter.record_result(rec["adaptation_id"])
    assert exc.value.code == "no_result"
