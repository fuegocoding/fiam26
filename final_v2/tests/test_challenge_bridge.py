"""Adversarial tests for the design-candidate -> protected check -> certificate/publication bridge (synthetic fixture only).

Nothing here is investment evidence: the panel is generated with a planted, mostly non-persistent signal so that the
existing IntegrityService criteria (placebo, stability, multiplicity) can actually be met or failed by measurement.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import random
from types import SimpleNamespace

import pytest

from pod_platform.application.artifacts import ArtifactIndex
from pod_platform.application.integrity import IntegrityPolicy, IntegrityService, Outcome
from pod_platform.application.namespaces import RoleContext
from pod_platform.application.publication import PublicationService
from pod_platform.contracts.errors import PodPlatformError
from pod_platform.pods.runner import ExecutorRegistry
from pod_platform.research.builtins import register_executors
from pod_platform.research.builtins.catalog import RECIPES, VariantConfig
from pod_platform.research.builtins.engine import score_frozen
from pod_platform.research.builtins.errors import BuiltinConfigError
from pod_platform.research.builtins.fixtures import fixture_manifest, generate_rows, write_csv
from pod_platform.research.builtins.manifest import ApprovedDataset
from pod_platform.research.challenge_bridge import ChallengeBridge, CheckAuthority
from pod_platform.research.evaluation import RegisteredEvaluator
from pod_platform.research.registration import ResearchLedger
from pod_platform.research.registry import Registry

from .test_builtin_executors import ASOF, NOW, make_request

UTC = dt.timezone.utc
HOUR = dt.timedelta(hours=1)
CHECK_START, CHECK_END = dt.date(2019, 1, 1), dt.date(2019, 6, 30)
EVAL_ASOF = ASOF + 2 * HOUR


def summary(exc) -> str:
    return exc.value.response.summary


def build_world(tmp, *, signal=True, variants=(("v1", {}),), multiplicity=None, policy=None, authority=None):
    """Executor runs (design partition) on a fixture panel, then freeze every variant as a candidate."""
    rows = generate_rows(seed=7, ret_signal=0.0)
    rng = random.Random(11)
    for row in rows:  # planted: reversal (iid-ish) + an iid second factor; baseline only sees reversal
        row["ret_12_1"] = f"{0.25 * rng.gauss(0, 1):.10g}"
        a, b = -float(row["ret_1_0"]) / 0.07, float(row["ret_12_1"]) / 0.25
        row["ret_exc_lead1m"] = f"{(0.01 * (a + b) if signal else 0.0) + 0.012 * rng.gauss(0, 1):.10g}"
    data = tmp / "data"
    data.mkdir()
    write_csv(rows, data / "panel.csv")
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
    registry = ExecutorRegistry()
    bctx = register_executors(registry, manifest_path=manifest, state_dir=tmp / "state", clock=lambda: NOW, research_asof=ASOF)
    for name, extra in variants:
        registry.run(make_request(name=name, method_id="ridge", ridge_lambda=0.05, **extra))
    authority = authority or CheckAuthority()
    bridge = ChallengeBridge(tmp / "bridge", authority=authority)
    policy = policy or IntegrityPolicy(policy_id="fixture_challenge_v1", frozen_at=ASOF + HOUR)
    integrity = RoleContext("integrity", "integrity", "run1", EVAL_ASOF, ("design", "check"))
    w = SimpleNamespace(tmp=tmp, manifest=manifest, bctx=bctx, authority=authority, bridge=bridge, policy=policy, integrity=integrity,
                        frozen=[])
    mult = multiplicity or len(variants)
    for name, _ in variants:
        w.frozen.append(bridge.freeze_candidate(integrity, bctx, "pod_dynamics:draft1", name, policy=policy, registered_at=ASOF, multiplicity=mult))
    return w


def make_service(w, cls=IntegrityService, policy=None, tag="", **kwargs):
    """One persistent-registry service per world (a second in-memory ledger would re-issue the same trial counter ids)."""
    cacheable = cls is IntegrityService and policy is None and not tag and not kwargs
    if cacheable and hasattr(w, "service"):
        return w.service
    registry = Registry(w.tmp / f"integrity_registry_{cls.__name__}_{policy.policy_id if policy else 'base'}{tag}.sqlite")
    kwargs = {"evidence_seal": w.authority.evidence_seal(), "allow_unbound_fixture": False, **kwargs}
    service = cls(registry=registry, ledger=ResearchLedger(registry), evaluator=RegisteredEvaluator(w.tmp / f"integrity_eval{tag}.sqlite"),
                  policy=policy or w.policy, source_cards=[w.bridge.integrity_source_card(w.bctx, w.frozen[0])], **kwargs)
    if cacheable:
        w.service = service
    return service


def authorize(w, frozen=None, *, start=CHECK_START, end=CHECK_END, budget=1, issued=None, authority=None, ctx=None):
    frozen = frozen or w.frozen[0]
    return (authority or w.authority).issue(ctx or w.integrity, frozen, check_start=start, check_end=end,
                                            issued_at=issued or ASOF + 90 * dt.timedelta(minutes=1),
                                            expires_at=EVAL_ASOF + dt.timedelta(days=1), max_family_accesses=budget)


_DEFAULT = object()


def challenge(w, service=None, frozen=None, auth=_DEFAULT, ctx=None, asof=EVAL_ASOF, bridge=None):
    """Only a truly omitted ``auth`` is replaced by a valid one; None / junk are passed through to the public API unchanged."""
    frozen = frozen or w.frozen[0]
    auth = authorize(w, frozen) if auth is _DEFAULT else auth
    return (bridge or w.bridge).run_challenge(ctx or w.integrity, w.bctx, service or make_service(w), frozen.candidate_id, auth,
                                              evaluation_asof=asof)


# ------------------------------------------------------------------ positive end to end


def test_frozen_candidate_reaches_challenge_certificate_and_publication(tmp_path):
    w = build_world(tmp_path)
    frozen = w.frozen[0]
    assert frozen.fixture and frozen.scope == "fixture" and frozen.proxy_labels
    assert w.bridge.family_accesses(frozen.mechanism_family) == 0, "freezing must not open the check window"
    service = make_service(w)
    auth = authorize(w, frozen)
    result = challenge(w, service, frozen, auth)
    decision = result.decision
    assert decision.outcome == Outcome.ADMITTED, decision.reasons
    assert decision.scope == "fixture" and decision.labels["synthetic"] and decision.certificate_id
    assert result.check_opened and len(result.hidden["check_dates"]) >= 4 and result.hidden["design_rows"] > 1000
    assert all(decision.criteria.values()) and decision.metrics["check_placebo_ic"] is not None
    assert w.bridge.family_accesses(frozen.mechanism_family) == 1
    certificate = service.certificate(frozen.candidate_id)
    assert certificate.verify_seal() and certificate.model_hash == frozen.model_hash
    # the model was frozen from design months only
    art = json.loads(w.bctx.candidate_path(frozen.spec_hash).read_text(encoding="utf-8"))
    last_train = max(art["frozen_model"]["train_months"])
    assert last_train <= 2018 * 12 + 11 and art["frozen_model"]["method_id"] == "ridge"

    publication = PublicationService(tmp_path / "pub.sqlite", registry=Registry(tmp_path / "pub_registry.sqlite"))
    artifacts = ArtifactIndex(tmp_path / "artifacts.sqlite")
    strategy = RoleContext("strategy", frozen.namespace, "run1", EVAL_ASOF)
    at = EVAL_ASOF + dt.timedelta(days=1)
    batch = w.bridge.publish_forecasts(strategy, w.bctx, service, publication, artifacts, frozen.candidate_id, auth, published_at=at)
    assert batch.scope == "fixture" and batch.forecast_count >= 40 and batch.admission_certificate_id == certificate.certificate_id
    again = w.bridge.publish_forecasts(strategy, w.bctx, service, publication, artifacts, frozen.candidate_id, auth, published_at=at)
    assert again.batch_id == batch.batch_id
    published = publication.export_forecasts(strategy)
    assert len(published) == batch.forecast_count and all(f.model_id == frozen.model_id for f in published)
    # a different strategy namespace may not publish this model
    with pytest.raises(PodPlatformError):
        w.bridge.publish_forecasts(RoleContext("strategy", "pod_other", "r", EVAL_ASOF), w.bctx, service, publication, artifacts,
                                   frozen.candidate_id, auth, published_at=at)


def test_retry_is_idempotent_and_never_reopens_the_check(tmp_path):
    w = build_world(tmp_path)
    service, auth = make_service(w), authorize(w)
    first = challenge(w, service, auth=auth)
    second = challenge(w, service, auth=auth)  # the stored decision answers; the evaluator is not consulted again
    assert second.decision.as_dict() == first.decision.as_dict() and second.hidden == first.hidden
    assert w.bridge.family_accesses(w.frozen[0].mechanism_family) == 1
    assert len(service.ledger.outcome_accesses()) == 1


def test_failure_is_persisted_and_the_spent_access_is_not_refunded(tmp_path):
    class Boom(IntegrityService):
        def evaluate(self, *a, **k):
            raise ValueError("synthetic evaluator crash")

    w = build_world(tmp_path)
    auth = authorize(w)
    with pytest.raises(ValueError, match="synthetic evaluator crash"):
        challenge(w, make_service(w, Boom), auth=auth)
    state = w.bridge.run_state(w.frozen[0].candidate_id)
    assert state["state"] == "failed" and "synthetic evaluator crash" in state["failure"]
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, make_service(w), auth=auth)
    assert "previously failed" in summary(exc) and w.bridge.family_accesses(w.frozen[0].mechanism_family) == 1
    assert any(kind == "failed" for _, kind, _ in w.bridge.events(w.frozen[0].candidate_id))


# ------------------------------------------------------------------ no self-admission


def test_design_result_that_is_not_accepted_cannot_be_frozen(tmp_path):
    with pytest.raises(PodPlatformError) as exc:
        build_world(tmp_path, signal=False)
    assert "design result not accepted" in summary(exc)


def test_proxy_target_on_real_data_stays_research_shadow_and_check_rows_stay_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(ApprovedDataset, "fixture", property(lambda self: False))
    w = build_world(tmp_path)
    frozen = w.frozen[0]
    assert frozen.scope == "research" and not frozen.fixture
    service = make_service(w)
    result = challenge(w, service, auth=authorize(w))
    assert result.decision.outcome == Outcome.SHADOW and not result.check_opened and result.decision.certificate_id is None
    assert w.bridge.family_accesses(frozen.mechanism_family) == 0 and service.ledger.outcome_accesses() == []
    publication = PublicationService(tmp_path / "pub.sqlite")
    with pytest.raises(PodPlatformError) as exc:
        w.bridge.publish_forecasts(RoleContext("strategy", frozen.namespace, "r", EVAL_ASOF), w.bctx, service, publication,
                                   ArtifactIndex(tmp_path / "a.sqlite"), frozen.candidate_id, authorize(w), published_at=EVAL_ASOF + 24 * HOUR)
    assert "no stored ADMITTED" in summary(exc)


def test_multiplicity_must_count_every_scored_variant(tmp_path):
    with pytest.raises(PodPlatformError) as exc:
        build_world(tmp_path, variants=(("v1", {}), ("v2", {"feature_set": "extended"})), multiplicity=1)
    assert "multiplicity" in summary(exc)


# ------------------------------------------------------------------ unapproved callers


def test_only_the_integrity_role_with_the_check_partition_may_run_or_authorize(tmp_path):
    w = build_world(tmp_path)
    strategy = RoleContext("strategy", "pod_dynamics", "r", EVAL_ASOF, ("design", "check"))
    with pytest.raises(PodPlatformError):
        w.authority.issue(strategy, w.frozen[0], check_start=CHECK_START, check_end=CHECK_END, issued_at=EVAL_ASOF,
                          expires_at=EVAL_ASOF + 2 * HOUR, max_family_accesses=1)
    auth = authorize(w)
    with pytest.raises(PodPlatformError):
        challenge(w, ctx=strategy, auth=auth)
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, ctx=RoleContext("integrity", "integrity", "r", EVAL_ASOF, ("design",)), auth=auth)
    assert "check partition" in summary(exc)
    assert w.bridge.family_accesses(w.frozen[0].mechanism_family) == 0


def test_forged_tampered_or_foreign_authorizations_are_refused(tmp_path):
    w = build_world(tmp_path, variants=(("v1", {}), ("v2", {"feature_set": "extended"})))
    other_authority = CheckAuthority()
    forged = authorize(w, authority=other_authority)
    widened = dataclasses.replace(authorize(w), max_family_accesses=99)
    wrong_candidate = authorize(w, w.frozen[1])
    for bad in (forged, widened, wrong_candidate, "yes", None):
        with pytest.raises(PodPlatformError):
            challenge(w, auth=bad)  # None reaches run_challenge unchanged (the helper no longer substitutes a valid authorization)
    with pytest.raises(PodPlatformError):  # and the public API itself refuses a missing authorization
        w.bridge.run_challenge(w.integrity, w.bctx, make_service(w), w.frozen[0].candidate_id, None, evaluation_asof=EVAL_ASOF)
    assert w.bridge.family_accesses(w.frozen[0].mechanism_family) == 0
    assert any(kind == "denied" for _, kind, _ in w.bridge.events(w.frozen[0].candidate_id))


# ------------------------------------------------------------------ partition / chronology / leakage


@pytest.mark.parametrize("kwargs, text", [
    ({"start": dt.date(2018, 6, 30)}, "after the design window"),            # overlaps the design partition
    ({"end": dt.date(2027, 1, 31)}, "not mature"),                           # labels unresolved at the clock
    ({"issued": ASOF}, "issued after the freeze"),                           # authorization predates the freeze
])
def test_chronology_overlap_and_label_maturity_are_enforced_before_rows_open(tmp_path, kwargs, text):
    w = build_world(tmp_path)
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, auth=authorize(w, **kwargs))
    assert text in summary(exc)
    assert w.bridge.family_accesses(w.frozen[0].mechanism_family) == 0 and w.bridge.run_state(w.frozen[0].candidate_id) is None


def test_policy_must_be_frozen_after_registration_and_unchanged_before_the_check(tmp_path):
    w = build_world(tmp_path)
    late = IntegrityPolicy(policy_id="fixture_challenge_v1", frozen_at=ASOF + HOUR, min_t_stat=2.5)  # loosened/tightened after freeze
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, make_service(w, policy=late))
    assert "differs from the policy frozen" in summary(exc)
    early = IntegrityPolicy(policy_id="early", frozen_at=ASOF - HOUR)
    with pytest.raises(PodPlatformError) as exc:
        w.bridge.freeze_candidate(w.integrity, w.bctx, "pod_dynamics:draft1", "v1", policy=early, registered_at=ASOF, multiplicity=1)
    assert "frozen" in summary(exc)


def test_check_window_labels_are_matured_only_for_resolved_months(tmp_path):
    w = build_world(tmp_path)
    result = challenge(w)
    # June 2019 has no resolved next-month label inside the window, so it is never scored
    assert "2019-06-30" not in " ".join(result.hidden["check_dates"]) and len(result.hidden["check_dates"]) == 5


# ------------------------------------------------------------------ amendments cannot reuse old checks / provenance drift


def test_data_or_artifact_drift_after_freeze_is_refused(tmp_path):
    w = build_world(tmp_path)
    auth = authorize(w)
    art = w.bctx.candidate_path(w.frozen[0].spec_hash)
    original = art.read_text(encoding="utf-8")
    art.write_text(original.replace('"ybar":', '"ybar": 1.0, "ybar_orig":', 1), encoding="utf-8")
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, auth=auth)
    assert "provenance mismatch" in summary(exc)
    art.write_text(original, encoding="utf-8")
    csv = w.tmp / "data" / "panel.csv"
    csv.write_text(csv.read_text(encoding="utf-8").replace("2019-03-31", "2019-03-30", 1), encoding="utf-8")  # post-freeze edit of check rows
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, auth=auth)
    assert "provenance mismatch" in summary(exc) and w.bridge.family_accesses(w.frozen[0].mechanism_family) == 0


def test_amended_variant_is_a_new_candidate_that_cannot_reuse_the_old_check(tmp_path):
    w = build_world(tmp_path, variants=(("v1", {}), ("v2", {"feature_set": "extended"})))
    old, new = w.frozen
    assert old.candidate_id != new.candidate_id and old.model_hash != new.model_hash and old.digest() != new.digest()
    old_auth = authorize(w, old, budget=1)
    assert challenge(w, make_service(w), old, old_auth).check_opened
    with pytest.raises(PodPlatformError) as exc:  # the old authorization is bound to the old digest
        challenge(w, make_service(w), new, old_auth)
    assert "not bound" in summary(exc) or "another candidate" in summary(exc)
    # the family budget is finite: even a correctly bound authorization cannot open a second look at budget 1
    with pytest.raises(PodPlatformError) as exc:
        challenge(w, make_service(w), new, authorize(w, new, budget=1))
    assert "budget" in summary(exc) and w.bridge.family_accesses(old.mechanism_family) == 1
    assert w.bridge.run_state(new.candidate_id) is None
    # raising the budget is an integrity-authority decision, then the amended candidate gets its own check
    assert challenge(w, make_service(w), new, authorize(w, new, budget=2)).check_opened
    assert w.bridge.family_accesses(old.mechanism_family) == 2


def test_frozen_store_is_append_only(tmp_path):
    w = build_world(tmp_path)
    w.bridge.event(w.frozen[0].candidate_id, "note", "probe")
    with pytest.raises(Exception, match="immutable"):
        w.bridge._db.execute("UPDATE frozen SET digest='x'")
    with pytest.raises(Exception, match="append-only"):
        w.bridge._db.execute("DELETE FROM events")


def test_frozen_model_scoring_rejects_a_mismatched_model(tmp_path):
    w = build_world(tmp_path)
    recipe = RECIPES[w.frozen[0].recipe_id]
    ds = w.bctx.manifest.dataset(w.frozen[0].dataset_id)
    cfg = VariantConfig(**w.frozen[0].variant_params)
    with pytest.raises(BuiltinConfigError):
        score_frozen(None, ds, recipe, cfg, {"method_id": "ridge", "feature_names": ["other"]}, [1])


def test_fixture_provenance_preserves_original_source_card(tmp_path):
    w=build_world(tmp_path)
    frozen=w.frozen[0]
    card=w.bctx.manifest.dataset(frozen.dataset_id).source_card
    assert w.bridge.integrity_source_card(w.bctx,frozen).digest()==card.digest()
    assert w.bridge.integrity_source_card(w.bctx,frozen).dataset_id==card.dataset_id
