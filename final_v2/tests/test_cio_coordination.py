"""CIO coordination / covariance / hedge reconciliation (spec 15.6, 16.1-16.4, 17, 18).

Fixtures are synthetic and deterministic. They exercise mechanics and fail-closed
behaviour only; nothing here shows investment or model skill.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal

import pytest

from pod_platform.allocation.construction import InternalPolicy, OfficialLimits
from pod_platform.allocation.coordination import (
    CIOCoordinator,
    CoordinationInputs,
    CoordinationPolicy,
    FillFeedback,
    HedgeInstrument,
    HedgeRequest,
    Holding,
    PendingOrder,
    PodAuthorization,
    PodForecast,
    PodIntention,
    PortfolioSnapshot,
    SecurityMeta,
    correlated_pod_budgets,
    group_forecasts,
    measure_pod_overlap,
    overlap_diagnostics,
    reconcile_hedges,
)
from pod_platform.allocation import covariance as covariance_module
from pod_platform.allocation.covariance import (
    CovarianceError,
    CovarianceEstimator,
    CovarianceModel,
    CovariancePolicy,
    ReturnRow,
)
from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.contracts.portfolio import RiskDecision, SolverStatus

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 3, 1, 21, 0, tzinfo=UTC)
IDS = [f"S{i:02d}" for i in range(1, 21)]
BETAS = {s: (1.3 if int(s[1:]) <= 4 else 0.7 if 11 <= int(s[1:]) <= 14 else 1.0) for s in IDS}
UNIVERSE = tuple(SecurityMeta(s, 1e10, "A" if int(s[1:]) <= 10 else "B", BETAS[s]) for s in IDS)
HEDGE = HedgeInstrument("HEDGE", beta=1.0, sector="IDX", market_cap_usd=1e11, adv_usd=1e12, max_abs_weight=0.5)
LONGS = {"S01": 0.020, "S02": 0.019, "S03": 0.018, "S04": 0.017}
SHORTS = {"S11": -0.020, "S12": -0.019, "S13": -0.018, "S14": -0.017}
LONGS3 = {"S05": 0.020, "S06": 0.019, "S07": 0.018, "S08": 0.017}
SHORTS3 = {"S15": -0.020, "S16": -0.019, "S17": -0.018, "S18": -0.017}


def _history(ids, n=40):
    x, rows = 12345, []
    beta = dict(BETAS, HEDGE=1.0)
    for k in range(n):
        x = (x * 1103515245 + 12345) % 2**31
        market = (x / 2**31 - 0.5) * 3.46
        vals = {}
        for s in ids:
            x = (x * 1103515245 + 12345) % 2**31
            vals[s] = 0.01 * (0.7 * beta[s] * market + (x / 2**31 - 0.5) * 3.46)
        rows.append(ReturnRow(T0 - dt.timedelta(days=n - k), vals))
    return tuple(rows)


HISTORY = _history(IDS + ["HEDGE"])


def fc(fid, pod, sec, alpha, **kw):
    kw.setdefault("as_of", T0 - dt.timedelta(hours=2))
    kw.setdefault("expires_at", T0 + dt.timedelta(days=1))
    return PodForecast(fid, pod, sec, alpha, kw.pop("unc", 0.01), publication_id=f"pub_{pod}", **kw)


def intention(pod, longs=LONGS, shorts=SHORTS, requests=(), prefix=None, **kw):
    prefix = prefix or pod
    fs = tuple(fc(f"{prefix}_{s}", pod, s, a, **kw) for s, a in {**longs, **shorts}.items())
    return PodIntention(pod, fs, hedge_requests=tuple(requests))


def hr(rid, who, dim="beta", cap=0.5, sec="HEDGE"):
    return HedgeRequest(rid, who, sec, dim, cap, T0 - dt.timedelta(hours=1), T0 + dt.timedelta(days=1))


def snapshot(holdings=(), pending=(), age_min=5):
    return PortfolioSnapshot("snap1", T0 - dt.timedelta(minutes=age_min), "nav1", "px1",
                             tuple(Holding(*h) for h in holdings), tuple(PendingOrder(*p) for p in pending))


def make_inputs(intentions, *, snap=None, cio=(), run_id="run1", history=HISTORY, universe=UNIVERSE, hedges=(HEDGE,)):
    pods = sorted({i.pod_id for i in intentions})
    auth = tuple(PodAuthorization(p, f"ns_{p}", T0 - dt.timedelta(days=30)) for p in pods)
    return CoordinationInputs(run_id, T0, auth, tuple(intentions), universe, tuple(hedges),
                              snap or snapshot(), cio_hedge_requests=tuple(cio), return_history=history)


OFFICIAL = OfficialLimits(min_positions=4, max_abs_name_weight=0.35)
INTERNAL = InternalPolicy(target_positions=8, sector_neutral=False, beta_neutral=False, retention_band=0.0)


def coordinator(**over):
    pol = dict(max_ex_ante_vol=100.0, max_abs_beta_net=0.5, max_sector_net=0.6, max_stress_loss=0.5)
    pol.update(over)
    return CIOCoordinator(OFFICIAL, INTERNAL, CoordinationPolicy(**pol), CovariancePolicy(min_observations=30))


def assert_fails(reason, call, code=None):
    with pytest.raises(PodPlatformError) as ei:
        call()
    assert ei.value.response.provenance["reason"] == reason
    if code is not None:
        assert ei.value.response.error.code == code


def weights(result):
    return {p.security_id: p.target_weight for p in result.target_book.positions}


# ------------------------------------------------------------ covariance (hand-calculated)

CP = CovariancePolicy(min_observations=4, variance_floor=1e-12)


def _rows(series, days=(4, 3, 2, 1)):
    return [ReturnRow(T0 - dt.timedelta(days=d), {k: v[i] for k, v in series.items()}) for i, d in enumerate(days)]


def test_hand_calculated_shrunk_covariance_and_netting():
    a = [0.01, -0.01, 0.01, -0.01]
    c = [-x for x in a]
    cov = CovarianceEstimator(CP).estimate(_rows({"A": a, "B": a, "C": c}), ["A", "B", "C"], as_of=T0)
    v = 4e-4 / 3  # sample variance, T-1 denominator
    delta = max(0.2, 3 / (4 + 3))  # N/(T+N) dominates min_shrinkage
    assert cov.shrinkage == pytest.approx(delta)
    assert cov.covariance("A", "A") == pytest.approx(v)
    assert cov.covariance("A", "B") == pytest.approx((1 - delta) * v)
    assert cov.covariance("A", "C") == pytest.approx(-(1 - delta) * v)
    assert cov.correlation("A", "B") == pytest.approx(1 - delta)
    # long A / short B nets most (not all) risk; long A / long B adds; long A / long C hedges
    assert cov.variance({"A": 1, "B": -1}) == pytest.approx(2 * v * delta)
    assert cov.variance({"A": 1, "B": 1}) == pytest.approx(2 * v * (2 - delta))
    assert cov.variance({"A": 1, "C": 1}) == pytest.approx(2 * v * delta)
    assert cov.volatility({"A": 1}) == pytest.approx(math.sqrt(v * 252))


def test_tiny_history_is_penalised_by_shrinkage_and_stays_positive_definite():
    a = [0.01, -0.01, 0.01, -0.01]
    ids = ["A", "B", "C", "D", "E"]
    cov = CovarianceEstimator(CP).estimate(_rows({k: a for k in ids}), ids, as_of=T0)
    assert cov.shrinkage == pytest.approx(5 / 9)  # more assets than comfortable history -> heavy shrinkage
    assert cov.correlation("A", "E") == pytest.approx(4 / 9)
    assert cov.pd_checked


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda r: r + [ReturnRow(T0 + dt.timedelta(hours=1), {"A": 0.0, "B": 0.0})], "FUTURE_ROW"),
        (lambda r: r[:3], "TOO_SHORT"),
        (lambda r: [ReturnRow(x.observed_at, {**x.returns, "A": float("nan")}) for x in r], "NONFINITE"),
        (lambda r: [ReturnRow(x.observed_at - dt.timedelta(days=30), x.returns) for x in r], "STALE"),
        (lambda r: r + [r[0]], "DUPLICATE_ROW"),
        (lambda r: [ReturnRow(x.observed_at, {"A": x.returns["A"]}) for x in r], "UNKNOWN_EXPOSURE"),
    ],
)
def test_estimator_rejects_bad_history(mutate, code):
    rows = _rows({"A": [0.01, -0.02, 0.015, -0.01], "B": [0.0, 0.01, -0.01, 0.02]})
    with pytest.raises(CovarianceError) as ei:
        CovarianceEstimator(CP).estimate(mutate(rows), ["A", "B"], as_of=T0)
    assert ei.value.code == code


def test_unknown_requested_exposure_and_non_psd_matrices_rejected():
    rows = _rows({"A": [0.01, -0.02, 0.015, -0.01]})
    with pytest.raises(CovarianceError) as ei:
        CovarianceEstimator(CP).estimate(rows, ["A", "ZZ"], as_of=T0)
    assert ei.value.code == "UNKNOWN_EXPOSURE"
    for m in ([[1.0, 1.2], [1.2, 1.0]], [[1.0, 1.0], [1.0, 1.0]]):  # indefinite; singular
        with pytest.raises(CovarianceError) as ei:
            CovarianceModel.from_matrix(["A", "B"], m, as_of=T0)
        assert ei.value.code == "NOT_PSD"
    ok = CovarianceModel.from_matrix(["A", "B"], [[1e-4, 0.0], [0.0, 1e-4]], as_of=T0)
    with pytest.raises(CovarianceError):
        ok.variance({"A": 1.0, "nope": 1.0})


# ------------------------------------------------------------ overlap & budgets (hand-calculated)


def test_pod_overlap_same_opposite_and_shared_beta_hand_calculated():
    meta = {k: SecurityMeta(k, 1e10, "s", 1.0) for k in "XYZ"}
    out = measure_pod_overlap({"a": {"X": 0.1, "Y": 0.1}, "b": {"X": -0.1, "Z": 0.1}}, meta)[0]
    assert out.opposite_sign_overlap == pytest.approx(0.5) and out.same_sign_overlap == 0.0
    assert out.name_jaccard == pytest.approx(1 / 3)
    same = measure_pod_overlap({"a": {"X": 0.1, "Y": 0.1}, "b": {"X": 0.05, "Z": 0.15}}, meta)[0]
    assert same.same_sign_overlap == pytest.approx(0.25)
    assert same.shared_beta == pytest.approx(0.2)  # both pods net-long beta; min(0.2, 0.2)


def test_risk_budgets_reflect_correlation_not_equal_votes():
    cov = CovarianceModel.from_matrix(["X", "Y"], [[1e-4, 0.0], [0.0, 1e-4]], as_of=T0)
    rep = correlated_pod_budgets({"a": {"X": 0.1}, "b": {"X": 0.1}, "c": {"Y": 0.1}}, cov, 0.12)
    by = {b.pod_id: b for b in rep.budgets}
    assert (by["a"].cluster_size, by["c"].cluster_size) == (pytest.approx(2.0), pytest.approx(1.0))
    assert (by["a"].share, by["b"].share, by["c"].share) == pytest.approx((0.25, 0.25, 0.5))
    assert by["c"].risk_budget == pytest.approx(0.06)
    assert rep.effective_independent_pods == pytest.approx(2.0)  # three pods, two independent bets
    assert by["a"].standalone_vol == pytest.approx(math.sqrt(0.01 * 1e-4 * 252))


# ------------------------------------------------------------ economic de-duplication


def test_grouping_by_label_source_event_and_statistics_but_not_by_chance():
    def f(fid, pod, vals, **kw):
        return [fc(f"{fid}{i}", pod, f"S{i + 1:02d}", v, **kw) for i, v in enumerate(vals)]

    a = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    b = [0.03, 0.06, 0.01, 0.05, 0.02, 0.04]  # Pearson with a is about -0.086 (hand computed)
    pol = CoordinationPolicy()
    assert len(group_forecasts(f("p1", "p1", a) + f("p2", "p2", b), pol)) == 2
    g = group_forecasts(f("p1", "p1", a) + f("p2", "p2", [2 * x + 0.001 for x in a]), pol)
    assert len(g) == 1 and any(x.startswith("alpha_correlation") for x in g[0].basis)  # no shared ids needed
    for kw, tag in (({"event_ids": ("ev1",)}, "shared_event"), ({"measurement_ids": ("m1",)}, "shared_measurement")):
        g = group_forecasts(f("p1", "p1", a, **kw) + f("p2", "p2", b, **kw), pol)
        assert len(g) == 1 and any(x.startswith(tag) for x in g[0].basis)
    # same broad source / declared exposure with distinct mechanisms stays distinct, reported as a diagnostic
    live = f("p1", "p1", a, source_ids=("panel",), exposure_ids=("beta",)) + f("p2", "p2", b, source_ids=("panel",), exposure_ids=("beta",))
    g = group_forecasts(live, pol)
    assert len(g) == 2
    diag = overlap_diagnostics(live, g)
    assert len(diag) == 2 and diag[0].startswith("shared_exposure:beta") and diag[1].startswith("shared_source:panel")
    g = group_forecasts(f("p1", "p1", a, family_label="mom") + f("p2", "p2", b, family_label="mom"), pol)
    assert len(g) == 1 and g[0].pods == ("p1", "p2")


# ------------------------------------------------------------ complete target construction


def test_complete_single_pod_target_hand_calculated_and_risk_passes():
    res = coordinator().coordinate(make_inputs([intention("p1")]))
    expect = {s: Decimal("0.125") for s in LONGS} | {s: Decimal("-0.125") for s in SHORTS}
    assert weights(res) == expect  # k=4 per side, 0.5 gross per side / 4
    assert res.vol_scale == 1.0 and res.target_book.solver_status == SolverStatus.FEASIBLE
    assert res.exposures["abs_beta_net"] == pytest.approx(0.5 * 1.3 - 0.5 * 0.7)
    assert res.risk_review.decision == RiskDecision.PASS and res.executable
    assert res.actual_review is None and res.post_fill_review is None  # nothing held yet
    assert res.target_book.covariance_version == res.accounting_refs["covariance_version"]
    assert res.target_book.forecast_publication_ids == ["pub_p1"]
    assert {s: round(v, 9) for s, v in res.trades.items()} == {s: float(w) for s, w in expect.items()}
    # Euler decomposition is exact: pod risk contributions sum to ex-ante vol, attribution sums to the book
    assert sum(res.risk_contributions.values()) == pytest.approx(res.ex_ante_vol, rel=1e-9)
    assert sum(res.attribution["p1"].values()) == pytest.approx(0.0, abs=1e-12)
    assert all(res.attribution["p1"][s] == pytest.approx(float(w)) for s, w in expect.items())


def test_covariance_changes_size_and_blocks_when_scale_is_too_small():
    v0 = coordinator().coordinate(make_inputs([intention("p1")])).ex_ante_vol
    half = coordinator(max_ex_ante_vol=v0 / 2).coordinate(make_inputs([intention("p1")]))
    assert half.vol_scale == pytest.approx(0.5, rel=1e-5) and half.ex_ante_vol == pytest.approx(v0 / 2, rel=1e-5)
    assert half.ex_ante_vol <= v0 / 2 + 1e-12  # the cap is never exceeded after truncation
    assert float(weights(half)["S01"]) == pytest.approx(0.0625, abs=1e-6)
    assert half.target_book.solver_status == SolverStatus.RELAXED and half.target_book.relaxations
    assert half.risk_review.decision == RiskDecision.PASS
    tiny = coordinator(max_ex_ante_vol=v0 / 10).coordinate(make_inputs([intention("p1")]))
    assert tiny.risk_review.decision == RiskDecision.BLOCK and not tiny.executable
    assert any("vol_scale_floor" in r for r in tiny.risk_review.remedy_constraints)


def test_identical_strategies_across_namespaces_are_one_economic_vote():
    single = coordinator().coordinate(make_inputs([intention("p1")]))
    twin = coordinator().coordinate(make_inputs([intention("p1"), intention("p2")]))
    assert weights(twin) == weights(single)  # not double alpha, not a second equal-family vote
    assert len(twin.groups) == 1 and twin.groups[0].pods == ("p1", "p2")
    assert twin.risk_budgets.effective_independent_pods == pytest.approx(1.0)
    assert [b.share for b in twin.risk_budgets.budgets] == pytest.approx([0.5, 0.5])
    ov = twin.overlap[0]
    assert ov.same_sign_overlap == pytest.approx(1.0) and ov.shared_group and ov.return_correlation == pytest.approx(1.0)
    assert twin.risk_contributions["p1"] == pytest.approx(twin.risk_contributions["p2"])
    assert len(twin.forecast_dispositions) == 16  # every competing original preserved
    third = coordinator().coordinate(make_inputs([intention("p1"), intention("p2"), intention("p3", LONGS3, SHORTS3)]))
    assert len(third.groups) == 2
    assert 1.0 < third.risk_budgets.effective_independent_pods < 3.0
    sh = {b.pod_id: b.share for b in third.risk_budgets.budgets}
    assert sh["p3"] > sh["p1"]  # the independent pod is not diluted by the duplicated pair


def test_replay_and_supersede_preserve_originals_without_double_counting():
    base = intention("p1")
    replay = PodIntention("p1", base.forecasts + base.forecasts[:2])  # identical resend of two forecasts
    res = coordinator().coordinate(make_inputs([replay]))
    assert sum(d.status == "replay_collapsed" for d in res.forecast_dispositions) == 2
    assert weights(res) == weights(coordinator().coordinate(make_inputs([base])))
    newer = fc("p1_S01_v2", "p1", "S01", 0.001, supersedes="p1_S01")
    res = coordinator().coordinate(make_inputs([PodIntention("p1", base.forecasts + (newer,))]))
    status = {d.forecast.forecast_id: d.status for d in res.forecast_dispositions}
    assert status["p1_S01"] == "superseded" and status["p1_S01_v2"] == "live"


# ------------------------------------------------------------ hedges


def test_three_simultaneous_hedge_requests_become_one_shared_hedge():
    reqs = [hr("h1", "p1"), hr("h2", "p2"), hr("h3", "CIO")]
    res = coordinator(pod_budget_tolerance=10.0).coordinate(make_inputs(
        [intention("p1", requests=[reqs[0]]), PodIntention("p2", (), hedge_requests=(reqs[1],))], cio=[reqs[2]]))
    w = weights(res)
    assert float(w["HEDGE"]) == pytest.approx(-0.30, abs=1e-6)  # one aggregate residual, not 3 x (-0.30)
    assert sum(1 for p in res.target_book.positions if p.security_id == "HEDGE") == 1
    assert [d.status for d in res.hedge_plan.dispositions] == ["satisfied_by_shared_hedge"] * 3
    assert res.hedge_plan.lines[0].requesters == ("CIO", "p1", "p2")
    assert res.exposures["abs_beta_net"] == pytest.approx(0.0, abs=1e-6)  # post-hedge residual
    assert res.attribution["CIO"]["HEDGE"] + res.attribution["p1"]["HEDGE"] + res.attribution["p2"]["HEDGE"] \
        == pytest.approx(-0.30, abs=1e-6)
    assert res.risk_review.decision == RiskDecision.PASS


def test_held_and_pending_hedge_quantity_counted_exactly_once_and_fill_feedback():
    snap = snapshot([("HEDGE", -0.10)], [("o1", "HEDGE", -0.05)])
    inp = make_inputs([intention("p1", requests=[hr("h1", "p1")])], snap=snap)
    res = coordinator().coordinate(inp)
    line = res.hedge_plan.lines[0]
    assert line.held_weight == -0.10 and line.pending_weight == -0.05
    assert line.new_order_weight == pytest.approx(-0.30 - (-0.15), abs=1e-6)
    assert res.trades["HEDGE"] == pytest.approx(-0.15, abs=1e-6)
    assert res.actual_review is not None and res.post_fill_review is not None
    # partial fill: the old plan is stale, the recomputed one does not double count
    fills = [FillFeedback("evt1", "o1", -0.02)]
    assert res.is_current_for(inp.snapshot, inp.all_hedge_requests())
    after = coordinator().recompute_after_fills(inp, fills, as_of=T0)
    new_snap = inp.snapshot.apply_fills(fills, as_of=T0)
    assert not res.is_current_for(new_snap, inp.all_hedge_requests())
    assert after.is_current_for(new_snap, inp.all_hedge_requests())
    l2 = after.hedge_plan.lines[0]
    assert (l2.held_weight, l2.pending_weight) == (pytest.approx(-0.12), pytest.approx(-0.03))
    assert l2.new_order_weight == pytest.approx(-0.15, abs=1e-6) and after.accounting_refs["applied_fill_events"] == ("evt1",)
    # replaying the same fill event is idempotent; conflicting replay or overfill fails closed
    once = inp.snapshot.apply_fills(fills, as_of=T0)
    assert once.apply_fills(fills, as_of=T0) == once
    assert_fails("CONFLICTING_FILL_EVENT", lambda: once.apply_fills([FillFeedback("evt1", "o1", -0.01)], as_of=T0))
    assert_fails("BAD_FILL", lambda: inp.snapshot.apply_fills([FillFeedback("evt2", "o1", -0.5)], as_of=T0))
    assert_fails("UNKNOWN_ORDER", lambda: inp.snapshot.apply_fills([FillFeedback("evt3", "nope", -0.01)], as_of=T0))


META = {"L": SecurityMeta("L", 1e10, "A", 1.0), "S": SecurityMeta("S", 1e10, "A", 1.0), "HEDGE": HEDGE}


def _reconcile(base, reqs, inst=HEDGE, held=None, pending=None):
    return reconcile_hedges(
        base, reqs, [inst], held=held or {}, pending=pending or {}, meta={**META, inst.security_id: inst},
        official=OFFICIAL, internal=INTERNAL, policy=CoordinationPolicy(max_abs_beta_net=0.5, max_sector_net=0.6),
        decision_at=T0)


def test_offsetting_pod_betas_need_no_hedge_even_though_both_pods_ask():
    plan = _reconcile({"L": 0.3, "S": -0.3}, [hr("h1", "p1"), hr("h2", "p2")])
    assert plan.weights["HEDGE"] == 0.0 and plan.lines[0].new_order_weight == 0.0
    assert {d.status for d in plan.dispositions} == {"not_needed"}


def test_hedge_constraints_bind_and_are_recorded():
    base = {"L": 0.4, "S": -0.1}  # beta residual 0.30
    unlimited = _reconcile(base, [hr("h1", "p1"), hr("h2", "CIO")])
    assert unlimited.weights["HEDGE"] == pytest.approx(-0.30, abs=1e-8) and unlimited.lines[0].binding == ()
    thin = _reconcile(base, [hr("h1", "p1")], HedgeInstrument("HEDGE", 1.0, "IDX", 1e11, 1e8))
    assert thin.weights["HEDGE"] == pytest.approx(-0.05) and "adv" in thin.lines[0].binding  # 10% x 5d x 1e8 / 1e9
    capped = _reconcile(base, [hr("h1", "p1", cap=0.1)])
    assert capped.weights["HEDGE"] == pytest.approx(-0.1) and "request_cap" in capped.lines[0].binding
    noshort = _reconcile(base, [hr("h1", "p1")], HedgeInstrument("HEDGE", 1.0, "IDX", 1e11, 1e12, shortable=False))
    assert noshort.weights["HEDGE"] == 0.0 and "not_shortable" in noshort.lines[0].binding
    assert {d.status for d in noshort.dispositions} == {"constrained"}
    small = _reconcile(base, [hr("h1", "p1")], HedgeInstrument("HEDGE", 1.0, "IDX", 1e8, 1e12))
    assert small.weights["HEDGE"] == 0.0 and "short_market_cap" in small.lines[0].binding


def test_hedge_requests_fail_closed_and_dimension_without_loading_is_rejected():
    assert_fails("HEDGE_NOT_ALLOWED", lambda: _reconcile({"L": 0.4}, [hr("h1", "p1", sec="SPY")]),
                 ErrorCode.ADMISSION_CERTIFICATE_INVALID)
    assert_fails("DUPLICATE_INTENT", lambda: _reconcile({"L": 0.4}, [hr("h1", "p1"), hr("h2", "p1")]))
    assert_fails("DUPLICATE_INTENT", lambda: _reconcile({"L": 0.4}, [hr("h1", "p1"), hr("h1", "p1", cap=0.2)]))
    assert_fails("EXPIRED_HEDGE_REQUEST", lambda: _reconcile({"L": 0.4}, [HedgeRequest(
        "h1", "p1", "HEDGE", "beta", 0.5, T0 - dt.timedelta(days=2), T0 - dt.timedelta(days=1))]))
    plan = _reconcile({"L": 0.4}, [hr("h1", "p1"), hr("h1", "p1")])  # identical replay counts once
    assert [d.status for d in plan.dispositions].count("replay_collapsed") == 1
    nolo = _reconcile({"L": 0.4}, [hr("h1", "p1", dim="sector:OTHER")])
    assert nolo.dispositions[0].status == "rejected_no_exposure_loading" and not nolo.lines


# ------------------------------------------------------------ actual / post-fill risk


def test_actual_and_post_fill_books_are_reviewed_against_the_same_covariance_limits():
    snap = snapshot([("S01", 0.30)], [("o1", "S01", 0.10)])
    res = coordinator().coordinate(make_inputs([intention("p1")], snap=snap))
    assert res.actual_review.reviewed_kind == "actual" and res.actual_review.decision == RiskDecision.PASS
    assert res.post_fill_review.decision == RiskDecision.BLOCK  # 0.30 + 0.10 pending breaches the 0.35 name cap
    assert float(res.post_fill_review.measured_exposures["max_abs_name"]) == pytest.approx(0.40)
    assert res.trades["S01"] == pytest.approx(0.125 - 0.40)  # target - (held + pending)
    tight = coordinator(max_ex_ante_vol=1e-4).coordinate(make_inputs([intention("p1")], snap=snap))
    assert tight.actual_review.decision == RiskDecision.BLOCK  # covariance vol limit applies to actual holdings too


def test_idempotent_replay_is_byte_identical():
    inp = make_inputs([intention("p1", requests=[hr("h1", "p1")]), intention("p3", LONGS3, SHORTS3)], snap=snapshot([("S01", 0.1)]))
    a, b = coordinator().coordinate(inp), coordinator().coordinate(inp)
    assert a.input_digest == b.input_digest and a.state_digest == b.state_digest
    assert a.target_book.digest() == b.target_book.digest()
    assert a.risk_review.digest() == b.risk_review.digest()
    assert a.attribution == b.attribution and a.trades == b.trades


# ------------------------------------------------------------ fail-closed inputs


def test_stale_future_snapshot_and_forecast_inputs_fail_closed():
    c, base = coordinator(), [intention("p1")]
    assert_fails("STALE_SNAPSHOT", lambda: c.coordinate(make_inputs(base, snap=snapshot(age_min=120))), ErrorCode.ASOF_VIOLATION)
    assert_fails("FUTURE_SNAPSHOT", lambda: c.coordinate(make_inputs(base, snap=snapshot(age_min=-5))))
    assert_fails("FUTURE_FORECAST", lambda: c.coordinate(make_inputs([intention("p1", as_of=T0 + dt.timedelta(hours=1))])))
    assert_fails("STALE_FORECAST", lambda: c.coordinate(make_inputs([intention("p1", as_of=T0 - dt.timedelta(days=5))])),
                 ErrorCode.FORECAST_EXPIRED)
    assert_fails("STALE_FORECAST", lambda: c.coordinate(make_inputs([intention("p1", expires_at=T0)])))


def test_unknown_identifiers_unauthorized_pods_and_hedge_universe_fail_closed():
    c = coordinator()
    assert_fails("UNKNOWN_SECURITY", lambda: c.coordinate(make_inputs([intention("p1", {**LONGS, "ZZZ": 0.01})])),
                 ErrorCode.IDENTITY_AMBIGUOUS)
    assert_fails("UNKNOWN_SECURITY", lambda: c.coordinate(make_inputs([intention("p1")], snap=snapshot([("GHOST", 0.1)]))))
    assert_fails("UNKNOWN_SECURITY", lambda: c.coordinate(make_inputs([intention("p1")], snap=snapshot(pending=[("o", "GHOST", 0.1)]))))
    assert_fails("HEDGE_NOT_ALLOWED", lambda: c.coordinate(make_inputs([intention("p1", requests=[hr("h", "p1", sec="S05")])])))
    assert_fails("HEDGE_OVERLAPS_ALPHA_UNIVERSE", lambda: c.coordinate(make_inputs(
        [intention("p1")], hedges=(HedgeInstrument("S05", 1.0, "A", 1e11, 1e12),))))
    assert_fails("HEDGE_REQUESTER_MISMATCH", lambda: c.coordinate(make_inputs([intention("p1", requests=[hr("h", "p2")])])))
    inp = make_inputs([intention("p1")])
    bad = CoordinationInputs(**{**inp.__dict__, "authorizations": ()})
    assert_fails("UNAUTHORIZED_POD", lambda: c.coordinate(bad), ErrorCode.ADMISSION_CERTIFICATE_INVALID)
    expired = PodAuthorization("p1", "ns", T0 - dt.timedelta(days=9), T0 - dt.timedelta(days=1))
    assert_fails("UNAUTHORIZED_POD", lambda: c.coordinate(CoordinationInputs(**{**inp.__dict__, "authorizations": (expired,)})))


def test_unexplained_duplicate_intents_fail_closed():
    c, base = coordinator(), intention("p1")
    second = fc("p1_S01_other", "p1", "S01", 0.5)
    assert_fails("DUPLICATE_INTENT", lambda: c.coordinate(make_inputs([PodIntention("p1", base.forecasts + (second,))])))
    clash = fc("p1_S01", "p1", "S01", 0.5)  # same id, different content
    assert_fails("DUPLICATE_INTENT", lambda: c.coordinate(make_inputs([PodIntention("p1", base.forecasts + (clash,))])))
    stray = fc("x", "p1", "S02", 0.1, supersedes="p1_S01")  # supersedes a different security
    assert_fails("BAD_SUPERSEDES", lambda: c.coordinate(make_inputs([PodIntention("p1", base.forecasts + (stray,))])))


def test_covariance_failures_surface_as_typed_blocking_errors():
    c, base = coordinator(), [intention("p1")]
    future = HISTORY + (ReturnRow(T0 + dt.timedelta(hours=1), HISTORY[0].returns),)
    assert_fails("COVARIANCE_FUTURE_ROW", lambda: c.coordinate(make_inputs(base, history=future)), ErrorCode.ASOF_VIOLATION)
    assert_fails("COVARIANCE_TOO_SHORT", lambda: c.coordinate(make_inputs(base, history=HISTORY[-10:])),
                 ErrorCode.INSUFFICIENT_HISTORY)
    no_hedge = tuple(ReturnRow(r.observed_at, {k: v for k, v in r.returns.items() if k != "HEDGE"}) for r in HISTORY)
    assert_fails("COVARIANCE_UNKNOWN_EXPOSURE", lambda: c.coordinate(make_inputs(base, history=no_hedge)),
                 ErrorCode.IDENTITY_AMBIGUOUS)
    assert_fails("COVARIANCE_SOURCE", lambda: c.coordinate(make_inputs(base, history=())))
    partial = CovarianceModel.from_matrix(["S01"], [[1e-4]], as_of=T0)
    inp = make_inputs(base, history=())
    assert_fails("COVARIANCE_UNKNOWN_EXPOSURE", lambda: c.coordinate(CoordinationInputs(**{**inp.__dict__, "covariance": partial})))


# ------------------------------------------------------------ PSD validation is never skipped


def _adversarial(n=500):
    """Positive diagonal, symmetric, NOT diagonally dominant, one indefinite 2x2 block (eigenvalue -0.2)."""
    m = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    m[0][1] = m[1][0] = 1.2
    return [f"A{i:03d}" for i in range(n)], m


def test_500_asset_non_psd_matrix_is_rejected_with_and_without_numpy(monkeypatch):
    ids, m = _adversarial()
    with pytest.raises(CovarianceError) as ei:  # whatever the environment has, it must not pass
        CovarianceModel.from_matrix(ids, m, as_of=T0)
    assert ei.value.code in ("NOT_PSD", "PD_UNVERIFIABLE")
    monkeypatch.setattr(covariance_module, "_load_numpy", lambda: None)
    with pytest.raises(CovarianceError) as ei:  # no NumPy and no sufficient certificate -> fail closed
        CovarianceModel.from_matrix(ids, m, as_of=T0)
    assert ei.value.code == "PD_UNVERIFIABLE"
    ok = CovarianceModel.from_matrix(ids, [[2.0 if i == j else 0.001 for j in range(500)] for i in range(500)], as_of=T0)
    assert ok.pd_checked and ok.pd_method == "gershgorin_diagonal_dominance"


def test_large_estimate_is_validated_not_skipped(monkeypatch):
    ids = [f"E{i}" for i in range(6)]
    rows = _rows({s: [0.01 * ((k * (i + 2)) % 5 - 2) for k in range(8)] for i, s in enumerate(ids)}, days=tuple(range(8, 0, -1)))
    small = CovariancePolicy(min_observations=4, variance_floor=1e-12, pd_check_max_assets=3)
    monkeypatch.setattr(covariance_module, "_load_numpy", lambda: None)
    model = CovarianceEstimator(small).estimate(rows, ids, as_of=T0)
    assert model.pd_checked and model.pd_method in ("gershgorin_diagonal_dominance", "shrinkage_construction")
    assert CovarianceEstimator(CP).estimate(rows, ids, as_of=T0).pd_method == "cholesky_python"
    unproven = CovarianceModel(model.security_ids, model.matrix, T0, 8, 0.2, 252.0, "v", pd_checked=False)
    inp = make_inputs([intention("p1")], history=())
    assert_fails("COVARIANCE_PD_UNVERIFIABLE", lambda: coordinator().coordinate(CoordinationInputs(**{**inp.__dict__, "covariance": unproven})))


def test_shared_broad_source_with_distinct_mechanisms_is_not_duplicate_economics():
    p1 = intention("p1", source_ids=("panel",))
    p2 = intention("p2", longs=LONGS3, shorts=SHORTS3, source_ids=("panel",))
    res = coordinator().coordinate(make_inputs([p1, p2]))
    assert len(res.groups) == 2 and all(g.basis == ("distinct_family",) for g in res.groups)
    assert len(res.overlap_diagnostics) == 1 and "shared_source:panel" in res.overlap_diagnostics[0]
    assert res.covariance is not None and res.covariance.pd_checked
    # identical economics under different labels and namespaces still collapse to one family
    d1, d2 = intention("p1", family_label="momA"), intention("p2", family_label="momB", prefix="q")
    assert len(coordinator().coordinate(make_inputs([d1, d2])).groups) == 1


def test_old_information_is_not_rejuvenated_by_new_publication():
    from dataclasses import replace
    from pod_platform.contracts.errors import PodPlatformError
    inputs=make_inputs((intention("pod_a"),))
    pod=inputs.intentions[0]
    forecasts=tuple(replace(f,as_of=T0,information_cutoff=T0-dt.timedelta(days=10)) for f in pod.forecasts)
    changed=replace(inputs,intentions=(replace(pod,forecasts=forecasts),)+inputs.intentions[1:])
    with pytest.raises(PodPlatformError,match='STALE_FORECAST'):
        coordinator().coordinate(changed)
