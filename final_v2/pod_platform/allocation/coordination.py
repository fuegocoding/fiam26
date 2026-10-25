"""CIO portfolio coordination and risk layer (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 15.6, 16.1-16.4, 17, 18).

``CIOCoordinator.coordinate`` turns dated, authorized pod forecasts plus actual
holdings and pending orders into ONE aggregate target book, using the existing
``FamilyIntegrator``, ``ConstructionEngine`` and ``RiskEngine``. All financial
authority is deterministic code under an explicit fixed host policy; nothing
is fitted to seen performance and no model output sets a size.

Key rules implemented here:
- economic de-duplication across pod namespaces BEFORE equal-family
  integration (shared family label / event / measurement ids, plus alpha-vector
  correlation, so duplicate economics are caught even without shared ids).
  A shared source id alone is only a diagnostic: strategies legitimately share
  one broad source (e.g. a characteristics panel) with distinct mechanisms;
- the regularized as-of covariance scales the book to the vol budget, and the
  same covariance drives risk review, risk budgets, stress and attribution;
- hedge requests are evidence of need, reconciled once against the residual
  aggregate exposure over an explicit allowed hedge universe; held and
  pending hedge quantity is counted exactly once;
- stale snapshots, unauthorized pods, unknown identifiers and unexplained
  duplicate intents raise ``PodPlatformError`` with ``provenance['reason']``;
- every original forecast and hedge request is preserved in the result.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import ROUND_DOWN, Decimal
from typing import Mapping, NoReturn, Optional, Sequence

from pydantic import Field

from ..contracts.base import StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from ..contracts.portfolio import (
    ConstraintResult,
    Relaxation,
    RiskDecision,
    RiskReview,
    SolverStatus,
    StressResult,
    TargetBook,
)
from ..risk.envelope import MeasuredExposure, RiskEngine, RiskLimit, RiskPolicy
from .construction import Candidate, ConstructionEngine, InternalPolicy, OfficialLimits
from .covariance import CovarianceError, CovarianceEstimator, CovarianceModel, CovariancePolicy, ReturnRow
from .integration import FamilyEstimate, FamilyIntegrator

ATTRIBUTION_METHOD = "euler_risk_contribution_group_alpha_share_v1"
CIO = "CIO"
_EPS = 1e-12


def _fail(code: ErrorCode, reason: str, summary: str) -> NoReturn:
    raise PodPlatformError(
        ToolResponse.failure(
            summary=f"{reason}: {summary}",
            code=code,
            stop_condition="CIO coordination fails closed; correct the input and re-run.",
            provenance={"reason": reason},
        )
    )


def _finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(ErrorCode.SCHEMA_MISMATCH, "NONFINITE", f"{name} must be finite")
    return float(value)


def _enc(o):
    if isinstance(o, dt.datetime):
        return o.astimezone(dt.timezone.utc).isoformat()
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: getattr(o, f.name) for f in fields(o)}
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"cannot digest {type(o)!r}")


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=_enc, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------ policy & inputs


class CoordinationPolicy(StrictModel):
    """Explicit fixed host policy. Defaults are not tuned to any realised result."""

    policy_version: str = Field(min_length=1, default="cio_coord_v1")
    max_forecast_age_hours: int = Field(ge=1, default=48)
    max_snapshot_age_minutes: int = Field(ge=0, default=60)
    duplicate_corr_threshold: float = Field(gt=0.0, le=1.0, default=0.9)
    duplicate_min_common: int = Field(ge=2, default=5)
    derived_pod_gross: float = Field(gt=0.0, default=1.0)
    max_ex_ante_vol: float = Field(gt=0.0, default=0.10)
    min_vol_scale: float = Field(gt=0.0, le=1.0, default=0.25)
    max_abs_beta_net: float = Field(ge=0.0, default=0.25)
    max_sector_net: float = Field(ge=0.0, default=0.10)
    hedge_residual_tolerance: float = Field(ge=0.0, default=0.01)
    hedge_adv_participation: float = Field(gt=0.0, le=1.0, default=0.10)
    pod_budget_tolerance: float = Field(ge=1.0, default=1.5)
    max_stress_pod_share: float = Field(gt=0.0, default=0.8)
    stress_corr: float = Field(ge=0.0, lt=1.0, default=0.8)
    stress_z: float = Field(gt=0.0, default=2.326)
    stress_horizon_periods: int = Field(ge=1, default=10)
    market_shock: float = Field(gt=0.0, default=0.10)
    sector_shock: float = Field(gt=0.0, default=0.15)
    max_stress_loss: float = Field(gt=0.0, default=0.15)
    review_ttl_minutes: int = Field(ge=1, default=60)


@dataclass(frozen=True)
class SecurityMeta:
    security_id: str
    market_cap_usd: float
    sector: str
    beta: float = 1.0
    price_usd: float = 10.0
    adv_usd: float = 1_000_000_000.0
    short_interest_pct: float = float("nan")
    tradable: bool = True
    shortable: bool = True


@dataclass(frozen=True)
class HedgeInstrument:
    """An explicitly allowed hedge security. Nothing is hedge-eligible by default."""

    security_id: str
    beta: float
    sector: str
    market_cap_usd: float
    adv_usd: float
    shortable: bool = True
    max_abs_weight: float = 1.0


@dataclass(frozen=True)
class PodAuthorization:
    pod_id: str
    namespace: str
    authorized_from: dt.datetime
    authorized_until: Optional[dt.datetime] = None


@dataclass(frozen=True)
class PodForecast:
    forecast_id: str
    pod_id: str
    security_id: str
    expected_excess_return: float
    uncertainty: float
    as_of: dt.datetime
    expires_at: dt.datetime
    publication_id: str
    family_label: Optional[str] = None
    source_ids: tuple[str, ...] = ()  # evidence lineage only: NEVER merges forecasts (diagnostic)
    event_ids: tuple[str, ...] = ()  # same economic event: merges
    supersedes: Optional[str] = None
    measurement_ids: tuple[str, ...] = ()  # same economic measurement of the same mechanism: merges
    exposure_ids: tuple[str, ...] = ()  # declared correlated exposure: diagnostic only (covariance handles it)
    information_cutoff: Optional[dt.datetime] = None  # evidence age is separate from publication/authorization time


@dataclass(frozen=True)
class HedgeRequest:
    """Evidence of a hedge need, not an order. ``dimension``: 'beta', 'net' or 'sector:<name>'."""

    request_id: str
    requester: str
    security_id: str
    dimension: str
    max_abs_weight: float
    as_of: dt.datetime
    expires_at: dt.datetime
    reason: str = ""


@dataclass(frozen=True)
class PodIntention:
    pod_id: str
    forecasts: tuple[PodForecast, ...]
    intended_weights: Mapping[str, float] = field(default_factory=dict)
    hedge_requests: tuple[HedgeRequest, ...] = ()


@dataclass(frozen=True)
class Holding:
    security_id: str
    weight: float


@dataclass(frozen=True)
class PendingOrder:
    order_id: str
    security_id: str
    remaining_weight: float  # signed NAV fraction still unfilled


@dataclass(frozen=True)
class FillFeedback:
    accounting_event_id: str
    order_id: str
    filled_weight: float  # signed, same sign as the order's remaining weight


@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_id: str
    as_of: dt.datetime
    nav_snapshot_id: str
    price_snapshot_id: str
    holdings: tuple[Holding, ...]
    pending_orders: tuple[PendingOrder, ...] = ()
    applied_fill_events: tuple[tuple[str, str], ...] = ()

    def apply_fills(self, fills: Sequence[FillFeedback], *, as_of: dt.datetime) -> "PortfolioSnapshot":
        """Idempotent by accounting_event_id; over-fills / unknown orders fail closed."""
        as_of = require_aware(as_of, "as_of")
        if as_of < self.as_of:
            _fail(ErrorCode.ASOF_VIOLATION, "FILL_BEFORE_SNAPSHOT", "fill feedback predates the snapshot")
        held = {h.security_id: h.weight for h in self.holdings}
        pend = {o.order_id: o for o in self.pending_orders}
        applied = dict(self.applied_fill_events)
        changed = False
        for fill in fills:
            qty = _finite(fill.filled_weight, "filled_weight")
            key = f"{fill.order_id}|{qty!r}"
            if fill.accounting_event_id in applied:
                if applied[fill.accounting_event_id] != key:
                    _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "CONFLICTING_FILL_EVENT",
                          f"event {fill.accounting_event_id!r} replayed with different content")
                continue
            order = pend.get(fill.order_id)
            if order is None:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "UNKNOWN_ORDER", f"no pending order {fill.order_id!r}")
            if qty * order.remaining_weight <= 0 or abs(qty) > abs(order.remaining_weight) + _EPS:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "BAD_FILL", f"fill {qty} vs remaining {order.remaining_weight}")
            held[order.security_id] = held.get(order.security_id, 0.0) + qty
            rest = order.remaining_weight - qty
            if abs(rest) <= _EPS:
                del pend[fill.order_id]
            else:
                pend[fill.order_id] = replace(order, remaining_weight=rest)
            applied[fill.accounting_event_id] = key
            changed = True
        if not changed:
            return self
        sid =f"{self.snapshot_id}+fills:{_digest(sorted(applied.items()))[:8]}"
        return PortfolioSnapshot(
            sid, as_of, self.nav_snapshot_id, self.price_snapshot_id,
            tuple(Holding(s, w) for s, w in sorted(held.items())),
            tuple(sorted(pend.values(), key=lambda o: o.order_id)),
            tuple(sorted(applied.items())),
        )


@dataclass(frozen=True)
class CoordinationInputs:
    run_id: str
    decision_at: dt.datetime
    authorizations: tuple[PodAuthorization, ...]
    intentions: tuple[PodIntention, ...]
    universe: tuple[SecurityMeta, ...]
    hedge_universe: tuple[HedgeInstrument, ...]
    snapshot: PortfolioSnapshot
    cost_model_version: str = "cost_model_unspecified"
    cio_hedge_requests: tuple[HedgeRequest, ...] = ()
    return_history: tuple[ReturnRow, ...] = ()
    covariance: Optional[CovarianceModel] = None

    def digest(self) -> str:
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "covariance"}
        payload["covariance_version"] = self.covariance.version if self.covariance else None
        return _digest(payload)

    def all_hedge_requests(self) -> tuple[HedgeRequest, ...]:
        return tuple(r for it in self.intentions for r in it.hedge_requests) + tuple(self.cio_hedge_requests)


def state_digest(snapshot: PortfolioSnapshot, requests: Sequence[HedgeRequest]) -> str:
    return _digest({"snapshot": snapshot, "requests": sorted(requests, key=lambda r: r.request_id)})


# ------------------------------------------------------------------ exposure helpers


def _gross(w: Mapping[str, float]) -> float:
    return sum(abs(v) for v in w.values())


def _beta_net(w, meta) -> float:
    return sum(v * meta[s].beta for s, v in w.items())


def _sector_net(w, meta) -> dict[str, float]:
    out: dict[str, float] = {}
    for s, v in w.items():
        out[meta[s].sector] = out.get(meta[s].sector, 0.0) + v
    return out


def _dim_exposure(dim: str, w, meta) -> float:
    if dim == "beta":
        return _beta_net(w, meta)
    if dim == "net":
        return sum(w.values())
    return sum(v for s, v in w.items() if meta[s].sector == dim.split(":", 1)[1])


def _dim_loading(dim: str, inst: HedgeInstrument) -> float:
    if dim == "beta":
        return inst.beta
    if dim == "net":
        return 1.0
    return 1.0 if inst.sector == dim.split(":", 1)[1] else 0.0


def _q(x: float) -> float:
    return float(Decimal(repr(x)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN))


# ------------------------------------------------------------------ overlap & budgets


@dataclass(frozen=True)
class PodOverlap:
    pod_a: str
    pod_b: str
    same_sign_overlap: float
    opposite_sign_overlap: float
    name_jaccard: float
    event_jaccard: float
    shared_group: bool
    shared_beta: float
    sector_cosine: float
    return_correlation: float


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def measure_pod_overlap(
    pod_weights: Mapping[str, Mapping[str, float]],
    meta: Mapping[str, object],
    *,
    pod_events: Optional[Mapping[str, set]] = None,
    pod_groups: Optional[Mapping[str, set]] = None,
    cov: Optional[CovarianceModel] = None,
) -> tuple[PodOverlap, ...]:
    """Pairwise holdings / event / factor / return overlap. Overlap fractions are over the smaller pod gross."""
    pods = sorted(pod_weights)
    out = []
    for i, a in enumerate(pods):
        for b in pods[i + 1:]:
            wa, wb = pod_weights[a], pod_weights[b]
            denom = min(_gross(wa), _gross(wb))
            same = opp = 0.0
            for s in set(wa) & set(wb):
                prod = wa[s] * wb[s]
                m = min(abs(wa[s]), abs(wb[s]))
                if prod > 0:
                    same += m
                elif prod < 0:
                    opp += m
            na = {s for s, v in wa.items() if v != 0}
            nb = {s for s, v in wb.items() if v != 0}
            ba, bb = _beta_net(wa, meta), _beta_net(wb, meta)
            sa, sb = _sector_net(wa, meta), _sector_net(wb, meta)
            dot = sum(sa[k] * sb.get(k, 0.0) for k in sa)
            norm = math.sqrt(sum(v * v for v in sa.values()) * sum(v * v for v in sb.values()))
            rho = 0.0
            if cov is not None:
                va, vb = cov.variance(wa), cov.variance(wb)
                if va > 0 and vb > 0:
                    rho = cov.cross_variance(wa, wb) / math.sqrt(va * vb)
            out.append(PodOverlap(
                a, b,
                same / denom if denom > 0 else 0.0,
                opp / denom if denom > 0 else 0.0,
                _jaccard(na, nb),
                _jaccard((pod_events or {}).get(a, set()), (pod_events or {}).get(b, set())),
                bool((pod_groups or {}).get(a, set()) & (pod_groups or {}).get(b, set())),
                min(abs(ba), abs(bb)) if ba * bb > 0 else 0.0,
                dot / norm if norm > 0 else 0.0,
                rho,
            ))
    return tuple(out)


@dataclass(frozen=True)
class PodBudget:
    pod_id: str
    standalone_vol: float
    cluster_size: float  # sum_j max(rho_ij, 0) including self; identical pods share one vote
    share: float
    risk_budget: float


@dataclass(frozen=True)
class RiskBudgetReport:
    budgets: tuple[PodBudget, ...]
    effective_independent_pods: float  # sum_i 1/cluster_size_i


def correlated_pod_budgets(
    pod_weights: Mapping[str, Mapping[str, float]], cov: CovarianceModel, total_budget: float
) -> RiskBudgetReport:
    """Fixed host rule: share_i proportional to 1 / sum_j max(rho_ij, 0)."""
    var = {p: cov.variance(pod_weights[p]) for p in sorted(pod_weights)}
    pods = [p for p in sorted(var) if var[p] > 0]  # a pod with no risk makes no budget claim
    cluster = {}
    for a in pods:
        total = 1.0
        for b in pods:
            if a != b and var[a] > 0 and var[b] > 0:
                rho = cov.cross_variance(pod_weights[a], pod_weights[b]) / math.sqrt(var[a] * var[b])
                total += max(rho, 0.0)
        cluster[a] = total
    raw = {p: 1.0 / cluster[p] for p in pods}
    norm = sum(raw.values())
    budgets = tuple(
        PodBudget(p, math.sqrt(max(var[p], 0.0) * cov.periods_per_year), cluster[p],
                  raw[p] / norm, total_budget * raw[p] / norm)
        for p in pods
    )
    return RiskBudgetReport(budgets, norm)


# ------------------------------------------------------------------ hedges


@dataclass(frozen=True)
class HedgeDisposition:
    request_id: str
    status: str
    detail: str


@dataclass(frozen=True)
class HedgeLine:
    security_id: str
    dimension: str
    residual_before: float
    loading: float
    desired_weight: float
    target_weight: float
    held_weight: float
    pending_weight: float
    new_order_weight: float  # target - (held + pending): held/pending counted exactly once
    binding: tuple[str, ...]
    satisfied_requests: tuple[str, ...]
    requesters: tuple[str, ...]


@dataclass(frozen=True)
class HedgePlan:
    lines: tuple[HedgeLine, ...]
    dispositions: tuple[HedgeDisposition, ...]
    weights: Mapping[str, float]  # final hedge target weight per instrument (pre vol-scale)
    requesters: Mapping[str, tuple[str, ...]]


def reconcile_hedges(
    base: Mapping[str, float],
    requests: Sequence[HedgeRequest],
    instruments: Sequence[HedgeInstrument],
    *,
    held: Mapping[str, float],
    pending: Mapping[str, float],
    meta: Mapping[str, object],
    official: OfficialLimits,
    internal: InternalPolicy,
    policy: CoordinationPolicy,
    decision_at: dt.datetime,
) -> HedgePlan:
    """Size ONE hedge per exposure dimension from the aggregate residual.

    Requests only name the dimension / instrument and cap the size; duplicates
    (several pods plus the CIO naming the same hedge) are coalesced, never
    summed. ``meta`` must contain every base security and every instrument.
    """
    inst = {i.security_id: i for i in instruments}
    reqs: dict[str, HedgeRequest] = {}
    dispositions: list[HedgeDisposition] = []
    keys: set[tuple[str, str, str]] = set()
    for r in requests:
        if r.request_id in reqs:
            if reqs[r.request_id] != r:
                _fail(ErrorCode.SCHEMA_MISMATCH, "DUPLICATE_INTENT", f"hedge request {r.request_id!r} replayed with different content")
            dispositions.append(HedgeDisposition(r.request_id, "replay_collapsed", "identical request id counted once"))
            continue
        if r.security_id not in inst:
            _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "HEDGE_NOT_ALLOWED", f"{r.security_id!r} is not in the explicit hedge universe")
        if r.dimension not in ("beta", "net") and not r.dimension.startswith("sector:"):
            _fail(ErrorCode.SCHEMA_MISMATCH, "BAD_HEDGE_DIMENSION", r.dimension)
        if _finite(r.max_abs_weight, "max_abs_weight") <= 0:
            _fail(ErrorCode.SCHEMA_MISMATCH, "BAD_HEDGE_CAP", "max_abs_weight must be positive")
        if require_aware(r.as_of, "as_of") > decision_at:
            _fail(ErrorCode.ASOF_VIOLATION, "FUTURE_HEDGE_REQUEST", r.request_id)
        if decision_at >= require_aware(r.expires_at, "expires_at"):
            _fail(ErrorCode.FORECAST_EXPIRED, "EXPIRED_HEDGE_REQUEST", r.request_id)
        key = (r.requester, r.security_id, r.dimension)
        if key in keys:
            _fail(ErrorCode.SCHEMA_MISMATCH, "DUPLICATE_INTENT", f"requester {r.requester!r} sent two requests for {r.security_id}/{r.dimension}")
        keys.add(key)
        reqs[r.request_id] = r

    w = dict(base)
    lines: list[HedgeLine] = []
    requesters: dict[str, tuple[str, ...]] = {}
    by_dim: dict[str, list[HedgeRequest]] = {}
    for r in sorted(reqs.values(), key=lambda x: x.request_id):
        by_dim.setdefault(r.dimension, []).append(r)
    for dim in sorted(by_dim):
        by_inst: dict[str, list[HedgeRequest]] = {}
        for r in by_dim[dim]:
            by_inst.setdefault(r.security_id, []).append(r)
        usable = {}
        for sid, rr in by_inst.items():
            if _dim_loading(dim, inst[sid]) == 0.0:
                dispositions += [HedgeDisposition(r.request_id, "rejected_no_exposure_loading", f"{sid} has no loading on {dim}") for r in rr]
            else:
                usable[sid] = rr
        if not usable:
            continue
        primary = sorted(usable, key=lambda s: (-len({r.requester for r in usable[s]}), s))[0]
        for sid, rr in usable.items():
            if sid != primary:
                dispositions += [HedgeDisposition(r.request_id, "redundant_alternative", f"{dim} hedged once via {primary}") for r in rr]
        prs = usable[primary]
        ins = inst[primary]
        loading = _dim_loading(dim, ins)
        h0 = w.get(primary, 0.0)
        w0 = {k: v for k, v in w.items() if k != primary}
        resid = _dim_exposure(dim, w, meta)
        desired = h0 if abs(resid) <= policy.hedge_residual_tolerance else -_dim_exposure(dim, w0, meta) / loading
        adv_cap = policy.hedge_adv_participation * ins.adv_usd * internal.execution_days / internal.capital_usd
        bounds = [
            ("request_cap", -max(r.max_abs_weight for r in prs), max(r.max_abs_weight for r in prs)),
            ("name_cap", -official.max_abs_name_weight, official.max_abs_name_weight),
            ("instrument_cap", -ins.max_abs_weight, ins.max_abs_weight),
            ("adv", -adv_cap, adv_cap),
        ]
        if not ins.shortable:
            bounds.append(("not_shortable", 0.0, math.inf))
        if ins.market_cap_usd < official.min_short_market_cap_usd:
            bounds.append(("short_market_cap", 0.0, math.inf))
        room = max(0.0, official.max_gross - _gross(w0))
        n0 = sum(w0.values())
        s0 = _sector_net(w0, meta).get(ins.sector, 0.0)
        b0 = _beta_net(w0, meta)
        sec_b, beta_b = max(policy.max_sector_net, abs(s0)), max(policy.max_abs_beta_net, abs(b0))
        bounds += [("gross", -room, room), ("net", official.min_net - n0, official.max_net - n0),
                   ("sector", -sec_b - s0, sec_b - s0)]
        if ins.beta != 0.0:
            lo_b, hi_b = sorted(((-beta_b - b0) / ins.beta, (beta_b - b0) / ins.beta))
            bounds.append(("beta", lo_b, hi_b))
        lo, hi = max(b[1] for b in bounds), min(b[2] for b in bounds)
        target = 0.0 if lo > hi else min(max(desired, lo), hi)
        target = _q(target)
        binding = tuple(n for n, a, b in bounds if desired < a - _EPS or desired > b + _EPS)
        w[primary] = target
        reqs_ids = tuple(r.request_id for r in prs)
        who = tuple(sorted({r.requester for r in prs}))
        requesters[primary] = who
        hp, pp = held.get(primary, 0.0), pending.get(primary, 0.0)
        lines.append(HedgeLine(primary, dim, resid, loading, desired, target, hp, pp,
                               target - (hp + pp), binding, reqs_ids, who))
        status = "not_needed" if abs(resid) <= policy.hedge_residual_tolerance and target == 0.0 else (
            "constrained" if binding else "satisfied_by_shared_hedge")
        detail = f"target {target} via {primary} for {dim}; requesters {who}; counted once"
        dispositions += [HedgeDisposition(r.request_id, status, detail) for r in prs]
    weights = {l.security_id: w[l.security_id] for l in lines}
    return HedgePlan(tuple(lines), tuple(dispositions), weights, requesters)


# ------------------------------------------------------------------ economic de-duplication


@dataclass(frozen=True)
class EconomicGroup:
    group_id: str
    nodes: tuple[tuple[str, str], ...]  # (pod_id, family label or '_')
    pods: tuple[str, ...]
    basis: tuple[str, ...]
    forecast_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForecastDisposition:
    forecast: PodForecast  # the original, untouched
    status: str  # live | superseded | replay_collapsed
    group_id: Optional[str]


def _pearson(a: list[float], b: list[float]) -> Optional[float]:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def group_forecasts(live: Sequence[PodForecast], policy: CoordinationPolicy) -> tuple[EconomicGroup, ...]:
    node = lambda f: (f.pod_id, f.family_label or "_")  # noqa: E731
    nodes = sorted({node(f) for f in live})
    parent = {n: n for n in nodes}
    why: list[tuple[tuple[str, str], str]] = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, reason):
        ra, rb = find(a), find(b)
        why.append((a, reason))
        why.append((b, reason))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[tuple[str, str], tuple[str, str]] = {}
    for f in sorted(live, key=lambda f: f.forecast_id):
        # A shared broad source (e.g. one characteristics panel feeding every strategy) is NOT duplicate
        # economics; only an explicit family label, event or measurement id merges. Alpha correlation below
        # still merges identical economics that were declared under different labels.
        keys = [("label", f.family_label)] if f.family_label else []
        keys += [("event", e) for e in f.event_ids] + [("measurement", m) for m in f.measurement_ids]
        for k in keys:
            other = first.setdefault(k, node(f))
            if other != node(f):
                union(other, node(f), f"shared_{k[0]}:{k[1]}")
    vec: dict[tuple[str, str], dict[str, list[float]]] = {n: {} for n in nodes}
    for f in live:
        vec[node(f)].setdefault(f.security_id, []).append(f.expected_excess_return)
    mean = {n: {s: sum(v) / len(v) for s, v in d.items()} for n, d in vec.items()}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if a[0] == b[0]:
                continue
            common = sorted(set(mean[a]) & set(mean[b]))
            if len(common) < policy.duplicate_min_common:
                continue
            rho = _pearson([mean[a][s] for s in common], [mean[b][s] for s in common])
            if rho is not None and rho >= policy.duplicate_corr_threshold:
                union(a, b, f"alpha_correlation={rho:.4f}")
    members: dict[tuple[str, str], list] = {}
    for n in nodes:
        members.setdefault(find(n), []).append(n)
    groups = []
    for root, ns in members.items():
        ns = sorted(ns)
        basis = tuple(sorted({r for n, r in why if find(n) == root})) or ("distinct_family",)
        gid = "grp_" + hashlib.sha1("|".join(f"{p}/{l}" for p, l in ns).encode()).hexdigest()[:10]
        fids = tuple(sorted(f.forecast_id for f in live if node(f) in ns))
        groups.append(EconomicGroup(gid, tuple(ns), tuple(sorted({p for p, _ in ns})), basis, fids))
    return tuple(sorted(groups, key=lambda g: g.group_id))


def overlap_diagnostics(live: Sequence[PodForecast], groups: Sequence[EconomicGroup]) -> tuple[str, ...]:
    """Shared sources / declared exposures between DISTINCT economic groups. Evidence only; nothing is merged."""
    group_of = {fid: g.group_id for g in groups for fid in g.forecast_ids}
    seen: dict[tuple[str, str], set[str]] = {}
    for f in live:
        for kind, ids in (("source", f.source_ids), ("exposure", f.exposure_ids)):
            for i in ids:
                seen.setdefault((kind, i), set()).add(group_of[f.forecast_id])
    return tuple(
        f"shared_{k}:{i} across distinct groups {','.join(sorted(g))} (not merged)"
        for (k, i), g in sorted(seen.items()) if len(g) > 1
    )


# ------------------------------------------------------------------ result


@dataclass(frozen=True)
class CoordinationResult:
    input_digest: str
    state_digest: str
    target_book: TargetBook
    risk_review: RiskReview
    actual_review: Optional[RiskReview]
    post_fill_review: Optional[RiskReview]
    groups: tuple[EconomicGroup, ...]
    forecast_dispositions: tuple[ForecastDisposition, ...]
    hedge_plan: HedgePlan
    overlap: tuple[PodOverlap, ...]
    risk_budgets: RiskBudgetReport
    attribution: Mapping[str, Mapping[str, float]]  # pod (or CIO) -> security -> final weight; sums to the book
    risk_contributions: Mapping[str, float]  # annualised vol contribution; sums to ex-ante vol
    trades: Mapping[str, float]  # final target - (held + pending), signed NAV fraction
    vol_scale: float
    ex_ante_vol: float
    exposures: Mapping[str, float]
    stress: tuple[StressResult, ...]
    accounting_refs: Mapping[str, object]
    overlap_diagnostics: tuple[str, ...] = ()
    covariance: Optional[CovarianceModel] = None  # the exact model that sized and reviewed this book

    @property
    def executable(self) -> bool:
        return self.risk_review.decision in (RiskDecision.PASS, RiskDecision.PASS_WITH_REMEDY)

    def is_current_for(self, snapshot: PortfolioSnapshot, requests: Sequence[HedgeRequest]) -> bool:
        return self.state_digest == state_digest(snapshot, requests)


# ------------------------------------------------------------------ coordinator


class CIOCoordinator:
    def __init__(
        self,
        official: OfficialLimits,
        internal: InternalPolicy,
        policy: CoordinationPolicy | None = None,
        covariance_policy: CovariancePolicy | None = None,
    ):
        self._official = official
        self._internal = internal
        self._policy = policy or CoordinationPolicy()
        self._estimator = CovarianceEstimator(covariance_policy)

    # -- public --
    def recompute_after_fills(
        self, inputs: CoordinationInputs, fills: Sequence[FillFeedback], *, as_of: dt.datetime
    ) -> CoordinationResult:
        """Partial-fill feedback invalidates the prior plan; hedge need is recomputed from new state."""
        snap = inputs.snapshot.apply_fills(fills, as_of=as_of)
        new = replace(
            inputs, snapshot=snap, decision_at=max(inputs.decision_at, require_aware(as_of, "as_of")),
            run_id=f"{inputs.run_id}.fb{_digest(snap.snapshot_id)[:6]}",
        )
        return self.coordinate(new)

    def coordinate(
        self, inputs: CoordinationInputs, *, fee_reserve: float = 0.0, execution_vol_buffer: float = 0.0
    ) -> CoordinationResult:
        """``fee_reserve`` is the fraction of NAV pre-reserved for trading costs (0 = legacy sizing).

        Fees lower post-trade NAV, which scales actual weights (and vol) by 1/(1-fees/NAV). Sizing the vol
        budget at ``max_ex_ante_vol * (1 - fee_reserve)`` keeps actual post-fee vol within the unchanged policy
        cap whenever realised fees stay within the reserve. ``execution_vol_buffer`` additionally
        reserves covariance volatility for planned trades that execution will skip or leave unfilled.
        Both are pre-trade sizing reductions; the policy and actual-holdings review stay unchanged.
        """
        if not (0.0 <= fee_reserve < 1.0) or not math.isfinite(fee_reserve):
            _fail(ErrorCode.PORTFOLIO_INFEASIBLE, "BAD_FEE_RESERVE", f"fee_reserve {fee_reserve!r} must be in [0, 1)")
        if not math.isfinite(execution_vol_buffer) or execution_vol_buffer < 0.0:
            _fail(ErrorCode.PORTFOLIO_INFEASIBLE, "BAD_EXECUTION_VOL_BUFFER", "execution volatility buffer must be finite and nonnegative")
        pol, official, internal = self._policy, self._official, self._internal
        vol_cap = pol.max_ex_ante_vol * (1.0 - fee_reserve) - execution_vol_buffer
        if vol_cap <= 0.0:
            _fail(ErrorCode.PORTFOLIO_INFEASIBLE, "EXECUTION_RISK_UNBOUND", "execution uncertainty consumes the covariance volatility budget")
        decision_at = require_aware(inputs.decision_at, "decision_at")
        snap = inputs.snapshot
        meta = self._meta(inputs)
        held, pending = self._check_snapshot(inputs, meta)
        live, dispositions = self._check_forecasts(inputs, meta)
        requests = inputs.all_hedge_requests()
        self._check_requesters(inputs)
        groups = group_forecasts(live, pol)
        group_of = {fid: g.group_id for g in groups for fid in g.forecast_ids}

        # intended pod weights (given, or derived from alpha and flagged)
        pod_weights = self._pod_weights(inputs, live, meta)
        needed = (
            {f.security_id for f in live} | set(held) | set(pending)
            | {i.security_id for i in inputs.hedge_universe}
            | {s for w in pod_weights.values() for s in w}
        )
        cov = self._covariance(inputs, needed, decision_at)

        integrated, shares = self._integrate(live, groups, group_of)
        candidates = [
            Candidate(
                security_id=s, alpha=integrated[s], market_cap_usd=meta[s].market_cap_usd,
                tradable=meta[s].tradable, incumbent=held.get(s, 0.0) != 0.0,
                incumbent_long=held.get(s, 0.0) > 0, incumbent_short=held.get(s, 0.0) < 0,
                sector=meta[s].sector, beta=meta[s].beta,
                # ConstructionEngine has no shortable flag: +inf short interest excludes the name from shorts
                short_interest_pct=meta[s].short_interest_pct if meta[s].shortable else math.inf,
                price_usd=meta[s].price_usd, adv_usd=meta[s].adv_usd,
            )
            for s in sorted(integrated)
        ]
        pubs = sorted({f.publication_id for f in live})
        base_book = ConstructionEngine(official, internal).build(
            candidates, run_id=inputs.run_id, decision_at=decision_at, forecast_publication_ids=pubs,
            current_book_snapshot_id=snap.snapshot_id, nav_snapshot_id=snap.nav_snapshot_id,
            price_snapshot_id=snap.price_snapshot_id, risk_envelope_version=pol.policy_version,
            covariance_version=cov.version, cost_model_version=inputs.cost_model_version,
            attribution_method=ATTRIBUTION_METHOD,
        )
        base = {p.security_id: float(p.target_weight) for p in base_book.positions}
        plan = reconcile_hedges(
            base, requests, inputs.hedge_universe, held=held, pending=pending, meta=meta,
            official=official, internal=internal, policy=pol, decision_at=decision_at,
        )
        pre = {**base, **{s: v for s, v in plan.weights.items() if v != 0.0}}

        # covariance-driven sizing: scale the whole book to the vol budget
        vol_pre = cov.volatility(pre)
        scale = min(1.0, vol_cap / vol_pre) if vol_pre > 0 else 1.0

        def quantized(sc: float) -> dict[str, float]:
            d = {s: _q(v * sc) for s, v in pre.items()}
            return {s: v for s, v in d.items() if v != 0.0}

        final = quantized(scale)
        for _ in range(5):  # truncation can move vol marginally; never leave the book above the cap
            vol_f = cov.volatility(final)
            if vol_f <= vol_cap or vol_f == 0.0:
                break
            scale *= vol_cap / vol_f * (1.0 - 1e-9)
            final = quantized(scale)
        relaxations = list(base_book.relaxations)
        if scale < 1.0:
            relaxations.append(Relaxation(
                step=1, description=f"reduced desired gross by factor {scale:.6f} to meet the covariance vol budget",
                estimated_cost="lower gross and expected alpha than requested"))
        for line in plan.lines:
            if line.binding:
                relaxations.append(Relaxation(
                    step=3, description=f"hedge {line.security_id}/{line.dimension} limited by {','.join(line.binding)}",
                    estimated_cost="residual exposure left unhedged"))

        attribution = self._attribute(final, base, plan, shares, scale)
        vol = cov.volatility(final)
        g = cov.cov_times(final)
        rc = {p: (cov.periods_per_year * sum(a[s] * g[s] for s in a) / vol if vol > 0 else 0.0)
              for p, a in attribution.items()}
        budgets = correlated_pod_budgets(pod_weights, cov, pol.max_ex_ante_vol)
        share_of = {b.pod_id: b.share for b in budgets.budgets}
        ratio = max([(rc.get(p, 0.0) / vol) / share_of[p] for p in share_of if vol > 0] or [0.0])
        stress, worst, pod_share = self._stress(final, meta, cov, attribution, set(share_of))
        exposures = self._exposures(final, meta, cov, vol, worst)
        exposures.update(vol_scale_floor=scale, pod_risk_share_ratio=ratio, stress_pod_share=pod_share)

        book = self._finalize_book(base_book, final, relaxations, scale, vol, exposures, group_of, pubs)
        if fee_reserve or execution_vol_buffer:
            book = book.model_copy(update={"diagnostics": {**book.diagnostics,
                "fee_reserve": repr(fee_reserve), "execution_vol_buffer": repr(execution_vol_buffer),
                "sized_vol_cap": repr(vol_cap)}})
        review = self._review("target", book, exposures, stress, decision_at)
        actual_review = self._review_weights(
            f"actual_{snap.snapshot_id}", held, "actual", inputs, meta, cov, decision_at)
        post_w = dict(held)
        for s, v in pending.items():
            post_w[s] = post_w.get(s, 0.0) + v
        post_review = self._review_weights(
            f"postfill_{snap.snapshot_id}", post_w, "actual", inputs, meta, cov, decision_at)

        trades = {}
        for s in sorted(set(final) | set(held) | set(pending)):
            d = final.get(s, 0.0) - (held.get(s, 0.0) + pending.get(s, 0.0))
            if abs(d) > _EPS:
                trades[s] = d
        refs = {
            "snapshot_id": snap.snapshot_id, "nav_snapshot_id": snap.nav_snapshot_id,
            "price_snapshot_id": snap.price_snapshot_id,
            "pending_order_ids": tuple(sorted(o.order_id for o in snap.pending_orders)),
            "applied_fill_events": tuple(e for e, _ in snap.applied_fill_events),
            "forecast_publication_ids": tuple(pubs), "forecast_ids": tuple(sorted(f.forecast_id for f in live)),
            "hedge_request_ids": tuple(sorted({r.request_id for r in requests})),
            "covariance_version": cov.version, "risk_policy_version": pol.policy_version,
        }
        events = {p: {e for f in live if f.pod_id == p for e in f.event_ids} for p in pod_weights}
        pgroups = {p: {g.group_id for g in groups if p in g.pods} for p in pod_weights}
        return CoordinationResult(
            inputs.digest(), state_digest(snap, requests), book, review, actual_review, post_review, groups,
            tuple(dispositions_with_groups(dispositions, group_of)), plan,
            measure_pod_overlap(pod_weights, meta, pod_events=events, pod_groups=pgroups, cov=cov),
            budgets, attribution, rc, trades, scale, vol, exposures, tuple(stress), refs,
            overlap_diagnostics(live, groups), cov,
        )

    # -- validation --
    def _meta(self, inputs: CoordinationInputs) -> dict[str, object]:
        meta: dict[str, object] = {}
        for m in inputs.universe:
            if m.security_id in meta:
                _fail(ErrorCode.IDENTITY_AMBIGUOUS, "DUPLICATE_SECURITY", m.security_id)
            for name in ("market_cap_usd", "beta", "price_usd", "adv_usd"):
                _finite(getattr(m, name), name)
            meta[m.security_id] = m
        for i in inputs.hedge_universe:
            if i.security_id in meta:
                _fail(ErrorCode.IDENTITY_AMBIGUOUS, "HEDGE_OVERLAPS_ALPHA_UNIVERSE", i.security_id)
            if _finite(i.adv_usd, "adv_usd") <= 0 or _finite(i.beta, "beta") != i.beta:
                _fail(ErrorCode.LIQUIDITY_UNAVAILABLE, "BAD_HEDGE_INSTRUMENT", i.security_id)
            meta[i.security_id] = i
        return meta

    def _check_snapshot(self, inputs, meta) -> tuple[dict[str, float], dict[str, float]]:
        snap, decision_at = inputs.snapshot, inputs.decision_at
        as_of = require_aware(snap.as_of, "snapshot.as_of")
        if as_of > decision_at:
            _fail(ErrorCode.ASOF_VIOLATION, "FUTURE_SNAPSHOT", snap.snapshot_id)
        if decision_at - as_of > dt.timedelta(minutes=self._policy.max_snapshot_age_minutes):
            _fail(ErrorCode.ASOF_VIOLATION, "STALE_SNAPSHOT", f"{snap.snapshot_id} is older than the allowed age")
        held: dict[str, float] = {}
        for h in snap.holdings:
            if h.security_id not in meta:
                _fail(ErrorCode.IDENTITY_AMBIGUOUS, "UNKNOWN_SECURITY", f"holding {h.security_id!r}")
            if h.security_id in held:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "DUPLICATE_HOLDING", h.security_id)
            held[h.security_id] = _finite(h.weight, "holding weight")
        pending: dict[str, float] = {}
        ids: set[str] = set()
        for o in snap.pending_orders:
            if o.security_id not in meta:
                _fail(ErrorCode.IDENTITY_AMBIGUOUS, "UNKNOWN_SECURITY", f"pending order {o.order_id!r}")
            if o.order_id in ids:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "DUPLICATE_ORDER", o.order_id)
            ids.add(o.order_id)
            pending[o.security_id] = pending.get(o.security_id, 0.0) + _finite(o.remaining_weight, "remaining_weight")
        return held, pending

    def _check_forecasts(self, inputs, meta):
        pol, decision_at = self._policy, inputs.decision_at
        auth: dict[str, PodAuthorization] = {}
        for a in inputs.authorizations:
            if a.pod_id in auth:
                _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "DUPLICATE_AUTHORIZATION", a.pod_id)
            auth[a.pod_id] = a
        ordered: list[PodForecast] = []
        by_id: dict[str, PodForecast] = {}
        collapsed: list[PodForecast] = []
        for it in inputs.intentions:
            a = auth.get(it.pod_id)
            if a is None or not (a.authorized_from <= decision_at and (a.authorized_until is None or decision_at < a.authorized_until)):
                _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "UNAUTHORIZED_POD", f"{it.pod_id!r} not authorized at decision time")
            for f in it.forecasts:
                if f.pod_id != it.pod_id:
                    _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "FORECAST_POD_MISMATCH", f.forecast_id)
                prior = by_id.get(f.forecast_id)
                if prior is None:
                    by_id[f.forecast_id] = f
                    ordered.append(f)
                elif prior == f:
                    collapsed.append(f)
                else:
                    _fail(ErrorCode.SCHEMA_MISMATCH, "DUPLICATE_INTENT", f"forecast id {f.forecast_id!r} replayed with different content")
        superseded: set[str] = set()
        for f in ordered:
            if f.supersedes:
                t = by_id.get(f.supersedes)
                if t is None or t.pod_id != f.pod_id or t.security_id != f.security_id:
                    _fail(ErrorCode.SCHEMA_MISMATCH, "BAD_SUPERSEDES", f.forecast_id)
                superseded.add(f.supersedes)
        live = [f for f in ordered if f.forecast_id not in superseded]
        seen: set[tuple[str, str, str]] = set()
        for f in live:
            key = (f.pod_id, f.family_label or "_", f.security_id)
            if key in seen:
                _fail(ErrorCode.SCHEMA_MISMATCH, "DUPLICATE_INTENT", f"unexplained second live forecast for {key}")
            seen.add(key)
            if f.security_id not in meta or f.security_id in {i.security_id for i in inputs.hedge_universe}:
                _fail(ErrorCode.IDENTITY_AMBIGUOUS, "UNKNOWN_SECURITY", f"forecast {f.forecast_id!r} names {f.security_id!r}")
            _finite(f.expected_excess_return, "expected_excess_return")
            if _finite(f.uncertainty, "uncertainty") <= 0:
                _fail(ErrorCode.SCHEMA_MISMATCH, "BAD_UNCERTAINTY", f.forecast_id)
            if require_aware(f.as_of, "as_of") > decision_at:
                _fail(ErrorCode.ASOF_VIOLATION, "FUTURE_FORECAST", f.forecast_id)
            if f.as_of < auth[f.pod_id].authorized_from:
                _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "FORECAST_BEFORE_AUTHORIZATION", f.forecast_id)
            information_at = require_aware(f.information_cutoff or f.as_of, 'information_cutoff')
            if information_at > f.as_of:
                raise ValueError('forecast information cutoff follows publication')
            if decision_at >= require_aware(f.expires_at, "expires_at") or decision_at - information_at > dt.timedelta(hours=pol.max_forecast_age_hours):
                _fail(ErrorCode.FORECAST_EXPIRED, "STALE_FORECAST", f.forecast_id)
        if not live:
            _fail(ErrorCode.INSUFFICIENT_HISTORY, "NO_LIVE_FORECASTS", "no authorized live forecast")
        status = [(f, "live") for f in live] + [(f, "superseded") for f in ordered if f.forecast_id in superseded]
        status += [(f, "replay_collapsed") for f in collapsed]
        return live, status

    def _check_requesters(self, inputs) -> None:
        for it in inputs.intentions:
            for r in it.hedge_requests:
                if r.requester != it.pod_id:
                    _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "HEDGE_REQUESTER_MISMATCH", r.request_id)
        for r in inputs.cio_hedge_requests:
            if r.requester != CIO:
                _fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, "HEDGE_REQUESTER_MISMATCH", r.request_id)

    def _pod_weights(self, inputs, live, meta) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for it in inputs.intentions:
            if it.intended_weights:
                for s, v in it.intended_weights.items():
                    if s not in meta:
                        _fail(ErrorCode.IDENTITY_AMBIGUOUS, "UNKNOWN_SECURITY", f"intended weight {s!r} of {it.pod_id!r}")
                    _finite(v, "intended weight")
                out[it.pod_id] = {s: float(v) for s, v in it.intended_weights.items()}
            else:
                alpha: dict[str, float] = {}
                for f in live:
                    if f.pod_id == it.pod_id:
                        alpha[f.security_id] = alpha.get(f.security_id, 0.0) + f.expected_excess_return
                tot = sum(abs(a) for a in alpha.values())
                out[it.pod_id] = {s: self._policy.derived_pod_gross * a / tot for s, a in alpha.items()} if tot > 0 else {}
        return out

    def _covariance(self, inputs, needed, decision_at) -> CovarianceModel:
        try:
            if (inputs.covariance is None) == (not inputs.return_history):
                _fail(ErrorCode.SCHEMA_MISMATCH, "COVARIANCE_SOURCE", "provide exactly one of return_history or covariance")
            if inputs.covariance is not None:
                cov = inputs.covariance
                if not cov.pd_checked:
                    raise CovarianceError("PD_UNVERIFIABLE", "supplied covariance carries no positive-definiteness proof")
                if cov.as_of > decision_at:
                    raise CovarianceError("FUTURE_ROW", "supplied covariance is dated after the decision")
                missing = sorted(needed - set(cov.security_ids))
                if missing:
                    raise CovarianceError("UNKNOWN_EXPOSURE", f"no covariance for {missing[:5]}")
                return cov
            return self._estimator.estimate(inputs.return_history, sorted(needed), as_of=decision_at)
        except CovarianceError as exc:
            code = {
                "TOO_SHORT": ErrorCode.INSUFFICIENT_HISTORY, "MISSING_VALUE": ErrorCode.INSUFFICIENT_HISTORY,
                "FUTURE_ROW": ErrorCode.ASOF_VIOLATION, "STALE": ErrorCode.ASOF_VIOLATION,
                "UNKNOWN_EXPOSURE": ErrorCode.IDENTITY_AMBIGUOUS, "NOT_PSD": ErrorCode.PORTFOLIO_INFEASIBLE,
            }.get(exc.code, ErrorCode.SCHEMA_MISMATCH)
            _fail(code, f"COVARIANCE_{exc.code}", str(exc))

    # -- integration & attribution --
    def _integrate(self, live, groups, group_of):
        gids = [g.group_id for g in groups]
        integrator = FamilyIntegrator(gids)
        per: dict[str, dict[str, list[PodForecast]]] = {}
        for f in live:
            per.setdefault(f.security_id, {}).setdefault(group_of[f.forecast_id], []).append(f)
        out: dict[str, float] = {}
        shares: dict[str, dict[str, float]] = {}
        for s, by_group in per.items():
            ests = []
            for gid, fs in by_group.items():
                # within a group duplicate information is averaged once; uncertainty never shrinks
                ests.append(FamilyEstimate(gid, s, sum(f.expected_excess_return for f in fs) / len(fs),
                                           max(f.uncertainty for f in fs)))
            out[s] = integrator.integrate(ests).alpha
            contrib = {e.family_id: e.expected_excess_return / len(gids) for e in ests}
            total = sum(contrib.values())
            pod_share: dict[str, float] = {}
            for gid, c in contrib.items():
                pods = sorted({f.pod_id for f in by_group[gid]})
                frac = c / total if abs(total) > _EPS else 1.0 / len(contrib)
                for p in pods:
                    pod_share[p] = pod_share.get(p, 0.0) + frac / len(pods)
            shares[s] = pod_share
        return out, shares

    def _attribute(self, final, base, plan, shares, scale) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for s, w in final.items():
            if s in plan.weights:
                who = plan.requesters.get(s) or (CIO,)
                split = {p: 1.0 / len(who) for p in who}
            else:
                split = shares.get(s, {})
            for p, frac in split.items():
                out.setdefault(p, {})[s] = w * frac
        return out

    # -- risk measurement --
    def _exposures(self, w, meta, cov, vol, worst_loss) -> dict[str, float]:
        sectors = _sector_net(w, meta)
        net = sum(w.values())
        return {
            "gross": _gross(w), "net_max": net, "net_min": net,
            "max_abs_name": max((abs(v) for v in w.values()), default=0.0),
            "abs_beta_net": abs(_beta_net(w, meta)),
            "max_sector_net_abs": max((abs(v) for v in sectors.values()), default=0.0),
            "ex_ante_vol": vol, "stress_loss": worst_loss,
        }

    def _stress(self, w, meta, cov, attribution, budget_pods):
        pol = self._policy
        scenarios: list[tuple[str, float, dict[str, float]]] = []
        b = _beta_net(w, meta)
        sign = 1.0 if b >= 0 else -1.0
        scenarios.append(("market_shock", abs(b) * pol.market_shock,
                          {p: sign * _beta_net(a, meta) * pol.market_shock for p, a in attribution.items()}))
        for sec, net in sorted(_sector_net(w, meta).items()):
            sg = 1.0 if net >= 0 else -1.0
            scenarios.append((f"sector_shock:{sec}", abs(net) * pol.sector_shock, {
                p: sg * _sector_net(a, meta).get(sec, 0.0) * pol.sector_shock for p, a in attribution.items()}))
        var_s = cov.stressed_variance(w, pol.stress_corr)
        k = pol.stress_z * math.sqrt(pol.stress_horizon_periods)
        loss_c = k * math.sqrt(max(var_s, 0.0))
        sig = {s: math.sqrt(cov.covariance(s, s)) for s in w}
        tot_sig = sum(sig[s] * w[s] for s in w)
        contrib = {}
        for p, a in attribution.items():
            part = (1 - pol.stress_corr) * sum(a[s] * w[s] * sig[s] ** 2 for s in a) + \
                pol.stress_corr * sum(sig[s] * a[s] for s in a) * tot_sig
            contrib[p] = k * part / math.sqrt(var_s) if var_s > 0 else 0.0
        scenarios.append(("correlation_spike", loss_c, contrib))
        results = [StressResult(scenario_id=n, impact=f"{loss:.8f}", evidence="deterministic fixed-host-policy scenario")
                   for n, loss, _ in scenarios]
        _, worst, parts = max(scenarios, key=lambda x: (x[1], x[0]))
        pods = [p for p in parts if p in budget_pods]
        share = max((parts[p] for p in pods), default=0.0) / worst if worst > 0 and len(pods) >= 2 else 0.0
        return results, worst, share

    def _limits(self, kind: str) -> RiskPolicy:
        p, o = self._policy, self._official
        lim = [
            RiskLimit(name="gross", threshold=f"<= {o.max_gross}"),
            RiskLimit(name="net_max", threshold=f"<= {o.max_net}"),
            RiskLimit(name="net_min", threshold=f">= {o.min_net}"),
            RiskLimit(name="max_abs_name", threshold=f"<= {o.max_abs_name_weight}"),
            RiskLimit(name="abs_beta_net", threshold=f"<= {p.max_abs_beta_net}"),
            RiskLimit(name="max_sector_net_abs", threshold=f"<= {p.max_sector_net}"),
            RiskLimit(name="ex_ante_vol", threshold=f"<= {p.max_ex_ante_vol}"),
            RiskLimit(name="stress_loss", threshold=f"<= {p.max_stress_loss}"),
        ]
        if kind == "target":
            lim += [
                RiskLimit(name="vol_scale_floor", threshold=f">= {p.min_vol_scale}"),
                RiskLimit(name="pod_risk_share_ratio", threshold=f"<= {p.pod_budget_tolerance}", hard=False),
                RiskLimit(name="stress_pod_share", threshold=f"<= {p.max_stress_pod_share}", hard=False),
            ]
        return RiskPolicy(policy_version=p.policy_version, limits=lim, review_ttl_minutes=p.review_ttl_minutes)

    def _measured(self, ex: Mapping[str, float], kind: str) -> list[MeasuredExposure]:
        p, o = self._policy, self._official
        spec = {
            "gross": (o.max_gross, True), "net_max": (o.max_net, True), "net_min": (o.min_net, False),
            "max_abs_name": (o.max_abs_name_weight, True), "abs_beta_net": (p.max_abs_beta_net, True),
            "max_sector_net_abs": (p.max_sector_net, True), "ex_ante_vol": (p.max_ex_ante_vol, True),
            "stress_loss": (p.max_stress_loss, True),
        }
        if kind == "target":
            spec.update(vol_scale_floor=(p.min_vol_scale, False), pod_risk_share_ratio=(p.pod_budget_tolerance, True),
                        stress_pod_share=(p.max_stress_pod_share, True))
        return [MeasuredExposure(n, ex[n], t, m) for n, (t, m) in spec.items()]

    def _review(self, kind, book, exposures, stress, at) -> RiskReview:
        return RiskEngine(self._limits(kind)).review(
            book, self._measured(exposures, kind), stress_results=stress, effective_at=at,
            reviewed_kind="target" if kind == "target" else "actual")

    def _review_weights(self, book_id, weights, kind, inputs, meta, cov, at) -> Optional[RiskReview]:
        w = {s: v for s, v in weights.items() if abs(v) > _EPS}
        if not w:
            return None
        snap = inputs.snapshot
        positions = [{"security_id": s, "target_weight": Decimal(repr(round(v, 10)))} for s, v in sorted(w.items())]
        book = TargetBook(
            owner="allocation", created_at=at, target_book_id=book_id, run_id=inputs.run_id, decision_at=at,
            forecast_publication_ids=[snap.snapshot_id], current_book_snapshot_id=snap.snapshot_id,
            nav_snapshot_id=snap.nav_snapshot_id, price_snapshot_id=snap.price_snapshot_id,
            risk_envelope_version=self._policy.policy_version, covariance_version=cov.version,
            cost_model_version=inputs.cost_model_version, positions=positions,
            solver_status=SolverStatus.FEASIBLE, attribution_method="actual_holdings_no_attribution",
        )
        attr = {"actual": w}
        stress, worst, _ = self._stress(w, meta, cov, attr, set())
        ex = self._exposures(w, meta, cov, cov.volatility(w), worst)
        return self._review(kind, book, ex, stress, at)

    def _finalize_book(self, base_book, final, relaxations, scale, vol, exposures, group_of, pubs) -> TargetBook:
        o = self._official
        gross, net = _gross(final), sum(final.values())
        biggest = max((abs(v) for v in final.values()), default=0.0)
        tol = 1e-9
        results = [
            ConstraintResult(name="position_count", threshold=f"{o.min_positions}..{o.max_positions}",
                             value=str(len(final)),
                             status="pass" if o.min_positions <= len(final) <= o.max_positions else "fail"),
            ConstraintResult(name="gross", threshold=f"<= {o.max_gross}", value=str(gross),
                             status="pass" if gross <= o.max_gross + tol else "fail"),
            ConstraintResult(name="net", threshold=f"[{o.min_net}, {o.max_net}]", value=str(net),
                             status="pass" if o.min_net - tol <= net <= o.max_net + tol else "fail"),
            ConstraintResult(name="max_abs_name_weight", threshold=f"<= {o.max_abs_name_weight}", value=str(biggest),
                             status="pass" if biggest <= o.max_abs_name_weight + tol else "fail"),
        ] + [r for r in base_book.official_constraint_results if r.name == "short_market_cap"]
        if any(r.status == "fail" for r in results):
            _fail(ErrorCode.PORTFOLIO_INFEASIBLE, "POST_HEDGE_OFFICIAL_LIMIT",
                  ", ".join(r.name for r in results if r.status == "fail"))
        data = base_book.model_dump()
        status = base_book.solver_status
        if relaxations and status in (SolverStatus.FEASIBLE, SolverStatus.OPTIMAL):
            status = SolverStatus.RELAXED
        data.update(
            positions=[{"security_id": s, "target_weight": Decimal(repr(v))} for s, v in sorted(final.items())],
            official_constraint_results=[r.model_dump() for r in results],
            relaxations=[r.model_dump() for r in relaxations], solver_status=status,
            diagnostics={**base_book.diagnostics, "vol_scale": f"{scale:.12f}", "ex_ante_vol": f"{vol:.12f}",
                         "gross": str(gross), "net": str(net), "economic_groups": str(len(set(group_of.values()))),
                         "abs_beta_net": f"{exposures['abs_beta_net']:.12f}"},
        )
        return TargetBook.model_validate(data)


def dispositions_with_groups(status_pairs, group_of) -> list[ForecastDisposition]:
    return [ForecastDisposition(f, st, group_of.get(f.forecast_id)) for f, st in status_pairs]
