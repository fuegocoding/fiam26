"""As-of regularized covariance (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 17, 18).

Estimated only from return rows whose ``observed_at`` (the moment the return is
fully matured and knowable) is at or before ``as_of``. Fixed host policy, no
fitting to fund performance:

- sample covariance over the last ``max_observations`` complete rows;
- shrinkage toward the diagonal with intensity
  ``max(min_shrinkage, N / (T + N))`` so a short history is penalised harder
  (a tiny-history matrix is never used unconstrained);
- variance floor on the diagonal;
- Cholesky verification (a shrunk matrix with positive diagonal is
  mathematically positive definite; the check guards numerics and
  ``from_matrix`` inputs).

Estimation is pure Python, O(T N^2). Positive-definiteness is NEVER skipped:
up to ``pd_check_max_assets`` a pure-Python Cholesky decides; above it, optional
NumPy (``eigvalsh``) decides; without NumPy only a mathematically sufficient
certificate is accepted (strict diagonal dominance via Gershgorin for explicit
matrices; the shrinkage construction for estimator output). If none applies the
matrix is rejected (``PD_UNVERIFIABLE``), i.e. validation fails closed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from pydantic import Field

from ..contracts.base import StrictModel, require_aware


class CovarianceError(ValueError):
    """Raised with a stable ``code`` so callers can fail closed with a reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class CovariancePolicy(StrictModel):
    min_observations: int = Field(ge=3, default=60)
    max_observations: int = Field(ge=3, default=252)
    min_shrinkage: float = Field(ge=0.0, le=1.0, default=0.2)
    variance_floor: float = Field(gt=0.0, default=1e-8)
    periods_per_year: float = Field(gt=0.0, default=252.0)
    max_staleness_days: int = Field(ge=0, default=7)
    max_assets: int = Field(ge=1, default=600)
    pd_check_max_assets: int = Field(ge=1, default=250)


@dataclass(frozen=True)
class ReturnRow:
    observed_at: dt.datetime
    returns: Mapping[str, float]


def _cholesky_ok(matrix: Sequence[Sequence[float]]) -> bool:
    n = len(matrix)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if not s > 1e-12 * max(abs(matrix[i][i]), 1e-300):
                    return False
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return True


def _load_numpy():
    """Optional NumPy; a seam so tests can simulate its absence."""
    try:
        import numpy
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return numpy


def _diagonally_dominant(matrix: Sequence[Sequence[float]]) -> bool:
    """Sufficient (Gershgorin): symmetric, positive diagonal, strictly dominant rows => positive definite."""
    for i, row in enumerate(matrix):
        off = sum(abs(v) for j, v in enumerate(row) if j != i)
        if not row[i] - off > 1e-12 * row[i]:
            return False
    return True


def verify_positive_definite(matrix: Sequence[Sequence[float]], max_python_assets: int) -> str:
    """Return the method that proved positive definiteness, or raise ``CovarianceError``.

    Never returns for an unverified matrix: ``NOT_PSD`` when a check refutes it,
    ``PD_UNVERIFIABLE`` when no available check can decide.
    """
    n = len(matrix)
    if n <= max_python_assets:
        if not _cholesky_ok(matrix):
            raise CovarianceError("NOT_PSD", "matrix is not positive definite")
        return "cholesky_python"
    np = _load_numpy()
    if np is not None:
        smallest = float(np.linalg.eigvalsh(np.array(matrix, dtype=float))[0])
        if not smallest > 1e-12 * max(matrix[i][i] for i in range(n)):
            raise CovarianceError("NOT_PSD", f"matrix is not positive definite (min eigenvalue {smallest:.3e})")
        return "eigvalsh_numpy"
    if _diagonally_dominant(matrix):
        return "gershgorin_diagonal_dominance"
    raise CovarianceError(
        "PD_UNVERIFIABLE",
        f"{n} assets exceeds pd_check_max_assets={max_python_assets}, NumPy is unavailable and no sufficient certificate applies",
    )


@dataclass(frozen=True)
class CovarianceModel:
    security_ids: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    as_of: dt.datetime
    n_observations: int
    shrinkage: float
    periods_per_year: float
    version: str
    pd_checked: bool = True
    pd_method: str = "cholesky_python"
    _index: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {s: i for i, s in enumerate(self.security_ids)})

    # ---- construction from an explicit matrix (validated) ----
    @classmethod
    def from_matrix(
        cls,
        security_ids: Sequence[str],
        matrix: Sequence[Sequence[float]],
        *,
        as_of: dt.datetime,
        periods_per_year: float = 252.0,
        n_observations: int = 0,
        shrinkage: float = 0.0,
        label: str = "explicit",
        pd_check_max_assets: int = 250,
    ) -> "CovarianceModel":
        ids = tuple(security_ids)
        n = len(ids)
        if n == 0 or len(set(ids)) != n or len(matrix) != n or any(len(r) != n for r in matrix):
            raise CovarianceError("SHAPE", "matrix must be square over unique security ids")
        for row in matrix:
            if not all(math.isfinite(v) for v in row):
                raise CovarianceError("NONFINITE", "covariance matrix has non-finite entries")
        for i in range(n):
            if matrix[i][i] <= 0:
                raise CovarianceError("NOT_PSD", f"non-positive variance for {ids[i]!r}")
            for j in range(i):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-12 * max(1.0, abs(matrix[i][j])):
                    raise CovarianceError("NOT_PSD", "matrix is not symmetric")
        method = verify_positive_definite(matrix, pd_check_max_assets)
        rows = tuple(tuple(float(v) for v in r) for r in matrix)
        return cls(ids, rows, require_aware(as_of, "as_of"), n_observations, shrinkage,
                   periods_per_year, _version(label, ids, rows, as_of, n_observations, shrinkage), True, method)

    # ---- queries ----
    def _check(self, weights: Mapping[str, float]) -> None:
        unknown = sorted(s for s in weights if s not in self._index)
        if unknown:
            raise CovarianceError("UNKNOWN_EXPOSURE", f"no covariance for {unknown[:5]}")

    def covariance(self, a: str, b: str) -> float:
        self._check({a: 0.0, b: 0.0})
        return self.matrix[self._index[a]][self._index[b]]

    def correlation(self, a: str, b: str) -> float:
        return self.covariance(a, b) / math.sqrt(self.covariance(a, a) * self.covariance(b, b))

    def cov_times(self, weights: Mapping[str, float]) -> dict[str, float]:
        """(Sigma w)_i for every modelled security (per-period units)."""
        self._check(weights)
        active = [(self._index[s], w) for s, w in weights.items() if w != 0.0]
        return {
            sid: sum(self.matrix[i][j] * w for j, w in active)
            for i, sid in enumerate(self.security_ids)
        }

    def cross_variance(self, a: Mapping[str, float], b: Mapping[str, float]) -> float:
        self._check(a)
        self._check(b)
        sb = self.cov_times(b)
        return sum(w * sb[s] for s, w in a.items())

    def variance(self, weights: Mapping[str, float]) -> float:
        return self.cross_variance(weights, weights)

    def volatility(self, weights: Mapping[str, float]) -> float:
        """Annualised ex-ante volatility."""
        return math.sqrt(max(self.variance(weights), 0.0) * self.periods_per_year)

    def stressed_variance(self, weights: Mapping[str, float], rho: float) -> float:
        """Per-period variance with every pairwise correlation set to ``rho``."""
        self._check(weights)
        sig = {s: math.sqrt(self.covariance(s, s)) * w for s, w in weights.items()}
        return (1.0 - rho) * sum(v * v for v in sig.values()) + rho * sum(sig.values()) ** 2


def _version(label, ids, rows, as_of, n_obs, shrinkage) -> str:
    blob = json.dumps(
        [label, list(ids), [[round(v, 15) for v in r] for r in rows],
         require_aware(as_of, "as_of").isoformat(), n_obs, round(shrinkage, 12)],
        separators=(",", ":"),
    )
    return "cov_" + hashlib.sha256(blob.encode()).hexdigest()[:16]


class CovarianceEstimator:
    def __init__(self, policy: CovariancePolicy | None = None):
        self._policy = policy or CovariancePolicy()

    def estimate(
        self, rows: Sequence[ReturnRow], security_ids: Sequence[str], *, as_of: dt.datetime
    ) -> CovarianceModel:
        p = self._policy
        as_of = require_aware(as_of, "as_of")
        ids = tuple(sorted(set(security_ids)))
        if not ids:
            raise CovarianceError("UNKNOWN_EXPOSURE", "no securities requested")
        if len(ids) > p.max_assets:
            raise CovarianceError("ASSET_CAP", f"{len(ids)} assets exceeds max_assets={p.max_assets}")
        seen: set[dt.datetime] = set()
        for row in rows:
            at = require_aware(row.observed_at, "observed_at")
            if at > as_of:
                raise CovarianceError("FUTURE_ROW", f"row observed at {at.isoformat()} is after as_of")
            if at in seen:
                raise CovarianceError("DUPLICATE_ROW", f"two rows observed at {at.isoformat()}")
            seen.add(at)
        ordered = sorted(rows, key=lambda r: r.observed_at)
        known = {s for r in ordered for s in r.returns}
        missing = [s for s in ids if s not in known]
        if missing:
            raise CovarianceError("UNKNOWN_EXPOSURE", f"no return history for {missing[:5]}")
        window = ordered[-p.max_observations:]
        data: list[list[float]] = []
        for row in window:
            vals = []
            for s in ids:
                if s not in row.returns:
                    raise CovarianceError("MISSING_VALUE", f"{s!r} missing at {row.observed_at.isoformat()}")
                v = row.returns[s]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise CovarianceError("NONFINITE", f"non-finite return for {s!r}")
                vals.append(float(v))
            data.append(vals)
        t, n = len(data), len(ids)
        if t < p.min_observations:
            raise CovarianceError("TOO_SHORT", f"{t} observations < min_observations={p.min_observations}")
        latest = window[-1].observed_at
        if as_of - latest > dt.timedelta(days=p.max_staleness_days):
            raise CovarianceError("STALE", f"latest matured row {latest.isoformat()} is stale")
        means = [sum(r[j] for r in data) / t for j in range(n)]
        dev = [[r[j] - means[j] for j in range(n)] for r in data]
        sample = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                c = sum(d[i] * d[j] for d in dev) / (t - 1)
                sample[i][j] = sample[j][i] = c
        delta = min(1.0, max(p.min_shrinkage, n / (t + n)))
        shrunk = [
            [
                max(sample[i][i], p.variance_floor) if i == j else (1.0 - delta) * sample[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]
        try:
            method = verify_positive_definite(shrunk, p.pd_check_max_assets)
        except CovarianceError as exc:
            if exc.code != "PD_UNVERIFIABLE" or not delta > 0.0:
                raise
            # Sufficient by construction: (1-d)*S is PSD (Gram matrix) and the diagonal part
            # d*diag(S) + floor top-up is positive definite for d > 0 and a positive floor.
            method = "shrinkage_construction"
        rows_t = tuple(tuple(r) for r in shrunk)
        return CovarianceModel(
            ids, rows_t, as_of, t, delta, p.periods_per_year,
            _version("shrunk_diag", ids, rows_t, as_of, t, delta), True, method,
        )
