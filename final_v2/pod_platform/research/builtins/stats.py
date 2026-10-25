"""Small pure-Python numerics (no numpy). Every function returns ``None`` instead of a fabricated number."""

from __future__ import annotations

import math
from typing import Optional, Sequence


def mean(values: Sequence[float]) -> Optional[float]:
    return math.fsum(values) / len(values) if values else None


def stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mu = math.fsum(values) / len(values)
    return math.sqrt(
        math.fsum((v - mu) ** 2 for v in values) / (len(values) - 1)
    )


def average_ranks(values: Sequence[float]) -> list[float]:
    """0-based ranks; ties share their average rank, so flat inputs carry no manufactured skill."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def centered_rank(values: Sequence[float]) -> list[float]:
    """Cross-sectional percentile centred on zero, in (-0.5, 0.5)."""
    n = len(values)
    if n == 0:
        return []
    return [(r + 0.5) / n - 0.5 for r in average_ranks(values)]


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n != len(y) or n < 3:
        return None
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx <= 1e-300 or syy <= 1e-300:
        return None
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    value = sxy / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, value))


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def solve_linear(
    matrix: list[list[float]], rhs: list[float]
) -> Optional[list[float]]:
    """Gaussian elimination with partial pivoting; ``None`` if (numerically) singular."""
    n = len(rhs)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            if factor:
                for k in range(col, n + 1):
                    a[row][k] -= factor * a[col][k]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        out[row] = (
            a[row][n] - math.fsum(a[row][k] * out[k] for k in range(row + 1, n))
        ) / a[row][row]
    return out


def hac_se(series: Sequence[float], lag: int) -> Optional[float]:
    """Newey-West (Bartlett) standard error of the mean; falls back to the plain SE if the estimate is not positive."""
    n = len(series)
    if n < 3:
        return None
    mu = math.fsum(series) / n
    dev = [v - mu for v in series]
    lag = max(0, min(lag, n - 2))
    var = math.fsum(d * d for d in dev) / n
    for k in range(1, lag + 1):
        gamma = math.fsum(dev[t] * dev[t - k] for t in range(k, n)) / n
        var += 2.0 * (1.0 - k / (lag + 1)) * gamma
    if var <= 0.0:
        sd = stdev(series)
        return None if sd is None else sd / math.sqrt(n)
    return math.sqrt(var / n)
