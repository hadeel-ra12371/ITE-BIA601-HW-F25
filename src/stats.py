"""
Statistical comparison of two algorithms over repeated runs.

Wilcoxon rank-sum test (= Mann-Whitney U), two-sided, normal approximation
with tie correction. Implemented with the standard library only so no extra
dependency (SciPy) is needed.

H0: both algorithms produce final values from the same distribution.
If p < alpha we reject H0, i.e. the difference is statistically significant.
"""

from __future__ import annotations

import math


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties receive the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mann_whitney(a: list[float], b: list[float]) -> dict:
    n1, n2 = len(a), len(b)
    data = list(a) + list(b)
    ranks = _ranks(data)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    mu = n1 * n2 / 2

    # tie correction of the variance
    N = n1 + n2
    counts: dict[float, int] = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    tie = sum(t**3 - t for t in counts.values())
    var = n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))

    if var <= 0:                       # all values identical -> no difference
        return {"U": u1, "z": 0.0, "p": 1.0}
    z = (u1 - mu - math.copysign(0.5, u1 - mu)) / math.sqrt(var)   # continuity correction
    if abs(u1 - mu) < 0.5:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2))                             # two-sided
    return {"U": min(u1, u2), "z": z, "p": min(1.0, p)}


def compare(name_a: str, a: list[float], name_b: str, b: list[float],
            alpha: float = 0.05) -> dict:
    res = mann_whitney(a, b)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    if res["p"] >= alpha:
        winner = "none"
    else:
        winner = name_a if mean_a > mean_b else name_b
    return {"a": name_a, "b": name_b, "mean_a": mean_a, "mean_b": mean_b,
            **res, "significant": res["p"] < alpha, "winner": winner}
