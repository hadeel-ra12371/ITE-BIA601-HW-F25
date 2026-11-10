"""
Greedy heuristic (fast baseline).

Sort projects by value/cost ratio (best first) and fund each one if it still
fits in the remaining budget. Time O(n log n), no randomness.
"""

from __future__ import annotations

import numpy as np

from ..problem import KnapsackInstance
from .base import RunResult


def solve_greedy(inst: KnapsackInstance) -> RunResult:
    x = np.zeros(inst.n, dtype=bool)
    remaining = inst.capacity
    for i in inst.asc_ratio_order[::-1]:          # highest ratio first
        w = int(inst.weights[i])
        if w <= remaining:
            x[i] = True
            remaining -= w
    best = int(inst.total_value(x))
    return RunResult(algorithm="Greedy", best_value=best, best_solution=x,
                     history=[best], evaluations=1, iterations=1,
                     stop_reason="deterministic")
