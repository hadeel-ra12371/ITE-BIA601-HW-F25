"""
Dynamic Programming (exact baseline).

Recurrence (classic 0/1 knapsack):
    dp[i][c] = max( dp[i-1][c],  dp[i-1][c - w_i] + v_i )   if w_i <= c
Time  : O(n * C)
Memory: O(n * C) bits for the decision table (needed to rebuild the solution)

The row update is vectorised with NumPy: the candidate vector is computed
from the previous row *before* writing, so a single 1-D array is enough
for the values while a boolean table stores the take/skip decisions.
"""

from __future__ import annotations

import numpy as np

from ..problem import KnapsackInstance
from .base import RunResult


def solve_dp(inst: KnapsackInstance) -> RunResult:
    n, C = inst.n, inst.capacity
    dp = np.zeros(C + 1, dtype=np.int64)          # dp[c] = best value with budget c
    keep = np.zeros((n, C + 1), dtype=bool)       # keep[i, c] = item i taken at budget c

    for i in range(n):
        w, v = int(inst.weights[i]), int(inst.values[i])
        if w > C:
            continue
        candidate = dp[: C + 1 - w] + v           # value if item i is taken
        better = candidate > dp[w:]
        keep[i, w:] = better
        dp[w:] = np.where(better, candidate, dp[w:])

    # Backtracking: rebuild the optimal subset
    x = np.zeros(n, dtype=bool)
    c = C
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            x[i] = True
            c -= int(inst.weights[i])

    best = int(dp[C])
    return RunResult(algorithm="DP", best_value=best, best_solution=x,
                     history=[best], evaluations=n * (C + 1), iterations=n,
                     stop_reason="exact")
