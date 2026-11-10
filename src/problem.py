"""
Problem definition: Budget-Constrained Project Portfolio Selection.

A company has `n` candidate projects. Project i costs w_i (budget units) and is
expected to return v_i (profit units). The total available budget is C.
Goal: choose a subset of projects that maximises total expected return while
the total cost does not exceed the budget.

This is the classic 0/1 Knapsack Problem (NP-hard):

    maximise    f(x) = sum_i v_i * x_i
    subject to  sum_i w_i * x_i <= C,     x_i in {0, 1}

Solution representation (shared by GA / PSO / ACO):
    a binary vector x of length n, where x[i] = 1 means "project i is funded".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KnapsackInstance:
    """A single problem instance (projects + budget)."""

    name: str
    values: np.ndarray        # expected return of every project, shape (n,)
    weights: np.ndarray       # cost of every project, shape (n,)
    capacity: int             # total budget C
    seed: int = 0
    # Value/weight ratio, pre-computed once because every algorithm uses it
    ratio: np.ndarray = field(init=False, repr=False)
    # Item indices sorted by ascending ratio (used by the repair operator)
    asc_ratio_order: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=np.int64)
        self.ratio = self.values / self.weights
        self.asc_ratio_order = np.argsort(self.ratio, kind="stable")

    @property
    def n(self) -> int:
        return int(self.values.size)

    # ------------------------------------------------------------------ #
    # Objective function
    # ------------------------------------------------------------------ #
    def total_value(self, x: np.ndarray) -> np.ndarray:
        """Objective f(x) for a single solution (1-D) or a population (2-D)."""
        return x.astype(np.int64) @ self.values

    def total_weight(self, x: np.ndarray) -> np.ndarray:
        """Total cost for a single solution (1-D) or a population (2-D)."""
        return x.astype(np.int64) @ self.weights

    def is_feasible(self, x: np.ndarray) -> bool:
        return bool(self.total_weight(x) <= self.capacity)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """
        Fitness used by the metaheuristics.

        All metaheuristics call `repair()` first, so every evaluated solution
        is feasible and fitness == objective value. As a safety net, an
        infeasible solution would receive fitness 0 (death penalty).
        """
        val = self.total_value(x)
        feasible = self.total_weight(x) <= self.capacity
        return np.where(feasible, val, 0)

    # ------------------------------------------------------------------ #
    # Repair operator (constraint handling)
    # ------------------------------------------------------------------ #
    def repair(self, pop: np.ndarray) -> np.ndarray:
        """
        Make every solution feasible by dropping the selected projects with
        the LOWEST value/cost ratio until the budget is respected.

        Works on a population matrix (k, n) in a fully vectorised way and
        returns a new boolean matrix. Feasible rows are left unchanged.
        """
        pop = np.atleast_2d(pop).astype(bool)
        excess = self.total_weight(pop) - self.capacity           # (k,)
        bad = excess > 0
        if not bad.any():
            return pop.copy()

        order = self.asc_ratio_order
        sub = pop[bad][:, order]                                  # worst ratio first
        w_sel = sub * self.weights[order]                         # weight of selected items
        removed_before = np.cumsum(w_sel, axis=1) - w_sel         # weight already removed
        drop = sub & (removed_before < excess[bad][:, None])      # drop until excess covered
        sub = sub & ~drop

        fixed = pop.copy()
        rows = fixed[bad]
        rows[:, order] = sub
        fixed[bad] = rows
        return fixed


def generate_instance(n: int, seed: int, name: str,
                      capacity_ratio: float = 0.5) -> KnapsackInstance:
    """
    Generate a reproducible "strongly correlated" instance (Pisinger, 2005).

    * cost   w_i ~ U{10..100}
    * return v_i = w_i + 10          (every project returns its cost + fixed margin)
    * budget C   = capacity_ratio * sum(w)

    Strongly correlated instances are among the hardest knapsack classes:
    all value/cost ratios are close to each other, so the greedy rule is not
    optimal and good solutions must fill the budget almost exactly.
    """
    rng = np.random.default_rng(seed)
    weights = rng.integers(10, 101, size=n)
    values = weights + 10
    capacity = int(capacity_ratio * weights.sum())
    return KnapsackInstance(name=name, values=values, weights=weights,
                            capacity=capacity, seed=seed)
