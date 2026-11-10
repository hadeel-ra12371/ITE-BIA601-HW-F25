"""
Ant Colony Optimization (MAX-MIN Ant System, hyper-cube framework) for 0/1 knapsack.

Representation : pheromone tau_i on every project (desirability of funding it)
                 + heuristic eta_i = (v_i / w_i) normalised to [0, 1].
Construction   : each ant repeatedly picks a not-yet-chosen project that still
                 fits in the remaining budget with probability
                     p_i  ~  tau_i^alpha * eta_i^beta
                 until no project fits -> solutions are feasible by construction.
Pheromone      :
    tau <- (1 - rho) * tau + rho * [ (1-g) * 1(i in iteration-best) + g * 1(i in global-best) ]
    tau <- clip(tau, tau_min, tau_max)          (MAX-MIN bounds avoid stagnation)
Stopping       : max_iter iterations or `patience` iterations w/o improvement.

Implementation note
-------------------
Sequential roulette-wheel sampling without replacement is equivalent to
sorting items by  log(p_i) + Gumbel noise  (Plackett-Luce / Gumbel-max trick).
An item skipped because it does not fit can never fit later (the budget only
shrinks), so "sort, then add every item that still fits" produces exactly the
same distribution as the textbook construction, but is vectorised over all ants.
"""

from __future__ import annotations

import numpy as np

from ..problem import KnapsackInstance
from .base import RunResult, StoppingCriterion


def _construct(inst: KnapsackInstance, log_p: np.ndarray, n_ants: int,
               rng: np.random.Generator) -> np.ndarray:
    """Build `n_ants` feasible solutions at once."""
    n = inst.n
    gumbel = -np.log(-np.log(rng.random((n_ants, n))))
    order = np.argsort(-(log_p[None, :] + gumbel), axis=1)   # visiting order per ant
    w_order = inst.weights[order]                              # (ants, n)

    remaining = np.full(n_ants, inst.capacity, dtype=np.int64)
    take = np.zeros((n_ants, n), dtype=bool)
    rows = np.arange(n_ants)
    for j in range(n):                                         # vectorised over ants
        fits = w_order[:, j] <= remaining
        take[rows, order[:, j]] = fits
        remaining -= np.where(fits, w_order[:, j], 0)
    return take


def solve_aco(inst: KnapsackInstance, seed: int, pop_size: int = 50,
              max_iter: int = 300, patience: int = 120,
              alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
              tau_min: float = 0.01, tau_max: float = 1.0,
              gb_weight: float = 0.5) -> RunResult:
    rng = np.random.default_rng(seed)
    n = inst.n

    eta = inst.ratio / inst.ratio.max()                       # heuristic information
    tau = np.full(n, 0.5 * (tau_min + tau_max))               # neutral initial pheromone

    best_x, best_f = np.zeros(n, dtype=bool), -1
    history: list[int] = []
    stop = StoppingCriterion(max_iter, patience)
    evals = 0

    it = 0
    while True:
        # ---- Solution construction --------------------------------------
        log_p = alpha * np.log(tau) + beta * np.log(eta)
        ants = _construct(inst, log_p, pop_size, rng)
        fit = inst.fitness(ants)
        evals += pop_size

        ib = int(np.argmax(fit))
        if fit[ib] > best_f:
            best_f, best_x = int(fit[ib]), ants[ib].copy()
        history.append(best_f)

        # ---- Pheromone evaporation + deposit ----------------------------
        deposit = (1 - gb_weight) * ants[ib] + gb_weight * best_x
        tau = (1 - rho) * tau + rho * deposit
        np.clip(tau, tau_min, tau_max, out=tau)

        if stop.update(it, best_f):
            break
        it += 1

    return RunResult("ACO", best_f, best_x, history, evals, it, stop.reason)
