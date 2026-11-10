"""
Genetic Algorithm (GA) for 0/1 knapsack.

Representation : binary chromosome of length n (gene i = project i funded?)
Fitness        : total value after repair (always feasible)
Loop           :
    1. Tournament selection (size k) picks parents.
    2. Uniform crossover with probability pc.
    3. Bit-flip mutation, each gene flipped with probability pm (default 1/n).
    4. Repair offspring (drop lowest-ratio projects until within budget).
    5. Elitism: the best `elite` individuals survive unchanged.
Stopping       : max_iter generations or `patience` generations w/o improvement.
"""

from __future__ import annotations

import numpy as np

from ..problem import KnapsackInstance
from .base import RunResult, StoppingCriterion


def _tournament(fit: np.ndarray, k: int, n_select: int,
                rng: np.random.Generator) -> np.ndarray:
    """Return indices of `n_select` winners of size-k tournaments."""
    contestants = rng.integers(0, fit.size, size=(n_select, k))
    winners = np.argmax(fit[contestants], axis=1)
    return contestants[np.arange(n_select), winners]


def solve_ga(inst: KnapsackInstance, seed: int, pop_size: int = 50,
             max_iter: int = 300, patience: int = 120,
             crossover_rate: float = 0.9, mutation_rate: float | None = None,
             tournament_k: int = 3, elite: int = 2) -> RunResult:
    rng = np.random.default_rng(seed)
    n = inst.n
    pm = (1.0 / n) if mutation_rate is None else mutation_rate

    # ---- Initial population: random bits, then repair --------------------
    pop = inst.repair(rng.random((pop_size, n)) < 0.5)
    fit = inst.fitness(pop)
    evals = pop_size

    best_idx = int(np.argmax(fit))
    best_x, best_f = pop[best_idx].copy(), int(fit[best_idx])
    history = [best_f]
    stop = StoppingCriterion(max_iter, patience)
    stop.update(0, best_f)

    it = 0
    while True:
        it += 1
        n_children = pop_size - elite

        # ---- Selection ----------------------------------------------------
        p1 = pop[_tournament(fit, tournament_k, n_children, rng)]
        p2 = pop[_tournament(fit, tournament_k, n_children, rng)]

        # ---- Uniform crossover -------------------------------------------
        do_cx = rng.random(n_children) < crossover_rate
        mask = rng.random((n_children, n)) < 0.5
        children = np.where(do_cx[:, None] & mask, p2, p1)

        # ---- Bit-flip mutation ------------------------------------------
        flips = rng.random((n_children, n)) < pm
        children = children ^ flips

        # ---- Repair + evaluation ----------------------------------------
        children = inst.repair(children)
        child_fit = inst.fitness(children)
        evals += n_children

        # ---- Elitism + replacement --------------------------------------
        elite_idx = np.argsort(fit)[-elite:]
        pop = np.vstack([pop[elite_idx], children])
        fit = np.concatenate([fit[elite_idx], child_fit])

        gen_best = int(np.argmax(fit))
        if fit[gen_best] > best_f:
            best_f, best_x = int(fit[gen_best]), pop[gen_best].copy()
        history.append(best_f)

        if stop.update(it, best_f):
            break

    return RunResult("GA", best_f, best_x, history, evals, it, stop.reason)
