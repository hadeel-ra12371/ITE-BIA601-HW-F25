"""
Hybrid (Memetic) Genetic Algorithm = GA + problem knowledge + local search.

Same representation, fitness, selection, crossover and mutation as `ga.py`,
plus three additions:
  1. Greedy seeding  : one individual of the initial population is the
                       Greedy (value/cost ratio) solution.
  2. Greedy fill     : after repair, unused budget is filled with the best-ratio
                       projects that still fit.
  3. Local search    : every generation the best child is improved with
                       "1-1 swap" moves (remove one funded project, add one
                       unfunded project) until no improving swap exists.
Steps 2-3 are applied to the best child only (Lamarckian: the improved
solution replaces the child), which keeps the extra cost per generation small.
"""

from __future__ import annotations

import numpy as np

from ..problem import KnapsackInstance
from .base import RunResult, StoppingCriterion
from .ga import _tournament
from .greedy import solve_greedy


def greedy_fill(inst: KnapsackInstance, x: np.ndarray) -> np.ndarray:
    """Add the best-ratio unselected projects that still fit (single solution)."""
    x = x.copy()
    remaining = inst.capacity - int(inst.total_weight(x))
    for i in inst.asc_ratio_order[::-1]:
        if not x[i] and inst.weights[i] <= remaining:
            x[i] = True
            remaining -= int(inst.weights[i])
    return x


def local_search(inst: KnapsackInstance, x: np.ndarray, max_moves: int = 100) -> tuple[np.ndarray, int]:
    """
    Best-improvement 1-1 swap local search.
    A swap (out i, in j) is feasible if w_j - w_i <= slack, and improving if v_j > v_i.
    Returns the improved solution and the number of evaluated neighbourhoods.
    """
    x = greedy_fill(inst, x)
    w, v = inst.weights, inst.values
    moves = 0
    for moves in range(1, max_moves + 1):
        sel, uns = np.flatnonzero(x), np.flatnonzero(~x)
        if sel.size == 0 or uns.size == 0:
            break
        slack = inst.capacity - int(w[sel].sum())
        dw = w[uns][None, :] - w[sel][:, None]          # weight change of every swap
        dv = v[uns][None, :] - v[sel][:, None]          # value change of every swap
        gain = np.where(dw <= slack, dv, 0)
        k = int(np.argmax(gain))
        if gain.flat[k] <= 0:
            break                                        # local optimum reached
        a, b = divmod(k, uns.size)
        x[sel[a]], x[uns[b]] = False, True
        x = greedy_fill(inst, x)
    return x, moves


def solve_hga(inst: KnapsackInstance, seed: int, pop_size: int = 50,
              max_iter: int = 300, patience: int = 120,
              crossover_rate: float = 0.9, mutation_rate: float | None = None,
              tournament_k: int = 3, elite: int = 2) -> RunResult:
    rng = np.random.default_rng(seed)
    n = inst.n
    pm = (1.0 / n) if mutation_rate is None else mutation_rate

    # ---- Initial population: random + one greedy-seeded individual -------
    pop = inst.repair(rng.random((pop_size, n)) < 0.5)
    pop[0] = solve_greedy(inst).best_solution
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

        # ---- Standard GA step (identical to ga.py) ----------------------
        p1 = pop[_tournament(fit, tournament_k, n_children, rng)]
        p2 = pop[_tournament(fit, tournament_k, n_children, rng)]
        do_cx = rng.random(n_children) < crossover_rate
        mask = rng.random((n_children, n)) < 0.5
        children = np.where(do_cx[:, None] & mask, p2, p1)
        children = children ^ (rng.random((n_children, n)) < pm)
        children = inst.repair(children)
        child_fit = inst.fitness(children)
        evals += n_children

        # ---- Memetic step: improve the best child with local search -----
        c = int(np.argmax(child_fit))
        children[c], moves = local_search(inst, children[c])
        child_fit[c] = int(inst.total_value(children[c]))
        evals += moves

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

    return RunResult("HGA", best_f, best_x, history, evals, it, stop.reason)
