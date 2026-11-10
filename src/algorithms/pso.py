"""
Binary Particle Swarm Optimization (BPSO, Kennedy & Eberhart 1997).

Representation : each particle has a real velocity vector v (length n) and a
                 binary position x (length n).
Update rules   :
    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)       (clamped to +-v_max)
    x_i = 1  with probability  sigmoid(v_i)
    x  <- repair(x)
Inertia weight : linearly decreases from w_max to w_min (exploration -> exploitation)
Stopping       : max_iter iterations or `patience` iterations w/o improvement.
"""

from __future__ import annotations

import numpy as np

from ..problem import KnapsackInstance
from .base import RunResult, StoppingCriterion


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def solve_pso(inst: KnapsackInstance, seed: int, pop_size: int = 50,
              max_iter: int = 300, patience: int = 120,
              w_max: float = 0.9, w_min: float = 0.4,
              c1: float = 2.0, c2: float = 2.0, v_max: float = 4.0) -> RunResult:
    rng = np.random.default_rng(seed)
    n = inst.n

    # ---- Initial swarm ---------------------------------------------------
    x = inst.repair(rng.random((pop_size, n)) < 0.5)
    v = rng.uniform(-v_max, v_max, size=(pop_size, n))
    fit = inst.fitness(x)
    evals = pop_size

    pbest, pbest_fit = x.copy(), fit.copy()
    g = int(np.argmax(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), int(pbest_fit[g])
    history = [gbest_fit]
    stop = StoppingCriterion(max_iter, patience)
    stop.update(0, gbest_fit)

    it = 0
    while True:
        it += 1
        w = w_max - (w_max - w_min) * (it - 1) / max(max_iter - 1, 1)

        # ---- Velocity update --------------------------------------------
        xf = x.astype(float)
        r1 = rng.random((pop_size, n))
        r2 = rng.random((pop_size, n))
        v = (w * v
             + c1 * r1 * (pbest.astype(float) - xf)
             + c2 * r2 * (gbest.astype(float) - xf))
        np.clip(v, -v_max, v_max, out=v)

        # ---- Position update (sigmoid transfer function) ---------------
        x = rng.random((pop_size, n)) < _sigmoid(v)
        x = inst.repair(x)
        fit = inst.fitness(x)
        evals += pop_size

        # ---- Personal & global best update -----------------------------
        improved = fit > pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g = int(np.argmax(pbest_fit))
        if pbest_fit[g] > gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), int(pbest_fit[g])
        history.append(gbest_fit)

        if stop.update(it, gbest_fit):
            break

    return RunResult("PSO", gbest_fit, gbest, history, evals, it, stop.reason)
