"""Common result container returned by every algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunResult:
    algorithm: str
    best_value: int
    best_solution: np.ndarray
    history: list[int] = field(default_factory=list)   # best-so-far per iteration
    evaluations: int = 0                               # number of fitness evaluations
    iterations: int = 0                                # iterations actually executed
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "best_value": int(self.best_value),
            "selected_items": np.flatnonzero(self.best_solution).tolist(),
            "evaluations": int(self.evaluations),
            "iterations": int(self.iterations),
            "stop_reason": self.stop_reason,
        }


class StoppingCriterion:
    """
    Stops when either:
      1. the maximum number of iterations is reached, or
      2. the best fitness has not improved for `patience` iterations.
    """

    def __init__(self, max_iter: int, patience: int):
        self.max_iter = max_iter
        self.patience = patience
        self.best = -np.inf
        self.since_improve = 0
        self.reason = ""

    def update(self, iteration: int, best_value: float) -> bool:
        """Return True if the search must stop."""
        if best_value > self.best:
            self.best = best_value
            self.since_improve = 0
        else:
            self.since_improve += 1
        if iteration >= self.max_iter:
            self.reason = "max_iter"
            return True
        if self.since_improve >= self.patience:
            self.reason = "stagnation"
            return True
        return False
