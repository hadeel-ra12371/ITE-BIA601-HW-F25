"""
Experiment runner.

For every problem size:
  * DP and Greedy are run once (deterministic).
  * GA, HGA, PSO and ACO are run N_RUNS times with different seeds
    (iteration budget taken from config.SIZES, so it grows with n).
After all runs, every pair of metaheuristics is compared with the Wilcoxon
rank-sum (Mann-Whitney U) test.
For every run we record: best value, gap to the DP optimum, wall-clock time,
peak memory (tracemalloc, measured in a separate run so it does not distort
the timing), number of evaluations and the convergence history.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .algorithms import solve_aco, solve_dp, solve_ga, solve_greedy, solve_hga, solve_pso
from .algorithms.base import RunResult
from .problem import KnapsackInstance, generate_instance
from .stats import compare

STOCHASTIC: dict[str, tuple[Callable, dict]] = {
    "GA": (solve_ga, config.GA_PARAMS),
    "PSO": (solve_pso, config.PSO_PARAMS),
    "ACO": (solve_aco, config.ACO_PARAMS),
    "HGA": (solve_hga, config.HGA_PARAMS),
}


def size_params(size: str, params: dict) -> dict:
    """Override the iteration budget with the per-size values from config."""
    s = config.SIZES[size]
    return {**params, "max_iter": s["max_iter"], "patience": s["patience"]}


def build_instances() -> dict[str, KnapsackInstance]:
    return {name: generate_instance(s["n"], s["seed"], name)
            for name, s in config.SIZES.items()}


def _timed(fn: Callable[[], RunResult]) -> tuple[RunResult, float]:
    t0 = time.perf_counter()
    res = fn()
    return res, time.perf_counter() - t0


def _peak_memory_mb(fn: Callable[[], RunResult]) -> float:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def _iter_to_reach(history: list[int], target: float) -> int | None:
    """First iteration whose best-so-far value is >= target (None if never)."""
    for i, h in enumerate(history):
        if h >= target:
            return i
    return None


def run_all(n_runs: int = config.N_RUNS, sizes: list[str] | None = None,
            out_dir: Path = Path("results"), verbose: bool = True) -> dict:
    instances = build_instances()
    sizes = sizes or list(instances)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": {"n_runs": n_runs, "sizes": config.SIZES,
                               "GA": config.GA_PARAMS, "PSO": config.PSO_PARAMS,
                               "ACO": config.ACO_PARAMS, "HGA": config.HGA_PARAMS},
                    "instances": {}}

    for size in sizes:
        inst = instances[size]
        if verbose:
            print(f"\n=== {size.upper()}  (n={inst.n}, budget={inst.capacity}) ===")
        entry: dict = {"n": inst.n, "capacity": inst.capacity,
                       "total_weight": int(inst.weights.sum()), "algorithms": {}}

        # ---- Exact + greedy baselines -----------------------------------
        dp_res, dp_t = _timed(lambda: solve_dp(inst))
        optimum = dp_res.best_value
        entry["optimum"] = optimum
        for name, fn in (("DP", solve_dp), ("Greedy", solve_greedy)):
            res, t = (dp_res, dp_t) if name == "DP" else _timed(lambda: fn(inst))
            mem = _peak_memory_mb(lambda: fn(inst))
            entry["algorithms"][name] = {
                "runs": [{**res.to_dict(), "time": t, "memory_mb": mem,
                          "gap_pct": 100 * (optimum - res.best_value) / optimum}],
            }
            if verbose:
                print(f"  {name:<7} value={res.best_value:<7} time={t:.4f}s  mem={mem:.2f}MB")

        # ---- Metaheuristics ---------------------------------------------
        for name, (solver, params) in STOCHASTIC.items():
            params = size_params(size, params)
            runs = []
            mem = _peak_memory_mb(lambda: solver(inst, seed=config.BASE_SEED, **params))
            for r in range(n_runs):
                seed = config.BASE_SEED + r
                res, t = _timed(lambda: solver(inst, seed=seed, **params))
                runs.append({**res.to_dict(), "seed": seed, "time": t,
                             "memory_mb": mem, "history": res.history,
                             "gap_pct": 100 * (optimum - res.best_value) / optimum,
                             "iter_to_99": _iter_to_reach(res.history, 0.99 * optimum),
                             "iter_to_opt": _iter_to_reach(res.history, optimum)})
            entry["algorithms"][name] = {"runs": runs}
            if verbose:
                vals = [r["best_value"] for r in runs]
                print(f"  {name:<7} mean={np.mean(vals):.1f} std={np.std(vals):.2f} "
                      f"best={max(vals)} time={np.mean([r['time'] for r in runs]):.3f}s")

        # ---- Statistical tests (every pair of metaheuristics) -----------
        names = list(STOCHASTIC)
        entry["tests"] = [
            compare(a, [r["best_value"] for r in entry["algorithms"][a]["runs"]],
                    b, [r["best_value"] for r in entry["algorithms"][b]["runs"]],
                    config.ALPHA)
            for i, a in enumerate(names) for b in names[i + 1:]]
        if verbose:
            for t in entry["tests"]:
                print(f"  test {t['a']:>4} vs {t['b']:<4} p={t['p']:.4f} winner={t['winner']}")

        report["instances"][size] = entry

    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    write_summary(report, out_dir)
    return report


def summarise(report: dict) -> list[dict]:
    """One row per (size, algorithm) with aggregated statistics."""
    rows = []
    for size, entry in report["instances"].items():
        opt = entry["optimum"]
        for alg, data in entry["algorithms"].items():
            runs = data["runs"]
            vals = np.array([r["best_value"] for r in runs], dtype=float)
            times = np.array([r["time"] for r in runs])
            it99 = [r.get("iter_to_99") for r in runs if r.get("iter_to_99") is not None]
            rows.append({
                "size": size, "n": entry["n"], "algorithm": alg, "optimum": opt,
                "best": int(vals.max()), "worst": int(vals.min()),
                "mean": float(vals.mean()), "std": float(vals.std()),
                "mean_gap_pct": float(100 * (opt - vals.mean()) / opt),
                "hit_opt_pct": float(100 * np.mean(vals == opt)),
                "mean_time_s": float(times.mean()),
                "memory_mb": float(runs[0]["memory_mb"]),
                "mean_iters": float(np.mean([r["iterations"] for r in runs])),
                "mean_evals": float(np.mean([r["evaluations"] for r in runs])),
                "mean_iter_to_99": float(np.mean(it99)) if it99 else None,
                "runs": len(runs),
            })
    return rows


def write_summary(report: dict, out_dir: Path) -> None:
    with open(out_dir / "stat_tests.csv", "w", encoding="utf-8") as f:
        f.write("size,alg_a,alg_b,mean_a,mean_b,U,z,p_value,significant,winner\n")
        for size, entry in report["instances"].items():
            for t in entry.get("tests", []):
                f.write(f"{size},{t['a']},{t['b']},{t['mean_a']:.2f},{t['mean_b']:.2f},"
                        f"{t['U']:.1f},{t['z']:.3f},{t['p']:.5f},{t['significant']},{t['winner']}\n")
    rows = summarise(report)
    keys = list(rows[0].keys())
    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else
                             (f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]))
                             for k in keys) + "\n")


def run_sensitivity(out_dir: Path = Path("results"), verbose: bool = True) -> dict:
    """Vary one key hyperparameter per algorithm on the medium instance."""
    inst = build_instances()["medium"]
    optimum = solve_dp(inst).best_value
    out: dict = {"optimum": optimum, "results": {}}
    for alg, (param, values) in config.SENSITIVITY.items():
        solver, base = STOCHASTIC[alg]
        out["results"][alg] = {"param": param, "points": []}
        for val in values:
            params = {**size_params("medium", base), param: val}
            vals = [solver(inst, seed=config.BASE_SEED + r, **params).best_value
                    for r in range(config.SENSITIVITY_RUNS)]
            gap = 100 * (optimum - np.mean(vals)) / optimum
            out["results"][alg]["points"].append(
                {"value": val, "mean": float(np.mean(vals)),
                 "std": float(np.std(vals)), "gap_pct": float(gap)})
            if verbose:
                print(f"  {alg} {param}={val:<8.4g} mean={np.mean(vals):.1f} gap={gap:.3f}%")
    with open(out_dir / "sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def run_dp_scaling(out_dir: Path = Path("results"), factors=(1, 5, 10, 20),
                   verbose: bool = True) -> dict:
    """
    Show the pseudo-polynomial nature of DP: keep n = 600 projects but express
    costs in finer units (cost * factor). The problem is identical, yet DP time
    and memory grow linearly with the budget C, while GA/PSO/ACO are unaffected.
    """
    base = build_instances()["large"]
    out: dict = {"n": base.n, "points": []}
    for f in factors:
        inst = KnapsackInstance(f"large_x{f}", base.values, base.weights * f,
                                base.capacity * f)
        res, t = _timed(lambda: solve_dp(inst))
        mem = _peak_memory_mb(lambda: solve_dp(inst))
        out["points"].append({"factor": f, "capacity": inst.capacity,
                              "time": t, "memory_mb": mem, "value": res.best_value})
        if verbose:
            print(f"  DP cost x{f:<3} C={inst.capacity:<8} time={t:.3f}s mem={mem:.1f}MB")
    with open(out_dir / "dp_scaling.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out
