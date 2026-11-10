"""
Entry point: run all experiments, save tables and figures.

Examples
--------
    python main.py                       # full experiment (10 runs, all sizes)
    python main.py --runs 3 --sizes small medium
    python main.py --no-sensitivity --no-scaling
    python main.py --demo medium         # single run of every algorithm, quick check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config
from src.algorithms import solve_aco, solve_dp, solve_ga, solve_greedy, solve_hga, solve_pso
from src.experiments import (build_instances, run_all, run_dp_scaling,
                             run_sensitivity, summarise)
from src.plotting import make_all

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def demo(size: str) -> None:
    """Run every algorithm once on one instance and print the chosen projects."""
    inst = build_instances()[size]
    budget = {k: config.SIZES[size][k] for k in ("max_iter", "patience")}
    print(f"Instance '{size}': n={inst.n}, budget={inst.capacity}")
    for res in (solve_dp(inst), solve_greedy(inst),
                solve_ga(inst, seed=config.BASE_SEED, **{**config.GA_PARAMS, **budget}),
                solve_pso(inst, seed=config.BASE_SEED, **{**config.PSO_PARAMS, **budget}),
                solve_aco(inst, seed=config.BASE_SEED, **{**config.ACO_PARAMS, **budget}),
                solve_hga(inst, seed=config.BASE_SEED, **{**config.HGA_PARAMS, **budget})):
        cost = int(inst.total_weight(res.best_solution))
        print(f"  {res.algorithm:<7} return={res.best_value:<7} cost={cost:<6} "
              f"projects={int(res.best_solution.sum()):<4} feasible={inst.is_feasible(res.best_solution)}")


def print_table(report: dict) -> None:
    print("\n" + "-" * 96)
    print(f"{'size':<7}{'alg':<8}{'best':>8}{'mean':>10}{'std':>8}{'gap%':>8}"
          f"{'hit%':>7}{'time(s)':>10}{'mem(MB)':>9}{'it->99%':>9}")
    for r in summarise(report):
        it99 = "-" if r["mean_iter_to_99"] is None else f"{r['mean_iter_to_99']:.1f}"
        print(f"{r['size']:<7}{r['algorithm']:<8}{r['best']:>8}{r['mean']:>10.1f}"
              f"{r['std']:>8.2f}{r['mean_gap_pct']:>8.3f}{r['hit_opt_pct']:>7.0f}"
              f"{r['mean_time_s']:>10.4f}{r['memory_mb']:>9.2f}{it99:>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Project portfolio selection: DP vs Greedy vs GA vs PSO vs ACO")
    ap.add_argument("--runs", type=int, default=config.N_RUNS, help="repetitions per stochastic algorithm")
    ap.add_argument("--sizes", nargs="+", choices=list(config.SIZES), default=None)
    ap.add_argument("--no-sensitivity", action="store_true", help="skip hyperparameter study")
    ap.add_argument("--no-scaling", action="store_true", help="skip DP budget-scaling study")
    ap.add_argument("--demo", choices=list(config.SIZES), help="single quick run on one size")
    args = ap.parse_args()

    if args.demo:
        demo(args.demo)
        return

    report = run_all(n_runs=args.runs, sizes=args.sizes, out_dir=RESULTS)
    print_table(report)

    sens = None
    if not args.no_sensitivity:
        print("\n=== Hyperparameter sensitivity (medium instance) ===")
        sens = run_sensitivity(out_dir=RESULTS)

    scal = None
    if not args.no_scaling:
        print("\n=== DP scaling with budget size ===")
        scal = run_dp_scaling(out_dir=RESULTS)

    make_all(report, sens, RESULTS / "figures", scal)
    print(f"\nSaved: {RESULTS / 'summary.csv'}, {RESULTS / 'raw_results.json'}, {RESULTS / 'figures'}")


if __name__ == "__main__":
    main()
