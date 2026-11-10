# Project Portfolio Selection — GA vs PSO vs ACO vs DP

Course project (Intelligent Algorithms, F25). We pick a realistic optimisation
problem, implement several algorithms **from scratch** (NumPy only, no
optimisation libraries), and compare them on three problem sizes.

## Problem

A company must choose which candidate projects to fund. Project *i* costs
`w_i` and returns `v_i`, and the total budget is `C`. Maximise the total return
without exceeding the budget. This is the 0/1 Knapsack problem (NP-hard):

```
maximize    Σ v_i·x_i
subject to  Σ w_i·x_i ≤ C ,   x_i ∈ {0,1}
```

Instances are *strongly correlated* (`w_i ~ U{10..100}`, `v_i = w_i + 10`,
`C = 0.5·Σw`), one of the hardest knapsack classes. Every size has a fixed seed.

| Size   | n   | Budget C | Optimum (DP) |
|--------|-----|----------|--------------|
| small  | 30  | 868      | 1,063        |
| medium | 150 | 3,751    | 4,791        |
| large  | 600 | 16,682   | 20,742       |

## Algorithms

| Algorithm | File | Role |
|-----------|------|------|
| Dynamic Programming | `src/algorithms/dp.py` | exact baseline, O(n·C) |
| Greedy (value/cost ratio) | `src/algorithms/greedy.py` | fast baseline |
| Genetic Algorithm | `src/algorithms/ga.py` | tournament selection, uniform crossover, bit-flip mutation, elitism |
| Binary PSO | `src/algorithms/pso.py` | sigmoid transfer, linearly decreasing inertia |
| Ant Colony (MAX-MIN) | `src/algorithms/aco.py` | item pheromone + ratio heuristic, vectorised construction |
| Hybrid / Memetic GA | `src/algorithms/hga.py` | GA + greedy seeding + greedy fill + 1-1 swap local search |

All three metaheuristics share:
- **Representation**: binary vector `x` of length n.
- **Fitness**: total return after the **repair** operator (drops the lowest-ratio
  selected projects until the solution fits the budget) — see `src/problem.py`.
- **Budget**: 50 individuals; the iteration limit grows with the size
  (small 300, medium 500, large 1000), identical for all metaheuristics.
- **Stopping criteria**: `max_iter` or `patience` (= 40 % of `max_iter`) iterations without improvement.

All hyperparameters are in [`src/config.py`](src/config.py).

## Project structure

```
optimization_project/
├── main.py                 # entry point
├── requirements.txt
├── src/
│   ├── config.py           # sizes, seeds, hyperparameters
│   ├── problem.py          # instance generator, objective, repair
│   ├── experiments.py      # repeated runs, timing, memory, statistics
│   ├── plotting.py         # all figures
│   ├── stats.py            # Wilcoxon rank-sum (Mann-Whitney U) test
│   └── algorithms/         # base.py dp.py greedy.py ga.py hga.py pso.py aco.py
└── results/
    ├── summary.csv         # aggregated table (one row per size × algorithm)
    ├── stat_tests.csv      # pairwise Wilcoxon rank-sum tests
    ├── raw_results.json    # every run incl. convergence history
    ├── sensitivity.json    # hyperparameter study
    ├── dp_scaling.json     # DP time/memory vs. budget size
    ├── console_log.txt
    └── figures/*.png
```

## How to run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python main.py                          # full experiment: 10 runs × 3 sizes + studies (~1 min)
python main.py --demo large             # one quick run of every algorithm on one size
python main.py --runs 3 --sizes small medium
python main.py --no-sensitivity --no-scaling
```

The full run writes the tables to `results/` and the figures to
`results/figures/`. Seeds are fixed (`BASE_SEED = 2025`, runs use seeds
2025–2034), so solution quality is reproduced exactly. Run times and memory
depend on the machine and change slightly between runs.

## Measured metrics

- Solution quality: best / mean / worst / std, gap to the DP optimum, % of runs that hit the optimum
- Run time (`time.perf_counter`) and peak memory (`tracemalloc`, measured in a separate run)
- Convergence: Iteration → Best Fitness curves, iterations needed to reach 99 % of the optimum
- Stability: box plots over the 10 runs
- Statistical significance: pairwise Wilcoxon rank-sum test (α = 0.05), no SciPy needed
- Hyperparameter sensitivity (medium instance) and DP scaling with budget size

## Team

| Name | Username | Class | Responsibility |
|------|----------|-------|----------------|
| أمل عواد | amal_153350 | C4 | problem model, objective, repair (`problem.py`) |
| هديل القصير | hadeel_194838 | C6 | DP and Greedy baselines (`dp.py`, `greedy.py`) |
| لانا الرفاعي | lana_198154 | C6 | Genetic Algorithm and Hybrid GA (`ga.py`, `hga.py`) |
| محمد سالمة | mohammad_188463 | C6 | Binary PSO (`pso.py`) |
| محمد بركات | mohamad_202768 | C6 | Ant Colony Optimization (`aco.py`) |
| عبد العليم الشيخ | abdil_aliem_196295 | C2 | experiment runner, timing, memory, statistics (`experiments.py`, `base.py`, `stats.py`) |
| عطاف عثمان | Etaf_154254 | C4 | figures and analysis (`plotting.py`) |
| منى جندلي | mona_187366 | C1 | configuration, sensitivity study, documentation (`config.py`, README) |
| نغم الشامي | nagham_191378| C7 | Report writing, formatting, and results verification |
The full Arabic report is in `../report/`.

## Key results (mean gap to the optimum over 10 runs)

| Size | Greedy | GA | PSO | ACO | HGA |
|------|--------|----|-----|-----|-----|
| small  | 0.000 % | 0.047 % | 0.000 % | 0.282 % | 0.000 % |
| medium | 0.480 % | 0.148 % | 2.041 % | 0.125 % | 0.000 % |
| large  | 0.275 % | 0.213 % | 3.024 % | 0.288 % | 0.000 % |

