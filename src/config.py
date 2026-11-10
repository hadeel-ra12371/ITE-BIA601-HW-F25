"""
Central configuration: problem sizes and algorithm hyperparameters.

Every experiment reads its settings from here, so results can be reproduced
exactly by re-running `python main.py`.
"""

# --------------------------------------------------------------------------- #
# Problem sizes (each size uses a fixed seed -> identical instance every run)
# --------------------------------------------------------------------------- #
# The iteration budget grows with the problem size (bigger search space ->
# more iterations). `patience` is kept at 40 % of `max_iter`.
SIZES = {
    "small":  {"n": 30,  "seed": 101, "max_iter": 300,  "patience": 120},
    "medium": {"n": 150, "seed": 202, "max_iter": 500,  "patience": 200},
    "large":  {"n": 600, "seed": 303, "max_iter": 1000, "patience": 400},
}

# Number of independent repetitions of every stochastic algorithm
N_RUNS = 10
BASE_SEED = 2025

# --------------------------------------------------------------------------- #
# Shared budget -> fair comparison (same population size & iteration limit,
# i.e. the same maximum number of fitness evaluations for every metaheuristic).
# max_iter / patience below are defaults; experiments override them per size.
# --------------------------------------------------------------------------- #
COMMON = {
    "pop_size": 50,          # chromosomes / particles / ants
    "max_iter": 300,         # stopping criterion 1: iteration limit
    "patience": 120,         # stopping criterion 2: stop after N iterations w/o improvement
}

GA_PARAMS = {
    **COMMON,
    "crossover_rate": 0.9,   # probability of applying uniform crossover
    "mutation_rate": None,   # None -> 1/n (one flipped bit per child on average)
    "tournament_k": 3,       # tournament selection size
    "elite": 2,              # number of best individuals copied unchanged
}

# Hybrid (memetic) GA: same GA parameters + greedy seeding + local search
HGA_PARAMS = dict(GA_PARAMS)

PSO_PARAMS = {
    **COMMON,
    "w_max": 0.9,            # inertia weight at the first iteration
    "w_min": 0.4,            # inertia weight at the last iteration (linear decay)
    "c1": 2.0,               # cognitive coefficient (pull to personal best)
    "c2": 2.0,               # social coefficient (pull to global best)
    "v_max": 4.0,            # velocity clamp -> sigmoid(4) ~ 0.98
}

ACO_PARAMS = {
    **COMMON,
    "alpha": 1.0,            # pheromone importance
    "beta": 2.0,             # heuristic (value/cost ratio) importance
    "rho": 0.1,              # evaporation rate
    "tau_min": 0.01,         # MAX-MIN pheromone bounds
    "tau_max": 1.0,
    "gb_weight": 0.5,        # share of deposit coming from the global-best ant
}

# Hyperparameter sensitivity study (run on the medium instance)
SENSITIVITY = {
    "GA":  ("mutation_rate", [0.002, 1 / 150, 0.02, 0.05]),
    "PSO": ("v_max", [2.0, 4.0, 6.0]),
    "ACO": ("beta", [0.0, 1.0, 2.0, 4.0]),
}
SENSITIVITY_RUNS = 5

# Significance level of the Wilcoxon rank-sum (Mann-Whitney U) test
ALPHA = 0.05
