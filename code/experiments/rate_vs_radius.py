"""Experiment: validate steady-state flux vs inner cylinder radius."""
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments.workers import worker_rate_vs_radius
from insulin.plotting import plot_rate_vs_radius


def run(rng, max_workers=None):
    D = 100
    R = 7.5
    H = 10.0
    dt = 1e-6
    T = 1
    N = 10000

    a_values = np.linspace(1.0, 5.0, 7)
    seeds = rng.integers(0, 2**63 - 1, size=len(a_values), dtype=np.int64)
    args_list = [
        (float(a), D, R, H, dt, T, N, int(seed))
        for a, seed in zip(a_values, seeds)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(worker_rate_vs_radius, args_list))

    results.sort(key=lambda t: t[0])
    a_sorted = np.array([r[0] for r in results])
    sim_rates = np.array([r[1] for r in results])
    theory_rates = np.array([r[2] for r in results])

    for a, k_sim in zip(a_sorted, sim_rates):
        print(f"a = {a:.3f}  k_sim = {k_sim:.3e} um^3/s")

    plot_rate_vs_radius(a_sorted, theory_rates, sim_rates)


if __name__ == "__main__":
    rng = np.random.default_rng(int(time.time()))
    run(rng)
