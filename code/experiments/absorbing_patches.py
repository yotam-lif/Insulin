"""Experiment: flux vs number of perfectly absorbing patches."""
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments.workers import worker_absorbing_patches
from insulin.plotting import plot_flux_vs_patches


def run(rng, max_workers=8):
    D = 10
    R = 7.5
    H = 10.0
    dt = 1e-6
    N = 10000
    a = 5
    patch_radius = 0.1

    L = R - a
    tau_diff = L**2 / (np.pi**2 * D)
    t_relax = 15 * tau_diff
    T = 2 * t_relax

    patch_list = np.arange(100, 1400, 200)
    seeds = rng.integers(0, 2**63 - 1, size=len(patch_list), dtype=np.int64)

    args_list = [
        (int(n), t_relax, D, R, H, dt, T, N, a, patch_radius, int(seed))
        for n, seed in zip(patch_list, seeds)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(worker_absorbing_patches, args_list))

    results.sort(key=lambda t: t[0])
    n_sorted = np.array([r[0] for r in results])
    sim_ratios = np.array([r[1] for r in results])
    analytical = np.array([r[2] for r in results])

    for n, sr, an in zip(n_sorted, sim_ratios, analytical):
        print(f"n_patches={n:4d}  sim={sr:.4f}  analytical={an:.4f}")

    plot_flux_vs_patches(n_sorted, analytical, sim_ratios)


if __name__ == "__main__":
    rng = np.random.default_rng(int(time.time()))
    run(rng)
