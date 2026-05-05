"""J/J_max vs patch count for perfectly absorbing patches (Berg-Purcell limit).

Compares simulation with the Berg-Purcell formula:
    mu = N*s / (N*s + pi*H / (2*ln(R/a)))

Run: python plotting/absorbing_patches.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from stochastic_sim.workers import worker_absorbing_patches

# --- Experiment parameters ---
D = 10.0
R = 7.5
H = 10.0
dt = 1e-6
N = 10000
a = 5.0
patch_radius = 0.1

L = R - a
tau_diff = (L**2) / (np.pi**2 * D)
t_relax = 15.0 * tau_diff
T = 2.0 * t_relax

patch_list = np.arange(100, 1400, 200, dtype=int)
MAX_WORKERS = 8

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rng = np.random.default_rng(int(time.time()))
    seeds = rng.integers(0, 2**63 - 1, size=len(patch_list), dtype=np.int64)

    args_list = [
        (int(n), t_relax, D, R, H, dt, T, N, a, patch_radius, int(s))
        for n, s in zip(patch_list, seeds)
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(worker_absorbing_patches, args_list))

    results.sort(key=lambda tup: tup[0])
    n_sorted     = np.array([r[0] for r in results])
    sim_ratios   = np.array([r[1] for r in results])
    ana_ratios   = np.array([r[2] for r in results])
    coverages    = np.array([r[3] for r in results])

    for n, sim, ana, cov in zip(n_sorted, sim_ratios, ana_ratios, coverages):
        print(f"  n={n:5d}  sim={sim:.4f}  analytical={ana:.4f}  coverage={cov:.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_sorted, ana_ratios, linewidth=2, label="Analytical (Berg-Purcell)")
    ax.plot(n_sorted, sim_ratios, "o--", markersize=6, label="Simulation")
    ax.set_xlabel("Number of patches")
    ax.set_ylabel(r"$J / J_{\max}$")
    ax.set_title("Flux vs patch count: perfectly absorbing patches")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "absorbing_patches.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
