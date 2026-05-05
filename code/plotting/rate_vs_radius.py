"""Validate the fully-absorbing rate formula k = 2*pi*H*D / ln(R/a).

Sweeps inner radius a and compares the simulated rate constant k_sim
against the analytical prediction k_theory.

Run: python plotting/rate_vs_radius.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from stochastic_sim.workers import worker_rate_vs_radius

# --- Experiment parameters ---
D = 100.0
R = 7.5
H = 10.0
dt = 1e-6
N = 10000
a_values = np.linspace(1.0, 5.0, 7)
MAX_WORKERS = 8

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rng = np.random.default_rng(int(time.time()))
    seeds = rng.integers(0, 2**63 - 1, size=len(a_values), dtype=np.int64)

    args_list = [(float(a), D, R, H, dt, 0.0, N, int(s))
                 for a, s in zip(a_values, seeds)]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(worker_rate_vs_radius, args_list))

    results.sort(key=lambda tup: tup[0])
    a_sorted      = np.array([r[0] for r in results])
    sim_rates     = np.array([r[1] for r in results])
    theory_rates  = np.array([r[2] for r in results])

    for a, k in zip(a_sorted, sim_rates):
        print(f"  a={a:.3f} µm  k_sim={k:.3e} µm³/s")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a_sorted, theory_rates, linewidth=2,
            label=r"Theory $2\pi H D / \ln(R/a)$")
    ax.plot(a_sorted, sim_rates, "o--", label="Simulation")
    ax.set_xlabel("Inner radius $a$ (µm)")
    ax.set_ylabel("Rate constant $k$ (µm³/s)")
    ax.set_title("Steady-state rate vs inner radius")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "rate_vs_radius.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
