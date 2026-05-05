"""J/J_max vs patch count for a sweep of internalization rates kc.

Each curve corresponds to a different kc value (release to inner bulk/absorbed).
ka=inf, kb=0 (no dissociation). Also shows the Berg-Purcell limit (kc->inf).

Run: python plotting/kinetics_kc.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from stochastic_sim.workers import worker_kinetics_steady

# --- Experiment parameters ---
D = 10.0
R = 7.5
H = 10.0
dt = 1e-6
a = 5.0
patch_radius = 0.1
C0 = 10.0

L = R - a
tau_diff = (L**2) / (np.pi**2 * D)
t_relax = 30.0 * tau_diff
T = t_relax + 2.0

ka = math.inf
kb = 0.0
kc_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
patch_list = np.arange(100, 6000, 800, dtype=int)
MAX_WORKERS = 8

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rng = np.random.default_rng(int(time.time()))

    sim_flux = np.zeros((len(kc_values), len(patch_list)))
    ana_flux = np.zeros((len(kc_values), len(patch_list)))

    seeds = rng.integers(0, 2**63 - 1,
                         size=(len(kc_values), len(patch_list)), dtype=np.int64)

    for i, kc in enumerate(kc_values):
        args_list = [
            (int(n), t_relax, D, R, H, dt, T, C0, a, patch_radius,
             ka, kb, kc, int(seeds[i, j]))
            for j, n in enumerate(patch_list)
        ]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(ex.map(worker_kinetics_steady, args_list))
        results.sort(key=lambda x: x[0])
        sim_flux[i] = [r[1] for r in results]
        ana_flux[i] = [r[2] for r in results]

    # Berg-Purcell limit (kc -> inf)
    log_ratio = np.log(R / a)
    geom = H * np.pi / (2.0 * log_ratio)
    bp_ana = [(n * patch_radius) / (n * patch_radius + geom) for n in patch_list]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, kc in enumerate(kc_values):
        color = f"C{i}"
        ax.plot(patch_list, sim_flux[i], marker="o", color=color,
                label=f"sim, $k_c={kc:g}$")
        ax.plot(patch_list, ana_flux[i], linestyle="--", color=color,
                label=f"ana, $k_c={kc:g}$")

    ax.plot(patch_list, bp_ana, linestyle="-.", linewidth=2.5,
            color="black", label=r"BP limit ($k_c\to\infty$)")

    ax.text(0.02, 0.02,
            rf"$k_a=\infty$" + "\n" + rf"$k_b={kb:.2g}$",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlabel("Number of patches")
    ax.set_ylabel(r"Normalised flux $J / J_{\max}$")
    ax.set_title(r"Flux vs patch count for different $k_c$")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "kinetics_kc.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
