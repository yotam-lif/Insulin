"""J/J_max vs patch count for a sweep of association rates ka.

Each curve corresponds to a different ka value (on-rate to receptor).
kb and kc are fixed. Finite ka < inf reduces effective binding via Erban-Chapman.

Run: python plotting/kinetics_ka.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

kb = 5.0
kc = 5.0
ka_values = np.array([0.1, 1.0, 2.0, 100.0])
patch_list = np.arange(100, 6000, 800, dtype=int)
MAX_WORKERS = 8

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rng = np.random.default_rng(int(time.time()))

    sim_flux = np.zeros((len(ka_values), len(patch_list)))
    ana_flux = np.zeros((len(ka_values), len(patch_list)))

    seeds = rng.integers(0, 2**63 - 1,
                         size=(len(ka_values), len(patch_list)), dtype=np.int64)

    for i, ka in enumerate(ka_values):
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

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, ka in enumerate(ka_values):
        color = f"C{i}"
        ax.plot(patch_list, sim_flux[i], marker="o", color=color,
                label=f"sim, $k_a={ka:g}$")
        ax.plot(patch_list, ana_flux[i], linestyle="--", color=color,
                label=f"ana, $k_a={ka:g}$")

    ax.text(0.02, 0.02,
            rf"$k_b={kb:.2g}\ \mathrm{{s^{{-1}}}}$" + "\n"
            + rf"$k_c={kc:.2g}\ \mathrm{{s^{{-1}}}}$",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlabel("Number of patches")
    ax.set_ylabel(r"Normalised flux $J / J_{\max}$")
    ax.set_title(r"Flux vs patch count for different $k_a$")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "kinetics_ka.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
