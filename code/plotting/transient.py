"""Transient survival fraction N(t)/N0: simulation vs analytical.

Compares two cases:
  1. Patchy absorption (Robin BC): particles absorbed only on patches.
  2. Fully absorbing (Dirichlet BC): all outer-wall hits absorbed.

Overlays random walk results with eigenfunction-expansion analytical solutions.

Run: python plotting/transient.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt

from classes.params import SimulationParams
from stochastic_sim.transient import TransientSimulation
from analytical.transient import TransientAnalytical

# --- Experiment parameters ---
D = 25.0
R = 7.5
H = 10.0
dt = 1e-6
T = 0.05
N = 1000
a = 5.0
n_patches = 5000
patch_radius = 0.05
record_stride = 100
n_modes = 75
p0 = "uniform_volume"

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rng = np.random.default_rng(int(time.time()))

    # Effective Robin parameter for the patchy case
    k_eff = (2.0 * n_patches * patch_radius) / (H * np.pi)

    # --- Patchy / Robin simulation ---
    params_patchy = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius,
    )
    res_patchy = TransientSimulation(params_patchy, rng).run(record_stride=record_stride)
    t = res_patchy["t"]
    rw_patchy = res_patchy["frac_alive"]

    ana_patchy = TransientAnalytical(params_patchy, n_modes=n_modes).survival_fraction(
        t, bc="robin", k=k_eff, p0=p0, eps=params_patchy.eps_respawn
    )

    # --- Fully absorbing / Dirichlet simulation ---
    params_full = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=0, patch_radius=None,
    )
    res_full = TransientSimulation(params_full, rng).run(record_stride=record_stride)
    rw_full = res_full["frac_alive"]
    if not np.array_equal(res_full["t"], t):
        rw_full = np.interp(t, res_full["t"], rw_full)

    ana_full = TransientAnalytical(params_full, n_modes=n_modes).survival_fraction(
        t, bc="dirichlet", p0=p0
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    ax.plot(t, rw_patchy, linestyle="None", marker="o", markersize=3,
            markeredgewidth=0.6, alpha=0.85, label="RW (patchy)")
    ax.plot(t, rw_full, linestyle="None", marker="s", markersize=3,
            markeredgewidth=0.6, alpha=0.85, label="RW (full absorbing)")
    ax.plot(t, ana_patchy, linewidth=2, label="Analytical (Robin, patchy)")
    ax.plot(t, ana_full, linewidth=2, label="Analytical (Dirichlet, full absorbing)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Survival fraction $N(t)/N_0$")
    ax.set_title("Transient survival: RW vs analytical")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()
    ax.legend(frameon=True, framealpha=0.9)

    text = (
        rf"$D={D}$, $R={R}$, $a={a}$, $H={H}$" "\n"
        rf"$dt={dt:.1e}$, $T={T}$, $N_0={N}$" "\n"
        rf"patchy: $N_p={n_patches}$, $s={patch_radius}$, $k_{{eff}}={k_eff:.3g}$"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.85))

    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "transient_survival.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
