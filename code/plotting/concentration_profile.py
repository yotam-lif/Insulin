"""Radial concentration profiles C(r) at multiple times during equilibration.

Runs a single steady-state simulation with concentration snapshots enabled
to visualise how the profile approaches the steady-state logarithmic form.

Run: python plotting/concentration_profile.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import numpy as np
import matplotlib.pyplot as plt

from classes.params import SimulationParams
from stochastic_sim.steady_state import SteadyStateSimulation

# --- Experiment parameters ---
D = 10.0
R = 7.5
H = 10.0
dt = 1e-5
N = 1000
a = 5.0
patch_radius = 0.1
n_patches = 1000

L = R - a
tau_diff = (L**2) / (np.pi**2 * D)
t_relax = 10.0 * tau_diff
T = t_relax + 4.0

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rng = np.random.default_rng(int(time.time()))

    params = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        t_relax=t_relax,
        n_patches=n_patches, patch_radius=patch_radius,
        eps_c0=0.1, eps_respawn=1e-10,
        ka=math.inf, kb=0.0, kc=1.0,
    )

    sim = SteadyStateSimulation(params, rng)
    result = sim.run(concentration_profile=True, n_profile_times=10, n_r_bins=40)

    r = result.r_centers
    C = result.concentrations
    times = result.profile_times

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(C.shape[0]):
        if np.any(np.isfinite(C[i])):
            ax.plot(r, C[i], label=f"t = {times[i]:.3g} s")

    ax.set_xlabel("Radius $r$ (µm)")
    ax.set_ylabel("Concentration $C(r)$ (particles/µm³)")
    ax.set_title("Radial concentration profiles during equilibration")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "concentration_profile.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
