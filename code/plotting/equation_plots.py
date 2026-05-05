"""Analytical equation visualisations (no simulation required).

Plots:
  1. mu = J/J_max vs receptor surface coverage for multiple C0 values.
  2. Bound receptor percentage vs C0 for multiple receptor counts N.

Run: python plotting/equation_plots.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import matplotlib.pyplot as plt

from classes.params import SimulationParams
from analytical.steady_state import AnalyticalModel, molar_to_molecules_per_um3

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# --- Physical parameters (edit here) ---
D = 125.0
a = 5.0
R = 7.5
H = 10.0
patch_radius = 0.005
ka = 3e-3
kb = 4e-3
kc = 4e-3

# C0 sweep (molar) and receptor count list
C0_M_list = np.logspace(-12, -4, 61)
N_list = np.arange(10000, 100001, 40000, dtype=int)


# ------------------------------------------------------------------
# Helper: build an AnalyticalModel for given n_patches
# ------------------------------------------------------------------

def _model(n_patches):
    return AnalyticalModel(SimulationParams(
        N=1, D=D, a=a, R=R, H=H,
        n_patches=int(n_patches), patch_radius=patch_radius,
        ka=ka, kb=kb, kc=kc,
    ))


# ------------------------------------------------------------------
# Plot 1: mu vs surface coverage for several C0 values
# ------------------------------------------------------------------

def plot_mu_vs_coverage():
    C0_um3 = molar_to_molecules_per_um3(C0_M_list)
    s = float(patch_radius)
    phi_list = (N_list.astype(float) * s * s) / (2.0 * float(R) * float(H))

    mu_mat = np.zeros((len(C0_um3), len(N_list)))
    for j, N in enumerate(N_list):
        model = _model(N)
        for i, C0 in enumerate(C0_um3):
            mu_mat[i, j] = model.mu(C0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, C0_M in enumerate(C0_M_list):
        ax.plot(phi_list, mu_mat[i], label=rf"$C_0={C0_M:.1e}$ M")

    for N_ref in [100_000, 1_000_000]:
        phi_ref = (N_ref * s * s) / (2.0 * R * H)
        ax.axvline(phi_ref, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(phi_ref, 0.02, rf"$N={N_ref:.0e}$",
                rotation=90, va="bottom", ha="right", fontsize=9)

    ax.set_xlabel(r"Receptor surface coverage $\phi = N s^2 / (2RH)$")
    ax.set_ylabel(r"$\mu = J/J_{\max}$")
    ax.set_title(r"Uptake efficiency $\mu$ vs receptor coverage")
    ax.grid(True)
    ax.legend(fontsize=8)

    ka_str = r"$\infty$" if math.isinf(ka) else f"{ka:g}"
    param_text = (
        f"D={D:g} µm²/s, a={a:g}, R={R:g}, H={H:g} µm\n"
        f"s={s:g} µm, ka={ka_str}, kb={kb:g} s⁻¹, kc={kc:g} s⁻¹"
    )
    plt.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, param_text, ha="left", va="bottom",
             fontsize=9, family="monospace")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "mu_vs_coverage.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ------------------------------------------------------------------
# Plot 2: bound receptor % vs C0 for several N values
# ------------------------------------------------------------------

def plot_bound_percent_vs_C0():
    C0_um3 = molar_to_molecules_per_um3(C0_M_list)

    fb_mat = np.zeros((len(N_list), len(C0_um3)))
    for i, N in enumerate(N_list):
        model = _model(N)
        for j, C0 in enumerate(C0_um3):
            fb_mat[i, j] = model.bound_percent(C0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, N in enumerate(N_list):
        ax.plot(C0_M_list, fb_mat[i], label=f"N={N:,}")

    ax.set_xscale("log")
    ax.set_xlabel(r"$C_0$ [M]")
    ax.set_ylabel("Bound receptors [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Receptor occupancy vs bulk concentration")
    ax.grid(True)
    ax.legend()

    ka_str = r"$\infty$" if math.isinf(ka) else f"{ka:g}"
    param_text = (
        f"D={D:g} µm²/s, a={a:g}, R={R:g}, H={H:g} µm\n"
        f"s={patch_radius:g} µm, ka={ka_str}, kb={kb:g} s⁻¹, kc={kc:g} s⁻¹"
    )
    plt.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.02, param_text, ha="left", va="bottom",
             fontsize=9, family="monospace")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    outpath = os.path.join(FIGURES_DIR, "bound_percent_vs_C0.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    plot_mu_vs_coverage()
    plot_bound_percent_vs_C0()
