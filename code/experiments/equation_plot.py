"""Equation-level analysis: mu and bound fraction vs receptor count and C0."""
import math
import numpy as np
import matplotlib.pyplot as plt
from insulin.analytical.steady_state import AnalyticalModel, molar_to_molecules_per_um3
from insulin import SimulationParams


AVOGADRO = 6.02214076e23


def plot_mu_vs_receptors(C0_M_list, N_list, *, D, a, R, H, patch_radius, ka, kb, kc):
    """Plot uptake efficiency mu vs receptor surface coverage for multiple C0."""
    C0_M_list = np.asarray(C0_M_list, dtype=float)
    N_list = np.asarray(N_list, dtype=int)
    C0_internal = molar_to_molecules_per_um3(C0_M_list)

    s = float(patch_radius)
    phi_list = (N_list.astype(float) * s * s) / (2.0 * R * H)

    mu_mat = np.zeros((C0_M_list.size, N_list.size))
    for i, C0_um3 in enumerate(C0_internal):
        for j, N in enumerate(N_list):
            params = SimulationParams(N=0, D=D, a=a, R=R, H=H,
                                      n_patches=int(N), patch_radius=s,
                                      ka=ka, kb=kb, kc=kc)
            model = AnalyticalModel(params)
            mu_mat[i, j] = model.mu(C0_um3)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, C0_M in enumerate(C0_M_list):
        ax.plot(phi_list, mu_mat[i], label=rf"$C_0={C0_M:.1e}\,\mathrm{{M}}$")

    for N_ref in [100_000, 1_000_000]:
        phi_ref = (N_ref * s * s) / (2.0 * R * H)
        ax.axvline(phi_ref, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(phi_ref, 0.02, rf"$N={N_ref:.0e}$", rotation=90,
                va="bottom", ha="right", fontsize=9)

    ax.set_xlabel(r"Receptor surface coverage $\phi$")
    ax.set_ylabel(r"$\mu = J/J_{\max}$")
    ax.set_title(r"$\mu$ vs receptor surface coverage")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return mu_mat, phi_list


def plot_bound_percent_vs_C0(C0_M_list, N_list, *, D, a, R, H, patch_radius, ka, kb, kc,
                              c0_marks=(5e-11, 1e-9)):
    """Plot bound receptor percentage vs C0 for multiple receptor counts."""
    C0_M_list = np.asarray(C0_M_list, dtype=float)
    C0_um3_list = molar_to_molecules_per_um3(C0_M_list)
    N_list = np.asarray(N_list, dtype=int)

    fb_mat = np.zeros((N_list.size, C0_M_list.size))
    for i, N in enumerate(N_list):
        for j, C0_um3 in enumerate(C0_um3_list):
            params = SimulationParams(N=0, D=D, a=a, R=R, H=H,
                                      n_patches=int(N), patch_radius=patch_radius,
                                      ka=ka, kb=kb, kc=kc)
            model = AnalyticalModel(params)
            fb_mat[i, j] = model.bound_percent(C0_um3)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, N in enumerate(N_list):
        ax.plot(C0_M_list, fb_mat[i], label=f"N={N:,}")

    ax.set_xlabel(r"$C_0$ [M]")
    ax.set_ylabel("Bound receptors [%]")
    ax.set_ylim(0, 100)
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()

    logx = np.log10(C0_M_list)
    for c0_mark in (c0_marks or []):
        if C0_M_list.min() <= c0_mark <= C0_M_list.max():
            ax.axvline(c0_mark, linestyle="--")
            ax.text(c0_mark, 98, rf"$C_0={c0_mark:.0e}$", rotation=90, va="top", ha="right")
            for i, N in enumerate(N_list):
                y = float(np.interp(np.log10(c0_mark), logx, fb_mat[i]))
                ax.scatter([c0_mark], [y], zorder=5)
                ax.annotate(f"{y:.2f}%", (c0_mark, y), textcoords="offset points",
                            xytext=(8, 6 + 12 * i), ha="left", fontsize=9)

    plt.tight_layout()
    plt.show()
    return fb_mat


if __name__ == "__main__":
    D, a, R, H = 125.0, 5.0, 7.5, 10.0
    patch_radius = 0.005
    ka, kb, kc = 3e-3, 4e-3, 4e-3

    C0_list = np.logspace(-12, -4, 61)
    N_list = np.arange(10000, 100000, 40000)

    plot_bound_percent_vs_C0(C0_list, N_list, D=D, a=a, R=R, H=H,
                              patch_radius=patch_radius, ka=ka, kb=kb, kc=kc)
    plot_mu_vs_receptors(C0_list, N_list, D=D, a=a, R=R, H=H,
                          patch_radius=patch_radius, ka=ka, kb=kb, kc=kc)
