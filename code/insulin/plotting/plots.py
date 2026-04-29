"""Visualization functions for simulation vs analytical comparisons."""
import matplotlib.pyplot as plt
import numpy as np


def plot_rate_vs_radius(a_sorted, theory_rates, sim_rates,
                        filename="rate_vs_a_cyl.png", figsize=(6, 4), dpi=300):
    plt.figure(figsize=figsize)
    plt.plot(a_sorted, theory_rates, label=r"Theory $2\pi H D / \ln(R/a)$", linewidth=2)
    plt.plot(a_sorted, sim_rates, "o--", label="Simulation")
    plt.xlabel("Inner radius a (um)")
    plt.ylabel("Rate constant k (um^3/s)")
    plt.title("Cylinder: steady-state rate vs a")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Saved: {filename}")


def plot_concentration_profiles(r, C, times, filename=None, figsize=(6, 4), dpi=300):
    plt.figure(figsize=figsize)
    for i in range(C.shape[0]):
        if np.any(np.isfinite(C[i])):
            plt.plot(r, C[i], label=f"t={times[i]:.3g}")
    plt.xlabel("radius r")
    plt.ylabel("concentration C(r)")
    plt.title("Radial concentration profiles at different times")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi)
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()


def plot_flux_vs_patches(n_patches_sorted, analytical_solutions, sim_ratios,
                         filename="Patches_cyl.png", figsize=(7, 5), dpi=300):
    plt.figure(figsize=figsize)
    plt.plot(n_patches_sorted, analytical_solutions, label="Analytical solution", linewidth=3)
    plt.plot(n_patches_sorted, sim_ratios, "o--", label="Simulation", markersize=6)
    plt.xlabel("Number of patches (on outer cylinder)")
    plt.ylabel(r"$J / J_{\max}$")
    plt.title("Cylinder: flux vs number of absorbing patches")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Saved: {filename}")


def plot_survival_comparison(t, rw_patchy, rw_full, ana_patchy, ana_full,
                             params, k_eff, savepath, title=None,
                             figsize=(6.8, 4.4), dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(t, rw_patchy, linestyle="None", marker="o", markersize=3.0,
            markeredgewidth=0.6, alpha=0.85, label="RW (patchy)")
    ax.plot(t, rw_full, linestyle="None", marker="s", markersize=3.0,
            markeredgewidth=0.6, alpha=0.85, label="RW (full absorbing)")
    ax.plot(t, ana_patchy, linewidth=2.0, alpha=0.95, label="Analytical (Robin, patchy)")
    ax.plot(t, ana_full, linewidth=2.0, alpha=0.95, label="Analytical (Dirichlet, full absorbing)")

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Survival fraction  $N(t)/N_0$")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()
    ax.legend(frameon=True, framealpha=0.9)

    text = (
        rf"$D={params.D}$, $R={params.R}$, $a={params.a}$, $H={params.H}$" "\n"
        rf"$dt={params.dt:.1e}$, $T={params.T}$, $N_0={params.N}$" "\n"
        rf"patchy: $N_p={params.n_patches}$, $s={params.patch_radius}$, "
        rf"$k_{{eff}}={k_eff:.3g}$"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.85))

    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savepath}")


def plot_kinetics_comparison(varied_values, patch_list, sim_flux, ana_flux,
                             varied_name="kc", fixed_params=None,
                             bp_analytical=None, filename=None):
    """Generic plotter for kinetics parameter sweeps (ka, kb, or kc)."""
    plt.figure(figsize=(7, 5))

    for i, val in enumerate(varied_values):
        plt.plot(patch_list, sim_flux[i], marker="o", label=f"sim, {varied_name}={val:g}")
        plt.plot(patch_list, ana_flux[i], linestyle="--", label=f"ana, {varied_name}={val:g}")

    if bp_analytical is not None:
        plt.plot(patch_list, bp_analytical, linestyle="-.", linewidth=2.5,
                 alpha=0.9, label=r"BP limit ($k_c=\infty$)")

    if fixed_params:
        text = "\n".join(rf"${k} = {v:.2g}$" for k, v in fixed_params.items())
        plt.text(0.02, 0.02, text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.xlabel("Number of patches")
    plt.ylabel(r"Normalized flux  $k_{eff}/k_{full}$")
    plt.title(f"Flux vs number of patches for different ${varied_name}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_name = filename or f"Variable_{varied_name}"
    plt.savefig(save_name)
    plt.show()
