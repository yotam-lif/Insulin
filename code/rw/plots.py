import matplotlib.pyplot as plt
import numpy as np

def plot_rate_vs_a_cyl(
    a_sorted,
    theory_rates,
    sim_rates,
    filename="rate_vs_a_cyl.png",
    figsize=(6, 4),
    dpi=300,
):

    plt.figure(figsize=figsize)

    plt.plot(a_sorted,theory_rates,label=r"Theory $2\pi H D / \ln(R/a)$",inewidth=2,)

    plt.plot(a_sorted, sim_rates,  "o--",label="Simulation",)

    plt.xlabel("Inner radius a (μm)")
    plt.ylabel("Rate constant k (μm³/s)")
    plt.title("Cylinder: steady-state rate vs a")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=dpi)
    plt.close()

    print(f"Saved: {filename}")

def plot_radial_concentration_profiles(
        r,
        C,
        times,
        filename=None,
        figsize=(6, 4),
        dpi=300,
):

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


def plot_flux_vs_patches_cyl(
    n_patches_sorted,
    analytical_solutions,
    sim_ratios,
    filename="Patches_cyl.png",
    figsize=(7, 5),
    dpi=300,
):

    plt.figure(figsize=figsize)

    plt.plot(
        n_patches_sorted,
        analytical_solutions,
        label="Analytical solution",
        linewidth=3,
    )

    plt.plot(
        n_patches_sorted,
        sim_ratios,
        "o--",
        label="Simulation",
        markersize=6,
    )

    plt.xlabel("Number of patches (on outer cylinder)")
    plt.ylabel(r"$J / J_{\max}$")
    plt.title("Cylinder: flux vs number of absorbing patches")

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=dpi)
    plt.close()

    print(f"Saved: {filename}")

def plot_survival_fraction_comparison(
            t,
            rw_patchy,
            rw_full,
            ana_patchy,
            ana_full,
            params,
            k_eff,
            savepath,
            title=None,
            figsize=(6.8, 4.4),
            dpi=200,
    ):


    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # --- RW markers ---
    ax.plot(
            t,rw_patchy,linestyle="None",marker="o",markersize=3.0,markeredgewidth=0.6,alpha=0.85,label="RW (patchy)",)

    ax.plot(t,rw_full,linestyle="None",marker="s",markersize=3.0,markeredgewidth=0.6,alpha=0.85,label="RW (full absorbing)",)

    # --- Analytical curves ---
    ax.plot(t,ana_patchy,linewidth=2.0,alpha=0.95,label="Analytical (Robin, patchy)",)

    ax.plot(t,ana_full,linewidth=2.0,alpha=0.95,label="Analytical (Dirichlet, full absorbing)",)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Survival fraction  $N(t)/N_0$")

    if title is not None:
        ax.set_title(title)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()
    ax.legend(frameon=True, framealpha=0.9)

    # --- Parameter box ---
    text = (
        rf"$D={params.D}$, $R={params.R}$, $a={params.a}$, $H={params.H}$" "\n"
         rf"$dt={params.dt:.1e}$, $T={params.T}$, $N_0={params.N}$" "\n"
         rf"patchy: $N_p={params.n_patches}$, $s={params.patch_radius}$, "
         rf"$k_{{eff}}={k_eff:.3g}$"
    )

    ax.text(0.02,0.02,text,transform=ax.transAxes,fontsize=9,va="bottom",ha="left",bbox=dict(boxstyle="round,pad=0.25", alpha=0.85),)

    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {savepath}")
def plot_kc_comparison(kc_values,patch_list,sim_flux,ana_flux,ka,kb,BP_ana):
     # ------------------ Plot ------------------
     plt.figure(figsize=(7, 5))

     for i, kc in enumerate(kc_values):
         plt.plot(
             patch_list,
             sim_flux[i],
             marker="o",
             label=f"sim, kc={kc:g}"
         )
         plt.plot(
             patch_list,
             ana_flux[i],
             linestyle="--",
             label=f"ana, kc={kc:g}"
         )
     plt.plot(
         patch_list,
         BP_ana,
         linestyle="-.",
         linewidth=2.5,
         alpha=0.9,
         label=r"BP limit ($k_c=\infty$)"
     )

     # ---- Minimal kinetics annotation ----
     text = rf"$k_a = {ka:.2g}$" + "\n" + rf"$k_b = {kb:.2g}$"

     plt.text(
         0.02, 0.02, text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
     )

     plt.xlabel("Number of patches")
     plt.ylabel("Normalized flux  $k_{eff}/k_{full}$")
     plt.title("Flux vs number of patches for different $k_c$")
     plt.legend()
     plt.grid(True)
     plt.tight_layout()
     plt.savefig("Variable_kc")
     plt.show()

def plot_kb_comparison(kb_values, patch_list, sim_flux, ana_flux,ka,kc):
         # ------------------ Plot ------------------
         plt.figure(figsize=(7, 5))

         for i, kb in enumerate(kb_values):
             plt.plot(
                 patch_list,
                 sim_flux[i],
                 marker="o",
                 label=f"sim, kb={kb:g}"
             )
             plt.plot(
                 patch_list,
                 ana_flux[i],
                 linestyle="--",
                 label=f"ana, kb={kb:g}"
             )

         # ---- Minimal kinetics annotation ----
         text = rf"$k_a = {ka:.2g}$" + "\n" + \
                rf"$k_c = {kc:.2g}\,\mathrm{{s^{{-1}}}}$"

         plt.text(
             0.02, 0.02, text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
         )

         plt.xlabel("Number of patches")
         plt.ylabel("Normalized flux  $k_{eff}/k_{full}$")
         plt.title("Flux vs number of patches for different $k_b$")
         plt.legend()
         plt.grid(True)
         plt.tight_layout()
         plt.savefig("Variable_kb")
         plt.show()


def plot_ka_comparison(ka_values, patch_list, sim_flux, ana_flux,kb,kc):
    # ------------------ Plot ------------------
    plt.figure(figsize=(7, 5))

    for i, ka in enumerate(ka_values):
        plt.plot(
            patch_list,
            sim_flux[i],
            marker="o",
            label=f"sim, ka={ka:g}"
        )
        plt.plot(
            patch_list,
            ana_flux[i],
            linestyle="--",
            label=f"ana, ka={ka:g}"
        )

        # ---- Minimal kinetics annotation ----
    text = rf"$k_c = {kc:.2g}\,\mathrm{{s^{{-1}}}}$" + "\n" + \
           rf"$k_b = {kb:.2g}\,\mathrm{{s^{{-1}}}}$"

    plt.text(
        0.02, 0.02, text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.xlabel("Number of patches")
    plt.ylabel("Normalized flux  $k_{eff}/k_{full}$")
    plt.title("Flux vs number of patches for different $k_b$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Variable_ka")
    plt.show()