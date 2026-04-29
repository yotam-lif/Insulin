import math

import numpy as np
import matplotlib.pyplot as plt

AVOGADRO = 6.02214076e23

def molar_to_um3(C0_M):
    """Convert concentration from M (mol/L) to molecules/µm³."""
    return C0_M * AVOGADRO / 1e15

def mu_analytical(
    C0: float,
    *,
    D: float,
    a: float,
    R: float,
    H: float,
    n_patches: int,
    patch_radius: float,
    ka: float,
    kb: float,
    kc: float
) -> float:
    """Return mu = J/Jmax."""
    # effective association to a patch with finite ka
    if np.isinf(ka):
        ka_star = 4.0 * D * patch_radius
    else:
        ka_star = 4.0 * D * patch_radius / (1.0 + (4.0 * D * patch_radius) / ka)
    if (kb==0 and kc==math.inf):
     mu= (n_patches * patch_radius) / (n_patches * patch_radius + H * np.pi / (2 * np.log(R / a)))
    else:
     Kd = (kb + kc) / ka_star
     denom_geom = R * np.log(R / a)
     sigma0 = n_patches / (2.0 * np.pi * R * H)

     A = D * (C0 + Kd) / denom_geom + sigma0 * kc
     B = 4.0 * (D * C0 / denom_geom) * sigma0 * kc
     C = 2.0 * (D * C0 / denom_geom)

     disc = A**2 - B
     disc = np.maximum(disc, 0.0)  # numerical safety
     mu = (A - np.sqrt(disc)) / C
    return float(mu)
def plot_mu_vs_receptors_for_C0s(
    C0_M_list,          # in MOLAR
    N_list,
    *,
    D, a, R, H,
    patch_radius,       # this is s
    ka, kb, kc,
    title=r"$\mu = J/J_{\max}$ vs receptor surface coverage"
):
    import numpy as np
    import matplotlib.pyplot as plt

    C0_M_list = np.asarray(C0_M_list, dtype=float)
    N_list = np.asarray(N_list, dtype=int)

    # convert to internal units (molecules / µm^3)
    C0_internal = molar_to_um3(C0_M_list)

    # ---- x-axis: fractional surface coverage phi ----
    s = float(patch_radius)
    phi_list = (N_list.astype(float) * s * s) / (2.0 * float(R) * float(H))

    mu_mat = np.zeros((C0_M_list.size, N_list.size), dtype=float)
    for i, C0_um3 in enumerate(C0_internal):
        for j, N in enumerate(N_list):
            mu_mat[i, j] = mu_analytical(
                C0_um3,
                D=D, a=a, R=R, H=H,
                n_patches=int(N),
                patch_radius=s,
                ka=ka, kb=kb, kc=kc
            )

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, C0_M in enumerate(C0_M_list):
        ax.plot(
            phi_list, mu_mat[i],
            label=rf"$C_0={C0_M:.1e}\,\mathrm{{M}}$"
        )

    # ---- vertical reference lines for specific N ----
    N_refs = [ 100_000, 1_000_000]
    for N_ref in N_refs:
        phi_ref = (N_ref * s * s) / (2.0 * R * H)
        ax.axvline(
            phi_ref,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7
        )
        ax.text(
            phi_ref, 0.02,
            rf"$N={N_ref:.0e}$",
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=9
        )

    ax.set_xlabel(r"Receptor surface coverage $\phi = \frac{N\pi s^2}{2\pi R H}$")
    ax.set_ylabel(r"$\mu = J/J_{\max}$")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # ---- parameter block ----
    ka_str = "∞" if np.isinf(ka) else f"{ka:g}"
    param_text = (
        "System parameters:\n"
        f"D={D:g} µm²/s, a={a:g} µm, R={R:g} µm, H={H:g} µm\n"
        f"s={s:g} µm, ka={ka_str}, kb={kb:g} s⁻¹, kc={kc:g} s⁻¹\n"
        f"C0 (input) = {[f'{c:.1e}' for c in C0_M_list]} M\n"
        f"N_list = {list(N_list)}"
    )

    plt.subplots_adjust(bottom=0.34)
    fig.text(
        0.01, 0.02, param_text,
        ha="left", va="bottom",
        fontsize=9, family="monospace"
    )

    plt.show()
    return mu_mat, phi_list
def percent_bound_receptors_from_mu(
    *,
    mu: float,
    C0_um3: float,
    D: float,
    patch_radius: float,
    ka: float,
    kb: float,
    kc: float
) -> float:
    """% bound receptors = 100 * C(R)/(C(R)+Kd), with C(R)=C0(1-mu)."""
    if np.isinf(ka):
        ka_star = 4.0 * D * patch_radius
    else:
        ka_star = 4.0 * D * patch_radius / (1.0 + (4.0 * D * patch_radius) / ka)

    Kd = (kb + kc) / ka_star
    C_R = C0_um3 * (1.0 - mu)
    frac = C_R / (C_R + Kd)
    return 100.0 * float(frac)


# ---- Sweep: curves are N, x-axis is C0 (M) ----
def plot_percent_bound_vs_C0_for_Ns(
    C0_M_list,     # x-axis in MOLAR
    N_list,        # curves
    *,
    D, a, R, H,
    patch_radius,
    ka, kb, kc,
    title="Bound receptor percentage vs $C_0$",
    c0_marks=(5e-11, 1e-9),        # <-- CHANGED: multiple vertical lines (tuple/list)
    annotate_intersections=True,    # annotate each curve at each marked C0
):
    C0_M_list = np.asarray(C0_M_list, dtype=float)
    C0_um3_list = molar_to_um3(C0_M_list)
    N_list = np.asarray(N_list, dtype=int)

    fb_mat = np.zeros((N_list.size, C0_M_list.size), dtype=float)  # rows=N, cols=C0

    for i, N in enumerate(N_list):
        for j, C0_um3 in enumerate(C0_um3_list):
            mu = mu_analytical(
                C0_um3,
                D=D, a=a, R=R, H=H,
                n_patches=int(N),
                patch_radius=patch_radius,
                ka=ka, kb=kb, kc=kc
            )
            fb_mat[i, j] = percent_bound_receptors_from_mu(
                mu=mu,
                C0_um3=C0_um3,
                D=D,
                patch_radius=patch_radius,
                ka=ka, kb=kb, kc=kc
            )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, N in enumerate(N_list):
        ax.plot(C0_M_list, fb_mat[i], label=f"N={N:,}")

    ax.set_xlabel(r"$C_0$ [M]")
    ax.set_ylabel("Bound receptors [%]")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    ax.set_xscale("log")

    # ---- NEW: multiple vertical lines + intersections ----
    # intersections[N][C0_mark] = y_at_mark
    intersections = {int(N): {} for N in N_list}

    x_min, x_max = float(np.min(C0_M_list)), float(np.max(C0_M_list))
    logx = np.log10(C0_M_list)

    # normalize c0_marks to a list of floats
    if c0_marks is None:
        c0_marks_list = []
    elif np.isscalar(c0_marks):
        c0_marks_list = [float(c0_marks)]
    else:
        c0_marks_list = [float(x) for x in c0_marks]

    # Keep only marks within range, and sort (nice for labeling)
    c0_marks_in = sorted([x for x in c0_marks_list if x_min <= x <= x_max])

    for m, c0_mark in enumerate(c0_marks_in):
        ax.axvline(c0_mark, linestyle="--")
        ax.text(
            c0_mark, 98,
            rf"$C_0={c0_mark:.0e}$",
            rotation=90,
            va="top",
            ha="right"
        )

        logx_mark = np.log10(c0_mark)

        for i, N in enumerate(N_list):
            y_curve = fb_mat[i]
            y_mark = float(np.interp(logx_mark, logx, y_curve))
            intersections[int(N)][float(c0_mark)] = y_mark

            ax.scatter([c0_mark], [y_mark], zorder=5)

            if annotate_intersections:
                # offset depends on curve index i and mark index m to reduce overlap
                ax.annotate(
                    f"{y_mark:.2f}%",
                    (c0_mark, y_mark),
                    textcoords="offset points",
                    xytext=(8 + 18*m, 6 + 12*i),
                    ha="left",
                    fontsize=9
                )

    # ---- Parameter block below plot ----
    ka_str = "∞" if np.isinf(ka) else f"{ka:g}"
    param_text = (
        "System parameters:\n"
        f"D={D:g} µm²/s, a={a:g} µm, R={R:g} µm, H={H:g} µm\n"
        f"patch_radius={patch_radius:g} µm, ka={ka_str} µm^3/s , kb={kb:g} s⁻¹, kc={kc:g} s⁻¹\n"
        f"N_list={list(N_list)}\n"
        f"c0_marks={c0_marks_in}"
    )

    plt.subplots_adjust(bottom=0.30)
    fig.text(0.01, 0.02, param_text, ha="left", va="bottom",
             fontsize=9, family="monospace")

    plt.show()
    return fb_mat, intersections


if __name__ == "__main__":
    # -------------------------
    # Example: fill YOUR values
    # -------------------------
    D = 125.0
    a = 5.0
    R = 7.5
    H = 10.0

    patch_radius = 0.005
    ka = 3e-3  # 1.6e-3
    kb = 4e-3   #4e-3
    kc = 4e-3    #4e-3

    #C0_list = [ 1e-12,1e-11,1e-10, 1e-9, 1e-8, 1e-7]
    C0_list=np.logspace(-12, -4, 61)
    N_list = np.arange(10000, 100000, 40000)

    fb_mat = plot_percent_bound_vs_C0_for_Ns(
        C0_list, N_list,
        D=D, a=a, R=R, H=H,
        patch_radius=patch_radius,
        ka=ka, kb=kb, kc=kc,
    )

    mu_mat = plot_mu_vs_receptors_for_C0s(
        C0_list, N_list,
        D=D, a=a, R=R, H=H,
        patch_radius=patch_radius,
        ka=ka, kb=kb, kc=kc
    )

