import math

import numpy as np
from .params import RWCylParams   # new param dataclass for cylinder geometry
from .simulation import simulate_flux_cyl
from .c0_numerical_calculations import N_from_c0_new

def worker_rate_vs_a_cyl(args):

    a, D, R, H, dt, T, N, seed = args

    rng = np.random.default_rng(seed)
    L = R - a
    tau_diff = (L ** 2) / (np.pi ** 2 * D)
    t_relax = 5 * tau_diff

    T = 2 * t_relax
    print(f"the T {T}")
    params = RWCylParams(
        N=N,
        D=D,
        a=a,
        R=R,
        H=H,
        dt=dt,
        T=T,
        # optional extras; tweak as you like
        n_r_segments=30,
        n_patches=0,
        patch_radius=None,
        t_relax=t_relax
    )

    out = simulate_flux_cyl(params, rng)

    k_sim = out["rate_normalized"]          # μm³/s
    # Analytic rate constant for cylinder:
    k_theory = 2.0 * np.pi * H * D / np.log(R / (a))

    return (a, k_sim, k_theory)




def worker_absorbing_patches_cyl(args):

    (n_patches,t_relax,
     D, R, H, dt, T,
     N, a, patch_radius,
     seed) = args

    rng = np.random.default_rng(seed)

    params = RWCylParams(
        N=N,
        D=D,
        a=a,
        R=R,
        H=H,
        dt=dt,
        T=T,
        n_r_segments=30,
        n_patches=n_patches,
        patch_radius=patch_radius,
        t_relax=t_relax
    )

    res = simulate_flux_cyl(params, rng)



    k_full = 2.0 * np.pi * H * D / np.log(R / a)
    sim_ratio = res["rate_normalized"] / k_full

    coverage = (n_patches * (patch_radius ** 2)) / (2.0 * R * H)

    analytical_solution = (n_patches * patch_radius) / (n_patches * patch_radius + H * np.pi / (2 * np.log(R / a)))


    return n_patches, sim_ratio, analytical_solution, coverage


def worker_kinetics_patches_cyl_steady(args):
    (n_patches, t_relax,
     D, R, H, dt, T,
     C0, a, patch_radius,ka,kb,kc,
     seed) = args

    rng = np.random.default_rng(seed)

    N_total = N_from_c0_new(
        C0,
        H=H,
        a=a,
        R=R,
        D=D,
        n_patches=n_patches,
        patch_radius=patch_radius,
        ka=ka,
        kb=kb,
        kc=kc,
        n_sites_per_patch=1,  # must match PatchSetCyl
    )

    params = RWCylParams(
        C0=C0,
        N=int(N_total),
        D=D,
        a=a,
        R=R,
        H=H,
        dt=dt,
        T=T,
        n_r_segments=30,
        n_patches=n_patches,
        patch_radius=patch_radius,
        t_relax=t_relax,
        ka=ka,
        kb=kb,
        kc=kc
    )

    print(f"the ka,kb,kc {ka},{kb},{kc} number of patches is  {params.n_patches}, the N particles is {N_total}")

    res = simulate_flux_cyl(params, rng)

    k_full = 2.0 * np.pi * H * D / np.log(R / a)

    sim_ratio = res["rate_normalized"] / k_full
    C0=res["c0_measured"]
    if (kc!=math.inf):

          ka_star = 4 * D * patch_radius / (1.0 + (4 * D * patch_radius) / ka)

          Kd=(kb+kc)/ka_star
          denom_geom = R * np.log(R / a)
          sigma0=n_patches/(2*np.pi*R*H)
          A = D * (C0 + Kd) / denom_geom + sigma0 * kc
          B = 4.0 * (D * C0 / denom_geom) * sigma0 * kc
          C = 2.0 * (D * C0 / denom_geom)

          analytical_solution = (A - np.sqrt(A ** 2 - B)) / C



    else:
         analytical_solution = (n_patches * patch_radius) / (n_patches * patch_radius + H * np.pi / (2 * np.log(R / a)))

    return n_patches, sim_ratio, analytical_solution

