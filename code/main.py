import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
from rw import (worker_rate_vs_a_cyl,simulate_flux_cyl,worker_absorbing_patches_cyl,worker_kinetics_patches_cyl_steady,plot_rate_vs_a_cyl,
                plot_radial_concentration_profiles,plot_flux_vs_patches_cyl,annulus_survival_fraction,simulate_transient_cyl_absorb,plot_survival_fraction_comparison,plot_kc_comparison,plot_kb_comparison,plot_ka_comparison)
from rw.params import RWCylParams


def fully_absorbrd_rate_vs_a_cyl(rng, max_workers=None):
    D = 100
    R = 7.5
    H = 10.0  # choose whatever height you like
    dt = 1e-6
    T = 1
    N = 10000

    # Range of inner radii to test
    a_values = np.linspace(1.0, 5.0, 7)

    # Give each a its own seed
    seeds = rng.integers(0, 2**63 - 1, size=len(a_values), dtype=np.int64)
    args_list = [
        (float(a), D, R, H, dt, T, N, int(seed))
        for a, seed in zip(a_values, seeds)
    ]

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(worker_rate_vs_a_cyl, args_list))

    # Sort by a
    results.sort(key=lambda tup: tup[0])
    a_sorted     = np.array([r[0] for r in results])
    sim_rates    = np.array([r[1] for r in results])
    theory_rates = np.array([r[2] for r in results])

    for a, k_sim in zip(a_sorted, sim_rates):
        print(f"a = {a:.3f} finished, k_sim = {k_sim:.3e} μm³/s")

    plot_rate_vs_a_cyl(a_sorted,theory_rates,sim_rates,filename="rate_vs_a_cyl.png",figsize=(6, 4),dpi=300)



def experiment_concentration_profile_cyl(rng):
    D = 10
    R = 7.5
    H = 10.0
    dt = 1e-5
    N = 1000
    a_for_profile = 5
    patch_radius = 0.1

    L = R - a_for_profile
    tau_diff = (L ** 2) / (np.pi ** 2 * D)
    t_relax = 10 * tau_diff
    T =  t_relax+4

    print(f"the t relax is {t_relax}")
    # Set up parameters for the cylindrical simulation
    params = RWCylParams(N=N,D=D,a=a_for_profile,R=R,H=H,dt=dt,t_relax=t_relax,T=T,n_r_segments=30,  n_patches=1000, patch_radius=patch_radius,eps_c0=0.1,eps_respawn=1e-10,ka=math.inf,kb=0,kc=1)
    # Run simulation
    res = simulate_flux_cyl(params, rng,concentration_check=True)

    r = res["r_centers"]
    C = res["concentrations_time"]
    times = res["profile_times"]

    plot_radial_concentration_profiles(r,C,times,filename=None,figsize=(6, 4),dpi=300,)



def absorbing_Patches_cyl(rng,max_workers):
    D = 10
    R = 7.5
    H = 10.0  # choose whatever height you like
    dt = 1e-6
    N = 10000
    a = 5
    patch_radius = 0.1

    L = R - a
    tau_diff = (L ** 2) / (np.pi ** 2 * D)
    t_relax = 15* tau_diff
    T=2*t_relax


    print(f"t_relax ≈ {t_relax:.3g} s")
    print(f"step sigma = {np.sqrt(2 * D * dt):.3g} μm")

    # Number of patches to test
    patch_list = np.arange(100, 1400, 200)  # 5,10,...,50 patches

    # seeds per n_patches
    seeds = rng.integers(0, 2 ** 63 - 1, size=len(patch_list), dtype=np.int64)

    # ------Calculate 3D system flux------
    args_list = [
        (int(n_patches),t_relax, D, R, H, dt, T, N, a, patch_radius, int(seed))
        for n_patches, seed in zip(patch_list, seeds)
    ]
    # parallel run
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(worker_absorbing_patches_cyl, args_list))

    # unpack & sort
    results.sort(key=lambda tup: tup[0])
    n_patches_sorted = np.array([r[0] for r in results])
    sim_ratios = np.array([r[1] for r in results])
    analytical_solutions = np.array([r[2] for r in results])
    coverages = np.array([r[3] for r in results])

    for n_patches, sim_ratio, analytical_solution, cov in zip(
                n_patches_sorted, sim_ratios, analytical_solutions, coverages
    ):
        print(
            f"n_patches={n_patches:4d} | sim={sim_ratio:.4f} "
            f"| naive_cov={analytical_solution:.4f} | coverage={cov:.3f}"
        )

    plot_flux_vs_patches_cyl(n_patches_sorted,analytical_solutions,sim_ratios)

def transient_simulation(rng):
    D = 25
    R = 7.5
    H = 10.0  # choose whatever height you like
    dt = 1e-6
    T = 0.05
    N = 1000
    a_for_profile = 5
    patch_radius = 0.05
    record_stride = 100
    n_modes = 75
    p0 = "uniform_volume"


    # Set up parameters for the cylindrical simulation
    params = RWCylParams(N=N,D=D,a=a_for_profile,R=R,H=H,dt=dt,T=T,n_r_segments=30, n_patches=5000, patch_radius=patch_radius,eps_c0=0.1,eps_respawn=1e-10)

    # ---------- Patchy / Robin case ----------
    # effective Robin parameter (your existing choice)
    k_eff = (2 * params.n_patches * params.patch_radius) / (params.H * np.pi)

    res_patchy = simulate_transient_cyl_absorb(params, rng, record_stride=record_stride)
    t = res_patchy["t"]
    rw_patchy = res_patchy["frac_alive"]

    ana_patchy = annulus_survival_fraction(t=t, a=params.a, R=params.R, D=params.D,n_modes=n_modes,p0=p0,eps=params.eps_respawn,bc="robin",k=k_eff)

    # ---------- Full absorbing / Dirichlet case ----------
    # Make a copy of params if your simulate_transient_cyl_absorb behavior

    params_full = RWCylParams(N=params.N,D=params.D,a=params.a,R=params.R,H=params.H,dt=params.dt,T=params.T,n_r_segments=30,n_patches=0,patch_radius=None,eps_c0=params.eps_c0,eps_respawn=params.eps_respawn)

    res_full = simulate_transient_cyl_absorb(params_full, rng, record_stride=record_stride)
    # time grids should match; if not, interpolate one to the other
    t_full = res_full["t"]
    rw_full = res_full["frac_alive"]

    if not np.array_equal(t_full, t):
        # interpolate full RW onto t
        rw_full = np.interp(t, t_full, rw_full)

    ana_full = annulus_survival_fraction(t=t, a=params_full.a, R=params_full.R, D=params_full.D,n_modes=n_modes,p0=p0,eps=params_full.eps_respawn,bc="dirichlet")

    plot_survival_fraction_comparison(
        t=t,
        rw_patchy=rw_patchy,
        rw_full=rw_full,
        ana_patchy=ana_patchy,
        ana_full=ana_full,
        params=params,
        k_eff=k_eff,
        savepath="survival_fraction_comparison.png",
        title="Cylinder survival: RW vs analytical",
    )


def patches_kinetics_steady_state_kc(rng, max_workers):
    D = 10
    R = 7.5
    H = 10.0
    dt = 1e-6
    a = 5
    C0=10
    patch_radius = 0.1

    L = R - a
    tau_diff = (L ** 2) / (np.pi ** 2 * D)
    t_relax = 30*tau_diff
    T =  t_relax+2

    ka = math.inf
    kb = 0.0

    # kc values → one curve per kc
    #kc_values = np.array([ 1e-1,1,10,100], dtype=float)
    kc_values = np.array([1e-1,0.5,1,2,5], dtype=float)

    # number of patches sweep
    #patch_list = np.arange(100, 1400, 200, dtype=int)
    patch_list = np.arange(100, 6000, 800, dtype=int)


    # storage
    sim_flux = np.zeros((len(kc_values), len(patch_list)))
    ana_flux = np.zeros((len(kc_values), len(patch_list)))

    # independent seeds
    seeds = rng.integers(
        0, 2**63 - 1,
        size=(len(kc_values), len(patch_list)),
        dtype=np.int64
    )

    for i, kc in enumerate(kc_values):

        args_list = [
            (
                int(n_patches),
                float(t_relax),
                float(D),
                float(R),
                float(H),
                float(dt),
                float(T),
                float(C0),
                float(a),
                float(patch_radius),
                float(ka),
                float(kb),
                float(kc),
                int(seeds[i, j]),
            )
            for j, n_patches in enumerate(patch_list)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(worker_kinetics_patches_cyl_steady, args_list))

        # sort by n_patches
        results.sort(key=lambda x: x[0])

        sim_flux[i, :] = [r[1] for r in results]
        ana_flux[i, :] = [r[2] for r in results]

    L = np.log(R / a)
    geom = H * np.pi / (2 * L)

    BP_ana = [
        (n * patch_radius) / (n * patch_radius + geom)
        for n in patch_list
    ]

    plot_kc_comparison(kc_values, patch_list, sim_flux, ana_flux,ka,kb,BP_ana)

def patches_kinetics_steady_state_kb(rng, max_workers):
        D = 10
        R = 7.5
        H = 10.0
        dt = 1e-5
        a = 5
        patch_radius = 0.1
        C0 = 5

        L = R - a
        tau_diff = (L ** 2) / (np.pi ** 2 * D)
        t_relax = 15 * tau_diff
        T = t_relax + 1

        ka = math.inf
        kc = 5

        # kc values → one curve per kc
        kb_values = np.array([1,10,50,100], dtype=float)

        # number of patches sweep
        # patch_list = np.arange(100, 1400, 200, dtype=int)
        patch_list = np.arange(100, 6000, 800, dtype=int)

        # storage
        sim_flux = np.zeros((len(kb_values), len(patch_list)))
        ana_flux = np.zeros((len(kb_values), len(patch_list)))

        # independent seeds
        seeds = rng.integers(
            0, 2 ** 63 - 1,
            size=(len(kb_values), len(patch_list)),
            dtype=np.int64
        )

        for i, kb in enumerate(kb_values):
            args_list = [
                (
                    int(n_patches),
                    float(t_relax),
                    float(D),
                    float(R),
                    float(H),
                    float(dt),
                    float(T),
                    float(C0),
                    float(a),
                    float(patch_radius),
                    float(ka),
                    float(kb),
                    float(kc),
                    int(seeds[i, j]),
                )
                for j, n_patches in enumerate(patch_list)
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(worker_kinetics_patches_cyl_steady, args_list))

            # sort by n_patches
            results.sort(key=lambda x: x[0])

            sim_flux[i, :] = [r[1] for r in results]
            ana_flux[i, :] = [r[2] for r in results]

        plot_kb_comparison(kb_values, patch_list, sim_flux, ana_flux,ka,kc)


def patches_kinetics_steady_state_ka(rng, max_workers):
    D = 10
    R = 7.5
    H = 10.0
    dt = 1e-6
    a = 5
    patch_radius = 0.1
    C0=10

    L = R - a
    tau_diff = (L ** 2) / (np.pi ** 2 * D)
    t_relax = 30 * tau_diff
    T = t_relax + 2

    kb = 5
    kc = 5

    # kc values → one curve per kc
    ka_values = np.array([1e-1,1,2,100], dtype=float)

    # number of patches sweep
    # patch_list = np.arange(100, 1400, 200, dtype=int)
    patch_list = np.arange(100, 6000, 800, dtype=int)

    # storage
    sim_flux = np.zeros((len(ka_values), len(patch_list)))
    ana_flux = np.zeros((len(ka_values), len(patch_list)))

    # independent seeds
    seeds = rng.integers(
        0, 2 ** 63 - 1,
        size=(len(ka_values), len(patch_list)),
        dtype=np.int64
    )

    for i, ka in enumerate(ka_values):
        args_list = [
            (
                int(n_patches),
                float(t_relax),
                float(D),
                float(R),
                float(H),
                float(dt),
                float(T),
                float(C0),
                float(a),
                float(patch_radius),
                float(ka),
                float(kb),
                float(kc),
                int(seeds[i, j]),
            )
            for j, n_patches in enumerate(patch_list)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(worker_kinetics_patches_cyl_steady, args_list))

        # sort by n_patches
        results.sort(key=lambda x: x[0])

        sim_flux[i, :] = [r[1] for r in results]
        ana_flux[i, :] = [r[2] for r in results]

    plot_ka_comparison(ka_values, patch_list, sim_flux, ana_flux,kb,kc)

if __name__ == "__main__":
    rng = np.random.default_rng(int(time.time()))

    #fully_absorbrd_rate_vs_a_cyl(rng)
    #experiment_concentration_profile_cyl(rng)
    #absorbing_Patches_cyl(rng,8)
    #transient_simulation(rng)
    #patches_kinetics_steady_state_kc(rng, 8)
    patches_kinetics_steady_state_kb(rng, 8)
    #patches_kinetics_steady_state_ka(rng, 8)