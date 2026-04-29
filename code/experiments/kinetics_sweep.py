"""Experiment: steady-state kinetics parameter sweeps (ka, kb, kc)."""
import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments.workers import worker_kinetics_steady
from insulin.plotting import plot_kinetics_comparison


def _run_sweep(rng, max_workers, varied_name, varied_values,
               D, R, H, dt, a, C0, patch_radius, t_relax, T,
               ka, kb, kc, patch_list):
    """Generic sweep: varies one of ka/kb/kc while holding others fixed."""
    sim_flux = np.zeros((len(varied_values), len(patch_list)))
    ana_flux = np.zeros((len(varied_values), len(patch_list)))

    seeds = rng.integers(0, 2**63 - 1, size=(len(varied_values), len(patch_list)), dtype=np.int64)

    for i, val in enumerate(varied_values):
        ka_i = val if varied_name == "ka" else ka
        kb_i = val if varied_name == "kb" else kb
        kc_i = val if varied_name == "kc" else kc

        args_list = [
            (int(n_patches), float(t_relax), float(D), float(R), float(H),
             float(dt), float(T), float(C0), float(a), float(patch_radius),
             float(ka_i), float(kb_i), float(kc_i), int(seeds[i, j]))
            for j, n_patches in enumerate(patch_list)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(worker_kinetics_steady, args_list))

        results.sort(key=lambda x: x[0])
        sim_flux[i, :] = [r[1] for r in results]
        ana_flux[i, :] = [r[2] for r in results]

    # Fixed params annotation (exclude the varied one)
    fixed = {}
    if varied_name != "ka":
        fixed["k_a"] = ka
    if varied_name != "kb":
        fixed["k_b"] = kb
    if varied_name != "kc":
        fixed["k_c"] = kc

    # BP limit curve (only meaningful for kc sweep)
    bp_ana = None
    if varied_name == "kc":
        L = np.log(R / a)
        geom = H * np.pi / (2 * L)
        bp_ana = [(n * patch_radius) / (n * patch_radius + geom) for n in patch_list]

    plot_kinetics_comparison(
        varied_values, patch_list, sim_flux, ana_flux,
        varied_name=varied_name, fixed_params=fixed,
        bp_analytical=bp_ana,
    )


def sweep_kc(rng, max_workers=8):
    D, R, H, a = 10, 7.5, 10.0, 5
    dt, C0, patch_radius = 1e-6, 10, 0.1

    L = R - a
    tau_diff = L**2 / (np.pi**2 * D)
    t_relax = 30 * tau_diff
    T = t_relax + 2

    _run_sweep(
        rng, max_workers, "kc",
        varied_values=np.array([1e-1, 0.5, 1, 2, 5]),
        D=D, R=R, H=H, dt=dt, a=a, C0=C0, patch_radius=patch_radius,
        t_relax=t_relax, T=T,
        ka=math.inf, kb=0.0, kc=None,
        patch_list=np.arange(100, 6000, 800, dtype=int),
    )


def sweep_kb(rng, max_workers=8):
    D, R, H, a = 10, 7.5, 10.0, 5
    dt, C0, patch_radius = 1e-5, 5, 0.1

    L = R - a
    tau_diff = L**2 / (np.pi**2 * D)
    t_relax = 15 * tau_diff
    T = t_relax + 1

    _run_sweep(
        rng, max_workers, "kb",
        varied_values=np.array([1, 10, 50, 100]),
        D=D, R=R, H=H, dt=dt, a=a, C0=C0, patch_radius=patch_radius,
        t_relax=t_relax, T=T,
        ka=math.inf, kb=None, kc=5,
        patch_list=np.arange(100, 6000, 800, dtype=int),
    )


def sweep_ka(rng, max_workers=8):
    D, R, H, a = 10, 7.5, 10.0, 5
    dt, C0, patch_radius = 1e-6, 10, 0.1

    L = R - a
    tau_diff = L**2 / (np.pi**2 * D)
    t_relax = 30 * tau_diff
    T = t_relax + 2

    _run_sweep(
        rng, max_workers, "ka",
        varied_values=np.array([1e-1, 1, 2, 100]),
        D=D, R=R, H=H, dt=dt, a=a, C0=C0, patch_radius=patch_radius,
        t_relax=t_relax, T=T,
        ka=None, kb=5, kc=5,
        patch_list=np.arange(100, 6000, 800, dtype=int),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(int(time.time()))
    sweep_kb(rng, 8)
