"""Parallel worker functions for experiment parameter sweeps.

Each worker is a top-level function (picklable) that constructs params,
runs a simulation, and returns results for one parameter combination.
"""
import math
import numpy as np
from insulin.params import SimulationParams
from insulin.simulation.steady_state import SteadyStateSimulation
from insulin.analytical.steady_state import AnalyticalModel


def worker_rate_vs_radius(args):
    a, D, R, H, dt, T, N, seed = args
    rng = np.random.default_rng(seed)

    L = R - a
    tau_diff = (L ** 2) / (np.pi ** 2 * D)
    t_relax = 5 * tau_diff
    T = 2 * t_relax

    params = SimulationParams(N=N, D=D, a=a, R=R, H=H, dt=dt, T=T, t_relax=t_relax)
    sim = SteadyStateSimulation(params, rng)
    result = sim.run()

    k_sim = result.rate_normalized
    k_theory = 2.0 * np.pi * H * D / np.log(R / a)
    return (a, k_sim, k_theory)


def worker_absorbing_patches(args):
    (n_patches, t_relax, D, R, H, dt, T, N, a, patch_radius, seed) = args
    rng = np.random.default_rng(seed)

    params = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius, t_relax=t_relax,
    )
    sim = SteadyStateSimulation(params, rng)
    result = sim.run()

    k_full = 2.0 * np.pi * H * D / np.log(R / a)
    sim_ratio = result.rate_normalized / k_full
    coverage = (n_patches * patch_radius**2) / (2.0 * R * H)

    analytical = AnalyticalModel(params)
    analytical_solution = analytical.mu(1.0)  # mu is independent of C0 for perfect sinks

    return n_patches, sim_ratio, analytical_solution, coverage


def worker_kinetics_steady(args):
    (n_patches, t_relax, D, R, H, dt, T,
     C0, a, patch_radius, ka, kb, kc, seed) = args
    rng = np.random.default_rng(seed)

    params = SimulationParams(
        N=1, C0=C0, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius,
        t_relax=t_relax, ka=ka, kb=kb, kc=kc,
    )

    analytical = AnalyticalModel(params)
    N_total = analytical.n_total(C0)

    params = SimulationParams(
        N=int(N_total), C0=C0, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius,
        t_relax=t_relax, ka=ka, kb=kb, kc=kc,
    )

    sim = SteadyStateSimulation(params, rng)
    result = sim.run()

    k_full = 2.0 * np.pi * H * D / np.log(R / a)
    sim_ratio = result.rate_normalized / k_full

    C0_meas = result.c0_measured
    analytical_solution = analytical.mu(C0_meas)

    return n_patches, sim_ratio, analytical_solution
