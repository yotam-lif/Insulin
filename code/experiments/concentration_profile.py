"""Experiment: measure radial concentration profile in steady state."""
import math
import time
import numpy as np
from insulin import SimulationParams, SteadyStateSimulation
from insulin.plotting import plot_concentration_profiles


def run(rng):
    D = 10
    R = 7.5
    H = 10.0
    dt = 1e-5
    N = 1000
    a = 5
    patch_radius = 0.1

    L = R - a
    tau_diff = L**2 / (np.pi**2 * D)
    t_relax = 10 * tau_diff
    T = t_relax + 4

    params = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T, t_relax=t_relax,
        n_patches=1000, patch_radius=patch_radius,
        eps_respawn=1e-10, ka=math.inf, kb=0, kc=1,
    )

    sim = SteadyStateSimulation(params, rng)
    result = sim.run(concentration_profile=True)

    plot_concentration_profiles(result.r_centers, result.concentrations, result.profile_times)


if __name__ == "__main__":
    rng = np.random.default_rng(int(time.time()))
    run(rng)
