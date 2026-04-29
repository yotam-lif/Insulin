"""Experiment: transient survival fraction — patchy vs fully absorbing."""
import time
import numpy as np
from insulin import SimulationParams, TransientSimulation
from insulin.analytical.transient import TransientAnalytical
from insulin.plotting import plot_survival_comparison


def run(rng):
    D = 25
    R = 7.5
    H = 10.0
    dt = 1e-6
    T = 0.05
    N = 1000
    a = 5
    patch_radius = 0.05
    record_stride = 100
    n_modes = 75
    p0 = "uniform_volume"

    # --- Patchy (Robin) ---
    params = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=5000, patch_radius=patch_radius, eps_respawn=1e-10,
    )
    k_eff = (2 * params.n_patches * params.patch_radius) / (params.H * np.pi)

    sim_patchy = TransientSimulation(params, rng)
    res_patchy = sim_patchy.run(record_stride=record_stride)
    t = res_patchy["t"]
    rw_patchy = res_patchy["frac_alive"]

    ana = TransientAnalytical(params, n_modes=n_modes)
    ana_patchy = ana.survival_fraction(t, bc="robin", k=k_eff, p0=p0, eps=params.eps_respawn)

    # --- Full absorbing (Dirichlet) ---
    params_full = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=0, patch_radius=None, eps_respawn=1e-10,
    )

    sim_full = TransientSimulation(params_full, rng)
    res_full = sim_full.run(record_stride=record_stride)
    t_full = res_full["t"]
    rw_full = res_full["frac_alive"]

    if not np.array_equal(t_full, t):
        rw_full = np.interp(t, t_full, rw_full)

    ana_full_solver = TransientAnalytical(params_full, n_modes=n_modes)
    ana_full = ana_full_solver.survival_fraction(t, bc="dirichlet", p0=p0, eps=params_full.eps_respawn)

    plot_survival_comparison(
        t, rw_patchy, rw_full, ana_patchy, ana_full,
        params, k_eff,
        savepath="survival_fraction_comparison.png",
        title="Cylinder survival: RW vs analytical",
    )


if __name__ == "__main__":
    rng = np.random.default_rng(int(time.time()))
    run(rng)
