"""Parallel worker functions for experiment parameter sweeps.

Each function is a top-level (picklable) callable for use with
``concurrent.futures.ProcessPoolExecutor``.  Each worker accepts a single
flat tuple of arguments, runs one complete simulation, and returns a small
tuple of scalar results.  Keeping everything in a flat tuple (rather than a
dict or object) avoids pickling issues across process boundaries.

Usage pattern::

    args_list = [(param1, param2, ..., seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker_function, args_list))
    results.sort(key=lambda x: x[0])   # sort by the first return value
"""
import math
import numpy as np
from classes.params import SimulationParams
from stochastic_sim.steady_state import SteadyStateSimulation
from analytical.steady_state import AnalyticalModel


def worker_rate_vs_radius(args):
    """Simulate the fully-absorbing flux rate constant for a single inner radius ``a``.

    Purpose
    -------
    Validates that the simulation recovers the analytical result
    ``k = 2πHD / ln(R/a)`` for a completely absorbing outer wall (no patches).
    Run for a range of ``a`` values to verify the log-distance dependence.

    Algorithm
    ---------
    1. Set the relaxation time adaptively: ``t_relax = 5 * tau_diff`` where
       ``tau_diff = (R-a)² / (π²D)`` is the diffusion time across the gap.
    2. Run a steady-state simulation with no patches (``n_patches=0``), which
       means every outer-wall hit is absorbed.
    3. Return the simulated rate constant ``k = rate / c0_measured`` alongside
       the analytical value.

    Input tuple
    -----------
    ``(a, D, R, H, dt, T, N, seed)``

    a : float
        Inner cylinder radius to test (µm).
    D : float
        Diffusion coefficient (µm²/s).
    R : float
        Outer cylinder radius (µm).
    H : float
        Cylinder height (µm).
    dt : float
        Time step (s).
    T : float
        Total run time (s) — overridden internally to ``2 * t_relax``.
    N : int
        Number of simulation particles.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (a, k_sim, k_theory)

    a : float
        Inner radius (echoed back for sorting after parallel execution).
    k_sim : float
        Simulated rate constant ``rate / c0_measured`` (µm³/s).
    k_theory : float
        Analytical result ``2πHD / ln(R/a)`` (µm³/s).
    """
    a, D, R, H, dt, T, N, seed = args
    rng = np.random.default_rng(seed)

    # Set relaxation time to 5 diffusion times across the gap
    L = R - a
    tau_diff = (L**2) / (np.pi**2 * D)
    t_relax = 5.0 * tau_diff
    T = 2.0 * t_relax

    params = SimulationParams(N=N, D=D, a=a, R=R, H=H, dt=dt, T=T, t_relax=t_relax)
    result = SteadyStateSimulation(params, rng).run()

    k_theory = 2.0 * np.pi * H * D / np.log(R / a)
    return (a, result.rate_normalized, k_theory)


def worker_absorbing_patches(args):
    """Simulate flux for a single patch count with perfectly absorbing patches.

    Purpose
    -------
    Validates the Berg-Purcell formula for patchy absorption::

        mu = N*s / (N*s + pi*H / (2*ln(R/a)))

    where ``N`` is the number of patches and ``s`` is the patch radius.  The
    formula predicts that the uptake efficiency saturates as patch coverage
    increases, with the effective receptor "aperture" competing against the
    diffusion resistance of the gap.

    Algorithm
    ---------
    1. Run a steady-state simulation with the specified patch count and
       perfectly absorbing kinetics (``ka=inf``, ``kb=0``, ``kc=inf`` via
       default ``SimulationParams``).
    2. Normalise the simulated rate by ``k_full = 2πHD/ln(R/a)`` (the rate for
       a fully absorbing wall) to get ``J/J_max``.
    3. Compute the analytical Berg-Purcell prediction via ``AnalyticalModel.mu()``.
       Note: ``mu()`` is independent of ``C0`` in the perfectly-absorbing limit
       (``kb=0``, ``kc=inf``), so we pass ``C0=1`` as a dummy value.

    Input tuple
    -----------
    ``(n_patches, t_relax, D, R, H, dt, T, N, a, patch_radius, seed)``

    Returns
    -------
    (n_patches, sim_ratio, analytical_solution, coverage)

    n_patches : int
        Number of patches (echoed back for sorting).
    sim_ratio : float
        Simulated ``J/J_max``.
    analytical_solution : float
        Berg-Purcell prediction for ``J/J_max``.
    coverage : float
        Fractional patch coverage ``n * s² / (2*R*H)`` — the fraction of the
        outer cylinder surface occupied by patches.
    """
    (n_patches, t_relax, D, R, H, dt, T, N, a, patch_radius, seed) = args
    rng = np.random.default_rng(seed)

    params = SimulationParams(
        N=N, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius, t_relax=t_relax,
    )
    result = SteadyStateSimulation(params, rng).run()

    k_full = 2.0 * np.pi * H * D / np.log(R / a)
    sim_ratio = result.rate_normalized / k_full
    coverage = (n_patches * patch_radius**2) / (2.0 * R * H)

    # mu() with C0=1 gives the Berg-Purcell limit (C0-independent when kb=0, kc=inf)
    analytical_solution = AnalyticalModel(params).mu(1.0)

    return n_patches, sim_ratio, analytical_solution, coverage


def worker_kinetics_steady(args):
    """Simulate flux for one (patch count, kinetics) combination at target concentration C0.

    Purpose
    -------
    Validates the general finite-kinetics formula for ``mu(C0)``, which applies
    when both dissociation (``kb``) and internalization (``kc``) are finite.
    Unlike the Berg-Purcell limit, ``mu`` now depends on ``C0`` through the
    receptor occupancy.

    Why we compute N from C0 analytically
    --------------------------------------
    The simulation requires a fixed integer particle count ``N``, but the
    experiment is specified by a target concentration ``C0``.  The total number
    of particles includes both free particles in the bulk *and* bound receptors::

        N_total = N_bulk(C0) + N_bound(C0)

    ``AnalyticalModel.n_total(C0)`` computes this self-consistently via the
    analytical steady-state solution.  We use the analytical ``N_total`` to
    initialise the simulation so that the actual simulated ``C0`` (measured
    near ``r=a``) matches the target.

    Input tuple
    -----------
    ``(n_patches, t_relax, D, R, H, dt, T, C0, a, patch_radius, ka, kb, kc, seed)``

    n_patches : int
        Number of receptor patches.
    t_relax : float
        Relaxation time before measurements (s).
    D : float
        Diffusion coefficient (µm²/s).
    R, H : float
        Outer cylinder radius and height (µm).
    dt : float
        Time step (s).
    T : float
        Total simulation time (s).
    C0 : float
        Target bulk concentration (molecules/µm³).
    a : float
        Inner cylinder radius (µm).
    patch_radius : float
        Patch radius (µm).
    ka : float
        Association rate (µm³/s).
    kb : float
        Dissociation rate (s⁻¹).
    kc : float
        Internalization rate (s⁻¹).
    seed : int
        RNG seed.

    Returns
    -------
    (n_patches, sim_ratio, analytical_solution)

    n_patches : int
        Echoed back for sorting after parallel execution.
    sim_ratio : float
        Simulated ``J/J_max`` using the measured ``c0_measured`` as the
        reference concentration.
    analytical_solution : float
        Analytical ``mu(c0_measured)`` for direct comparison.
    """
    (n_patches, t_relax, D, R, H, dt, T, C0, a, patch_radius,
     ka, kb, kc, seed) = args
    rng = np.random.default_rng(seed)

    # Step 1: Determine the correct total particle count for the target C0.
    # We build a template params with N=1 (not used in the n_total calculation)
    # to get the geometry and kinetics right.
    params_template = SimulationParams(
        N=1, C0=C0, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius,
        t_relax=t_relax, ka=ka, kb=kb, kc=kc,
    )
    N_total = AnalyticalModel(params_template).n_total(C0)

    # Step 2: Run the simulation with the correct particle count
    params = SimulationParams(
        N=int(N_total), C0=C0, D=D, a=a, R=R, H=H, dt=dt, T=T,
        n_patches=n_patches, patch_radius=patch_radius,
        t_relax=t_relax, ka=ka, kb=kb, kc=kc,
    )
    result = SteadyStateSimulation(params, rng).run()

    # Normalise simulated rate by the fully-absorbing rate constant
    k_full = 2.0 * np.pi * H * D / np.log(R / a)
    sim_ratio = result.rate_normalized / k_full

    # Evaluate the analytical model at the *measured* C0 (not the target C0)
    # so the comparison is self-consistent with what the simulation actually ran at
    analytical_solution = AnalyticalModel(params).mu(result.c0_measured)

    return n_patches, sim_ratio, analytical_solution
