"""Simulation parameters for the cylinder-in-cylinder chemoreception model."""
from dataclasses import dataclass
import math


@dataclass
class SimulationParams:
    """All parameters for a single simulation run.

    Physical picture
    ----------------
    The geometry is a cylindrical annulus: an inner solid cylinder (the
    capillary/nucleus, radius ``a``) surrounded by an outer hollow cylinder
    (the tissue cell, radius ``R``), both of height ``H``.  Insulin molecules
    diffuse freely in the annular gap ``a <= r <= R``.  Receptor patches sit
    on the *outer* cylinder wall (at ``r = R``).

    The molecule tries to reach the outer wall, bind to a receptor patch, and
    get internalized.  The three-step kinetics:
        1. Diffuse to a patch and bind  (rate constant ``ka``)
        2. Bound receptor dissociates back to free  (rate constant ``kb``)
        3. Bound receptor internalizes the molecule  (rate constant ``kc``)

    Units: lengths in µm, time in s, concentration in molecules/µm³.

    Fields
    ------
    N : int
        Number of simulation particles.  In steady-state simulations this is
        kept constant by respawning absorbed particles; in transient simulations
        it counts down as particles are absorbed.  Must be >= 1.

    C0 : float
        Target bulk concentration (molecules/µm³).  Used for logging and for
        computing the correct ``N`` via ``AnalyticalModel.n_total(C0)`` in
        worker scripts.  Not enforced by the simulation itself.

    D : float
        Diffusion coefficient (µm²/s).  Governs the Gaussian step size:
        each coordinate displacement per step ~ N(0, sqrt(2*D*dt)).
        Typical insulin value: 125 µm²/s in aqueous; lower in tissue.

    a : float
        Radius of the inner cylinder (µm).  Defines the reflective inner
        boundary: any particle that diffuses inside ``r < a`` is reflected
        specularly back to ``r = a``.

    R : float
        Radius of the outer cylinder (µm).  Receptor patches live on this
        surface.  Must satisfy ``R > a``.

    H : float
        Height of the cylinder (µm).  The z-coordinate runs in ``[-H/2, H/2]``
        with *periodic* boundary conditions — a particle leaving through the
        top reappears at the bottom, modelling an infinitely periodic stack.

    dt : float
        Simulation time step (s).  Should be small enough that the diffusion
        step ``sqrt(2*D*dt)`` is much smaller than the patch radius ``s``;
        otherwise patches can be jumped over.

    T : float
        Total simulation time (s).  Should be at least ``t_relax + T_measure``
        where ``T_measure`` is long enough to collect statistically good flux
        estimates.

    t_relax : float
        Equilibration (relaxation) time (s) before flux measurements begin.
        The particle cloud needs time to reach a steady-state concentration
        profile ``C(r) ~ ln(r/a)`` before we start counting absorptions.
        Typical choice: ``10--30 * tau_diff`` where
        ``tau_diff = (R-a)^2 / (pi^2 * D)``.

    n_patches : int
        Number of receptor patches on the outer cylinder surface.  Set to 0
        for a fully absorbing outer wall (every outer-wall hit is absorbed).

    patch_radius : float or None
        Radius of each circular patch as measured on the *unrolled* cylinder
        surface (µm).  The unrolled surface has dimensions
        ``2*pi*R`` (circumference) × ``H`` (height), treated as a flat
        rectangle with periodic boundaries.  None if ``n_patches = 0``.

    n_sites_per_patch : int
        Number of independent binding sites per patch.  Affects only the
        ``n_bound`` / ``bound_fraction`` calculations in ``AnalyticalModel``;
        the stochastic simulation treats each patch as having one site (one
        particle captured at a time).

    ka : float
        Association rate constant (µm³/s) for a molecule hitting a patch.
        Controls how quickly a molecule that is *touching* the patch actually
        binds.  ``math.inf`` means perfectly absorbing (every patch contact
        leads to binding, Berg-Purcell limit).  The Erban-Chapman binding
        probability formula converts this into a per-collision probability:
        ``p_bind = (kappa * sqrt(pi*dt/D)) / (1 + kappa*sqrt(pi*dt/D)/2)``
        where ``kappa = ka / (pi * s^2)``.

    kb : float
        Dissociation rate (s⁻¹).  A bound molecule detaches from the receptor
        and returns to the outer bulk (near ``r = R``) with rate ``kb``.
        0.0 means no dissociation — once bound, the molecule stays bound
        until internalized.

    kc : float
        Internalization rate (s⁻¹).  A bound molecule is transported through
        the outer membrane and deposited near the inner wall (``r ≈ a``) with
        rate ``kc``.  In the steady-state simulation, internalized particles
        are respawned near ``r = a``; in the transient simulation they are
        removed permanently.  ``math.inf`` means instant internalization.

    eps_c0 : float
        Thickness (µm) of the two thin shells near ``r = a`` used to
        extrapolate the bulk concentration ``C0`` from the simulation.
        The simulation measures ``<C>`` in ``[a, a+eps_c0]`` and
        ``[a+2*eps_c0, a+3*eps_c0]`` and linearly extrapolates to ``r = a``
        to estimate the "source" concentration.

    eps_respawn : float
        Thickness (µm) of the shell near ``r = a`` where internalized
        (absorbed) particles are respawned.  Respawning in a thin shell rather
        than exactly at ``r = a`` avoids numerical issues with the reflecting
        boundary condition.

    eps_out : float
        Thickness (µm) of the shell near ``r = R`` used to measure the local
        concentration at the outer wall.  Also used as the placement shell
        when dissociated particles are returned to the outer bulk.
    """

    # Number of simulation particles (required)
    N: int

    # Target bulk concentration -- used for logging and N_total calculations
    C0: float = 0.0

    # Diffusion coefficient (µm²/s)
    D: float = 0.0

    # Geometry
    a: float = 0.0          # inner cylinder radius (µm)
    R: float = 0.0          # outer cylinder radius (µm)
    H: float = 0.0          # cylinder height (µm)

    # Time stepping
    dt: float = 0.0         # time step (s)
    T: float = 0.0          # total simulation time (s)
    t_relax: float = 1.0    # equilibration time before measurements (s)

    # Receptor patches on outer cylinder
    n_patches: int = 0
    patch_radius: float | None = None   # patch radius in unrolled coords (µm)
    n_sites_per_patch: int = 1          # receptor sites per patch

    # Binding kinetics
    ka: float = math.inf    # association rate (µm³/s); inf = perfectly absorbing
    kb: float = 0.0         # dissociation rate -- release to outer bulk (s⁻¹)
    kc: float = math.inf    # internalization rate -- release to inner bulk (s⁻¹)

    # Numerical tolerances (µm)
    eps_c0: float = 0.1       # shell thickness for C0 extrapolation near r=a
    eps_respawn: float = 0.1  # shell thickness for respawning internalized particles
    eps_out: float = 0.1      # shell thickness for outer-wall concentration measurement
