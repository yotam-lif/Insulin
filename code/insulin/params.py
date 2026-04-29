from dataclasses import dataclass
import math


@dataclass
class SimulationParams:
    """Parameters for the cylinder-in-cylinder chemoreception model.

    Geometry: inner cylinder (radius a) and outer cylinder (radius R),
    height H, with diffusion in the annular gap.

    Kinetics: receptors (patches) on the outer cylinder with association (ka),
    dissociation (kb), and internalization (kc) rates.
    """
    # Particle / concentration
    N: int
    C0: float = 0.0

    # Diffusion
    D: float = 0.0

    # Geometry
    a: float = 0.0       # inner cylinder radius
    R: float = 0.0       # outer cylinder radius
    H: float = 0.0       # cylinder height

    # Time stepping
    dt: float = 0.0
    T: float = 0.0
    t_relax: float = 1.0

    # Radial binning
    n_r_segments: int = 30

    # Patches / receptors
    n_patches: int = 0
    patch_radius: float | None = None
    n_sites_per_patch: int = 1

    # Kinetic rates
    ka: float = math.inf   # association rate
    kb: float = 0.0        # dissociation (release to outer)
    kc: float = math.inf   # internalization (release to inner)

    # Numerical tolerances
    eps_c0: float = 0.1       # shell thickness for C0 extrapolation
    eps_respawn: float = 0.1  # shell thickness for respawning
    eps_out: float = 0.1      # shell thickness for outer concentration

    # Legacy compatibility
    use_kon_koff: bool = False
    p_bind: float = 1.0
    koff: float = 0.0
