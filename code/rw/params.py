from dataclasses import dataclass
import math

@dataclass
class RWCylParams:
    N: int
    C0: float
    D: float
    a: float
    R: float
    H: float
    dt: float
    T: float

    t_relax: float = 1
    n_r_segments: int = 30
    n_patches: int = 0
    patch_radius: float | None = None
    eps_c0: float = 0.1
    eps_respawn: float = 0.1
    #init_thickness: float = 0.2

    n_sites_per_patch: int = 1
    use_kon_koff: bool = False
    p_bind: float = 1
    koff: float = 0
    ka: float = math.inf
    kb: float= 0
    kc: float = math.inf
    eps_out: float = 0.1



