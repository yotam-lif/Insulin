import math
import numpy as np

# ------------------------------------------------------------
# 1) mu(C0): your "new" expression (unchanged)
# ------------------------------------------------------------
def mu_from_c0_new(
    C0: float,
    *,
    D: float,
    a: float,
    R: float,
    H: float,
    n_patches: int,
    patch_radius: float,
    ka: float,
    kb: float,
    kc: float
) -> float:
    """
    New mu(C0):
      ka_star = 4 D s / (1 + (4 D s)/ka)   (or 4Ds if ka=inf)
      Kd = (kb+kc)/ka_star
      denom_geom = R ln(R/a)
      sigma0 = n_patches / (2π R H)

      A = D (C0+Kd)/denom_geom + sigma0 kc
      B = 4 (D C0/denom_geom) sigma0 kc
      C = 2 (D C0/denom_geom)

      mu = (A - sqrt(A^2 - B)) / C
    """
    if C0 <= 0:
        raise ValueError("C0 must be > 0")

    L = math.log(R / a)
    if L <= 0:
        raise ValueError("Need R > a")

    denom_geom = R * L
    sigma0 = n_patches / (2.0 * math.pi * R * H)

    # ka_star handling
    if math.isinf(ka):
        ka_star = 4.0 * D * patch_radius
    else:
        if ka <= 0:
            raise ValueError("ka must be positive (or inf)")
        ka_star = (4.0 * D * patch_radius) / (1.0 + (4.0 * D * patch_radius) / ka)

    if ka_star <= 0:
        raise ValueError("ka_star computed non-positive; check parameters")

    Kd = (kb + kc) / ka_star

    A = D * (C0 + Kd) / denom_geom + sigma0 * kc
    B = 4.0 * (D * C0 / denom_geom) * sigma0 * kc
    C = 2.0 * (D * C0 / denom_geom)

    disc = A * A - B
    if disc < 0:
        if disc > -1e-14 * max(1.0, A * A):
            disc = 0.0
        else:
            raise ValueError(f"Negative discriminant A^2-B={disc:g}; params inconsistent.")

    mu = (A - math.sqrt(disc)) / C
    return float(mu)


# ------------------------------------------------------------
# 2) Bulk particles from C0 via your cylindrical profile integral
# ------------------------------------------------------------
def total_particles_from_c0(
    C0: float,
    *,
    H: float,
    a: float,
    R: float,
    mu_func
) -> float:
    """
    Computes N_bulk(C0) = ∫_V C(r) dV for
      C(r)=C0*(1 - mu(C0)*ln(r/a)/ln(R/a)),
    with V being the cylindrical shell a<=r<=R, 0<=z<=H.
    """
    L = math.log(R / a)
    if L <= 0:
        raise ValueError("Need R > a.")

    mu = float(mu_func(C0))

    # Derived bracket:
    bracket = (R**2 - a**2)/2.0 - (mu / (4.0*L)) * ((2.0*L - 1.0)*R**2 + a**2)

    if bracket <= 0:
        return float("nan")

    N_bulk = 2.0 * math.pi * H * C0 * bracket
    return float(N_bulk)


# ------------------------------------------------------------
# 3) NEW: MM/occupancy estimate of bound particles
# ------------------------------------------------------------
def n_bound_from_mm(
    C0: float,
    *,
    n_patches: int,
    n_sites_per_patch: int,
    D: float,
    a: float,
    R: float,
    H: float,
    patch_radius: float,
    ka: float,
    kb: float,
    kc: float,
) -> float:
    """
    Steady-state bound-site estimate consistent with your derivation:

      C(R) = C0 * (1 - mu(C0))
      koff = kb + kc
      ka_star = 4 D s / (1 + (4 D s)/ka)   (or 4Ds if ka=inf)
      Kd = koff / ka_star

      f_bound = C(R) / (C(R) + Kd)
      N_bound = Rtot * f_bound

    IMPORTANT: C0 units must match Kd units.
    """
    if C0 <= 0:
        return 0.0

    Rtot = float(n_patches) * float(n_sites_per_patch)
    if Rtot <= 0:
        return 0.0

    koff = float(kb + kc)
    if koff <= 0:
        # irreversible binding -> all sites fill
        return Rtot

    # ka_star (same as mu_from_c0_new convention)
    if math.isinf(ka):
        ka_star = 4.0 * D * patch_radius
    else:
        if ka <= 0:
            raise ValueError("ka must be positive (or inf)")
        ka_star = (4.0 * D * patch_radius) / (1.0 + (4.0 * D * patch_radius) / ka)

    if ka_star <= 0:
        raise ValueError("ka_star computed non-positive; check parameters")

    Kd = koff / ka_star

    # --- use C(R) not C0 ---
    mu = mu_from_c0_new(
        C0,
        D=D, a=a, R=R, H=H,
        n_patches=n_patches,
        patch_radius=patch_radius,
        ka=ka, kb=kb, kc=kc
    )
    C_R = C0 * (1.0 - mu)
    if C_R < 0:
        C_R = 0.0  # numerical safety

    f_bound = C_R / (C_R + Kd)
    return float(Rtot * f_bound)


# ------------------------------------------------------------
# 4) TOTAL N from bulk C0: N_total = N_bulk + N_bound
# ------------------------------------------------------------
def N_from_c0_new(
    C0: float,
    *,
    H: float,
    a: float,
    R: float,
    D: float,
    n_patches: int,
    patch_radius: float,
    ka: float,
    kb: float,
    kc: float,
    n_sites_per_patch: int = 1,
) -> float:
    """
    Given target BULK C0, return total particles:
      N_total(C0) = N_bulk(C0) + N_bound(C0)

    - N_bulk uses your mu(C0) profile integral.
    - N_bound uses MM/occupancy estimate.
    """
    def mu_func(c0):
        return mu_from_c0_new(
            c0,
            D=D, a=a, R=R, H=H,
            n_patches=n_patches,
            patch_radius=patch_radius,
            ka=ka, kb=kb, kc=kc
        )

    N_bulk = total_particles_from_c0(C0, H=H, a=a, R=R, mu_func=mu_func)
    if not np.isfinite(N_bulk):
        raise ValueError(
            "Got non-finite N_bulk. This usually means mu(C0) makes the bulk integral "
            "invalid (bracket <= 0). Check parameters/units or whether the assumed C(r) "
            "form is valid."
        )

    N_bound = n_bound_from_mm(
        C0,
        n_patches=n_patches,
        n_sites_per_patch=n_sites_per_patch,
        D=D,
        a=a, R=R, H=H,  # <-- add these
        patch_radius=patch_radius,
        ka=ka, kb=kb, kc=kc,
    )

    return float(N_bulk + N_bound)


# ------------------------------------------------------------
# 5) OPTIONAL but recommended: inverse solver that matches the new N_total
#    (so it's consistent both directions)
# ------------------------------------------------------------
def c0_from_total_particles_new(
    N: float,
    *,
    H: float,
    a: float,
    R: float,
    D: float,
    n_patches: int,
    patch_radius: float,
    ka: float,
    kb: float,
    kc: float,
    n_sites_per_patch: int = 1,
    # solver controls
    c0_min: float = 1e-30,
    c0_max: float = 1e6,
    rtol: float = 1e-10,
    max_iter: int = 200
) -> float:
    """
    Solve for BULK C0 given total particles N_total, using:
      N_total(C0) = N_bulk(C0) + N_bound(C0)

    Uses bracketing + bisection (robust).
    """
    if N <= 0:
        raise ValueError("N must be > 0")

    def mu_func(C0):
        return mu_from_c0_new(
            C0,
            D=D, a=a, R=R, H=H,
            n_patches=n_patches,
            patch_radius=patch_radius,
            ka=ka, kb=kb, kc=kc
        )

    def N_total_from_C0(C0):
        N_bulk = total_particles_from_c0(C0, H=H, a=a, R=R, mu_func=mu_func)
        if not np.isfinite(N_bulk):
            return float("nan")
        N_bound = n_bound_from_mm(
            C0,
            n_patches=n_patches,
            n_sites_per_patch=n_sites_per_patch,
            D=D,
            a=a, R=R, H=H,  # <-- add these
            patch_radius=patch_radius,
            ka=ka, kb=kb, kc=kc,
        )
        return N_bulk + N_bound

    def f(C0):
        return N_total_from_C0(C0) - N

    lo = float(c0_min)
    hi = float(c0_max)

    flo = f(lo)
    if not np.isfinite(flo):
        lo = max(lo, 1e-24)
        flo = f(lo)

    fhi = f(hi)
    expand = 0
    while (not np.isfinite(fhi) or np.sign(flo) == np.sign(fhi)) and expand < 80:
        hi *= 10.0
        fhi = f(hi)
        expand += 1

    if not np.isfinite(flo) or not np.isfinite(fhi) or np.sign(flo) == np.sign(fhi):
        raise ValueError(
            "Failed to bracket a root for C0. "
            "Try increasing c0_max (or check units/parameters)."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)

        if not np.isfinite(fmid):
            mid = np.nextafter(mid, hi)
            fmid = f(mid)

        if abs(hi - lo) <= rtol * max(1.0, abs(mid)):
            return float(mid)

        if np.sign(fmid) == np.sign(flo):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid

    return float(0.5 * (lo + hi))