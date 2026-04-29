"""Analytical steady-state solutions for the cylinder-in-cylinder model.

Consolidates all theory: uptake efficiency (mu), flux ratios, concentration
profiles, particle counts, and receptor occupancy.
"""
import math
import numpy as np
from ..params import SimulationParams


AVOGADRO = 6.02214076e23


def molar_to_molecules_per_um3(c_molar: float) -> float:
    """Convert concentration from M (mol/L) to molecules/um^3."""
    return c_molar * AVOGADRO / 1e15


class AnalyticalModel:
    """Analytical solutions for the steady-state chemoreception model.

    Constructed from SimulationParams, provides methods for:
    - Uptake efficiency mu(C0)
    - Flux and flux ratios
    - Concentration profiles C(r)
    - Particle counts (bulk + bound)
    - Receptor occupancy
    """

    def __init__(self, params: SimulationParams):
        self.p = params

    @property
    def ka_star(self) -> float:
        """Effective association rate: ka* = 4Ds / (1 + 4Ds/ka)."""
        s = self.p.patch_radius
        D = self.p.D
        if s is None or s <= 0:
            return 0.0
        if math.isinf(self.p.ka):
            return 4.0 * D * s
        if self.p.ka <= 0:
            raise ValueError("ka must be positive or inf")
        return (4.0 * D * s) / (1.0 + (4.0 * D * s) / self.p.ka)

    @property
    def kd(self) -> float:
        """Dissociation constant Kd = (kb + kc) / ka*."""
        ka_s = self.ka_star
        if ka_s <= 0:
            return float('inf')
        return (self.p.kb + self.p.kc) / ka_s

    @property
    def k_max(self) -> float:
        """Fully-absorbing rate constant: 2*pi*H*D / ln(R/a)."""
        return 2.0 * math.pi * self.p.H * self.p.D / math.log(self.p.R / self.p.a)

    @property
    def sigma0(self) -> float:
        """Receptor surface density on the outer cylinder."""
        return self.p.n_patches / (2.0 * math.pi * self.p.R * self.p.H)

    def mu(self, C0: float) -> float:
        """Uptake efficiency mu(C0) = J/J_max.

        For perfectly absorbing patches (kb=0, kc=inf):
            mu = Ns / (Ns + pi*H / (2*ln(R/a)))

        For finite kinetics, solves the quadratic from flux balance.
        """
        if C0 <= 0:
            raise ValueError("C0 must be > 0")

        p = self.p
        L = math.log(p.R / p.a)
        if L <= 0:
            raise ValueError("Need R > a")

        # Perfect-sink limit (Berg-Purcell)
        if p.kb == 0 and math.isinf(p.kc):
            Ns = p.n_patches * p.patch_radius
            return Ns / (Ns + p.H * math.pi / (2.0 * L))

        # Finite kinetics: quadratic solution
        denom_geom = p.R * L
        ka_s = self.ka_star
        if ka_s <= 0:
            raise ValueError("ka_star is non-positive; check parameters")

        Kd = (p.kb + p.kc) / ka_s
        sigma0 = self.sigma0

        A = p.D * (C0 + Kd) / denom_geom + sigma0 * p.kc
        B = 4.0 * (p.D * C0 / denom_geom) * sigma0 * p.kc
        C = 2.0 * (p.D * C0 / denom_geom)

        disc = A * A - B
        if disc < 0:
            if disc > -1e-14 * max(1.0, A * A):
                disc = 0.0
            else:
                raise ValueError(f"Negative discriminant A^2-B={disc:g}; params inconsistent.")

        return float((A - math.sqrt(disc)) / C)

    def concentration_at_r(self, r: np.ndarray, C0: float) -> np.ndarray:
        """Radial concentration profile: C(r) = C0 * (1 - mu * ln(r/a) / ln(R/a))."""
        r = np.asarray(r, dtype=float)
        m = self.mu(C0)
        L = math.log(self.p.R / self.p.a)
        return C0 * (1.0 - m * np.log(r / self.p.a) / L)

    def concentration_at_R(self, C0: float) -> float:
        """Concentration at the outer cylinder surface: C(R) = C0*(1 - mu)."""
        return C0 * (1.0 - self.mu(C0))

    def flux_ratio(self, C0: float) -> float:
        """Normalized flux J/J_max = mu(C0)."""
        return self.mu(C0)

    def flux(self, C0: float) -> float:
        """Total uptake rate J = mu * k_max * C0."""
        return self.mu(C0) * self.k_max * C0

    def bound_fraction(self, C0: float) -> float:
        """Steady-state receptor occupancy theta = C(R) / (C(R) + Kd)."""
        C_R = self.concentration_at_R(C0)
        Kd = self.kd
        if C_R < 0:
            C_R = 0.0
        return C_R / (C_R + Kd)

    def bound_percent(self, C0: float) -> float:
        """Bound receptor percentage (0-100)."""
        return 100.0 * self.bound_fraction(C0)

    def n_bound(self, C0: float) -> float:
        """Number of bound receptors at steady state (Michaelis-Menten)."""
        n_total_sites = self.p.n_patches * self.p.n_sites_per_patch
        return n_total_sites * self.bound_fraction(C0)

    def n_bulk(self, C0: float) -> float:
        """Number of free particles in the annular volume.

        Integrates C(r) over the cylindrical shell:
        N_bulk = 2*pi*H * C0 * [(R^2-a^2)/2 - mu/(4*ln(R/a)) * ((2*ln(R/a)-1)*R^2 + a^2)]
        """
        p = self.p
        L = math.log(p.R / p.a)
        m = self.mu(C0)
        bracket = (p.R**2 - p.a**2) / 2.0 - (m / (4.0 * L)) * ((2.0 * L - 1.0) * p.R**2 + p.a**2)
        if bracket <= 0:
            return float('nan')
        return 2.0 * math.pi * p.H * C0 * bracket

    def n_total(self, C0: float) -> float:
        """Total particles (bulk + bound) for a given bulk concentration C0."""
        nb = self.n_bulk(C0)
        if not np.isfinite(nb):
            raise ValueError(
                "Non-finite N_bulk. mu(C0) may make the bulk integral invalid. "
                "Check parameters/units."
            )
        return nb + self.n_bound(C0)

    def c0_from_n_total(
        self,
        N: float,
        c0_min: float = 1e-30,
        c0_max: float = 1e6,
        rtol: float = 1e-10,
        max_iter: int = 200,
    ) -> float:
        """Solve for bulk C0 given total particle count (bisection)."""
        if N <= 0:
            raise ValueError("N must be > 0")

        def f(C0):
            return self.n_total(C0) - N

        lo, hi = float(c0_min), float(c0_max)
        flo = f(lo)
        if not np.isfinite(flo):
            lo = max(lo, 1e-24)
            flo = f(lo)

        fhi = f(hi)
        for _ in range(80):
            if np.isfinite(fhi) and np.sign(flo) != np.sign(fhi):
                break
            hi *= 10.0
            fhi = f(hi)
        else:
            raise ValueError("Failed to bracket a root for C0.")

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
                hi = mid
        return float(0.5 * (lo + hi))
