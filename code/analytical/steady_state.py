"""Analytical steady-state solutions for the cylinder-in-cylinder model."""
import math
import numpy as np
from classes.params import SimulationParams


AVOGADRO = 6.02214076e23


def molar_to_molecules_per_um3(c_molar: float) -> float:
    """Convert a concentration from molar (mol/L) to molecules per µm³.

    The conversion factor is Avogadro's number divided by the number of µm³
    per litre::

        1 L = 1e15 µm³
        => 1 M = 6.022e23 molecules / 1e15 µm³ = 6.022e8 molecules/µm³

    Parameters
    ----------
    c_molar : float or array
        Concentration in mol/L.

    Returns
    -------
    float or array
        Concentration in molecules/µm³.
    """
    return c_molar * AVOGADRO / 1e15


class AnalyticalModel:
    """Closed-form steady-state solution for the chemoreception model.

    Physical picture
    ----------------
    Molecules (e.g. insulin) diffuse in the cylindrical gap between an inner
    capillary (radius ``a``) and an outer cell membrane (radius ``R``).  At
    steady state the concentration profile is the solution to the 2-D
    cylindrical Laplace equation with:

    * Neumann (no-flux) BC at ``r = a``: ``∂C/∂r = 0``
    * Mixed BC at ``r = R``: receptors consume molecules at rate ``J``

    This gives the logarithmic profile::

        C(r) = C₀ · [1 − μ · ln(r/a) / ln(R/a)]

    where ``μ = J/J_max`` is the *uptake efficiency* — the fraction of the
    maximum possible flux actually achieved.

    Three-step receptor kinetics
    ----------------------------
    Each receptor has three rate constants:

    ``ka`` (µm³/s)
        Association: a free molecule touching the receptor binds at this rate.
        The effective rate is limited by diffusion to the patch:
        ``ka* = 4Ds / (1 + 4Ds/ka)`` where ``s`` is the patch radius.

    ``kb`` (s⁻¹)
        Dissociation: a bound molecule detaches and returns to the outer bulk.

    ``kc`` (s⁻¹)
        Internalization: a bound molecule is transported through the membrane
        and is counted as "absorbed".

    The receptor occupancy (fraction of receptors bound at steady state) is::

        θ = C(R) / (C(R) + Kd)   where   Kd = (kb + kc) / ka*

    Provided methods
    ----------------
    * ``mu(C0)``              — uptake efficiency J/J_max
    * ``concentration_at_r``  — radial concentration profile C(r)
    * ``flux``                — total absorption rate J (particles/s)
    * ``bound_fraction``      — fraction of receptors bound
    * ``n_bound``, ``n_bulk``, ``n_total`` — particle counts
    * ``c0_from_n_total``     — inverse: find C0 given N_total
    """

    def __init__(self, params: SimulationParams):
        """
        Parameters
        ----------
        params : SimulationParams
            The geometry and kinetics parameters.  ``N`` is ignored by all
            methods (the analytical model works directly with concentrations).
        """
        self.p = params

    # ------------------------------------------------------------------
    # Derived constants (computed on demand from params)
    # ------------------------------------------------------------------

    @property
    def ka_star(self) -> float:
        """Effective patch association rate ``ka*`` (µm³/s per patch).

        Physical meaning
        ----------------
        A molecule diffusing freely is not really "free" when it reaches the
        receptor — it still needs to cross the last nanometre gap and actually
        bind.  The true binding rate is limited by the smaller of:

        * **Diffusion-limited rate**: ``4Ds`` — how quickly diffusion brings a
          molecule touching the patch rim into contact with the receptor.
        * **Intrinsic rate**: ``ka`` — the chemical binding rate once the
          molecule is there.

        The interpolation formula is::

            ka* = 4Ds / (1 + 4Ds/ka)

        When ``ka = inf`` (diffusion limited): ``ka* = 4Ds``.
        When ``ka → 0`` (reaction limited):   ``ka* ≈ ka``.
        """
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
        """Dissociation constant ``Kd = (kb + kc) / ka*`` (molecules/µm³).

        Physical meaning
        ----------------
        ``Kd`` is the bulk concentration at which exactly half the receptors
        are occupied at steady state.  It is the ratio of the total "off" rate
        (dissociation ``kb`` plus internalization ``kc``) to the effective
        "on" rate ``ka*``::

            θ = C(R) / (C(R) + Kd)

        A small ``Kd`` means the receptor has a high affinity (binds at low
        concentrations).  A large ``Kd`` means the receptor needs high
        concentrations to be occupied.
        """
        ka_s = self.ka_star
        if ka_s <= 0:
            return float('inf')
        return (self.p.kb + self.p.kc) / ka_s

    @property
    def k_max(self) -> float:
        """Maximum (fully-absorbing) rate constant ``k_max = 2πHD / ln(R/a)`` (µm³/s).

        Physical meaning
        ----------------
        This is the rate at which molecules would be absorbed if every molecule
        that reached the outer wall was instantly consumed — a perfectly absorbing
        boundary (Dirichlet condition ``C(R) = 0``).

        It follows from Fick's first law applied to the logarithmic profile:
        at full absorption ``μ = 1`` and the flux simplifies to::

            J = k_max * C₀

        The factor ``2πH/ln(R/a)`` encodes the cylindrical geometry: diffusion
        over a log-distance (not a linear distance) gives a logarithmic profile.
        """
        return 2.0 * math.pi * self.p.H * self.p.D / math.log(self.p.R / self.p.a)

    @property
    def sigma0(self) -> float:
        """Receptor surface density on the outer cylinder (patches/µm²).

        This is simply ``n_patches`` divided by the lateral surface area of
        the outer cylinder: ``2πRH``.
        """
        return self.p.n_patches / (2.0 * math.pi * self.p.R * self.p.H)

    # ------------------------------------------------------------------
    # Core formulas
    # ------------------------------------------------------------------

    def mu(self, C0: float) -> float:
        """Uptake efficiency ``μ(C0) = J / J_max``, a dimensionless number in ``[0, 1]``.

        Physical meaning
        ----------------
        ``μ`` is the fraction of the maximum possible flux actually achieved.
        ``μ = 1`` means every molecule reaching the outer wall is absorbed
        (fully absorbing wall).  ``μ = 0`` means no molecules are absorbed
        (no receptors, or receptors fully saturated at low ``kc``).

        Two analytical limits
        ---------------------
        **Berg-Purcell limit** (``kb = 0``, ``kc = inf``, ``ka = inf``):
        patches are instantly absorbing.  The formula is a simple competition
        between the "receptor aperture" ``N*s`` and the diffusion resistance
        ``π*H / (2*ln(R/a))``::

            μ = N*s / (N*s + πH / (2*ln(R/a)))

        **General finite kinetics** (any ``ka``, ``kb``, ``kc``):
        solved by flux-balance at steady state.  The occupancy equation and
        diffusion equation together yield a quadratic for ``μ``::

            A = D*(C0+Kd) / (R*ln(R/a)) + σ₀*kc
            B = 4*(D*C0 / (R*ln(R/a))) * σ₀*kc
            C_coeff = 2*D*C0 / (R*ln(R/a))
            μ = (A − √(A² − B)) / C_coeff

        When ``C0 ≫ Kd`` (high concentration), receptors are saturated and
        ``μ`` approaches the Berg-Purcell value.  When ``C0 ≪ Kd``, few
        receptors are occupied and ``μ`` is small.

        Parameters
        ----------
        C0 : float
            Bulk concentration at ``r = a`` (molecules/µm³).  Must be > 0.

        Returns
        -------
        float
            Uptake efficiency in ``[0, 1]``.
        """
        if C0 <= 0:
            raise ValueError("C0 must be > 0")

        p = self.p
        L = math.log(p.R / p.a)
        if L <= 0:
            raise ValueError("Need R > a")

        # Berg-Purcell limit: no dissociation, instant internalization
        if p.kb == 0 and math.isinf(p.kc):
            Ns = p.n_patches * p.patch_radius
            return Ns / (Ns + p.H * math.pi / (2.0 * L))

        # General kinetics: solve the quadratic from flux-balance
        denom_geom = p.R * L
        ka_s = self.ka_star
        if ka_s <= 0:
            raise ValueError("ka_star is non-positive; check parameters")

        Kd = (p.kb + p.kc) / ka_s
        s0 = self.sigma0

        A = p.D * (C0 + Kd) / denom_geom + s0 * p.kc
        B = 4.0 * (p.D * C0 / denom_geom) * s0 * p.kc
        C = 2.0 * (p.D * C0 / denom_geom)

        disc = A * A - B
        if disc < 0:
            if disc > -1e-14 * max(1.0, A * A):
                disc = 0.0
            else:
                raise ValueError(f"Negative discriminant A^2-B={disc:g}; check parameters.")

        return float((A - math.sqrt(disc)) / C)

    def concentration_at_r(self, r, C0: float) -> np.ndarray:
        """Radial concentration profile at steady state: ``C(r) = C₀ · [1 − μ·ln(r/a)/ln(R/a)]``.

        Physical meaning
        ----------------
        The steady-state solution to ``∇²C = 0`` in cylindrical coordinates
        with reflecting inner wall and absorbing outer wall is logarithmic in
        ``r``.  The profile rises from ``C(R) = C₀·(1−μ)`` at the receptor
        wall to ``C₀`` at the inner wall.

        Parameters
        ----------
        r : float or array (µm)
            Radial position(s) in ``[a, R]``.
        C0 : float
            Bulk concentration at ``r = a`` (molecules/µm³).

        Returns
        -------
        numpy array
            Concentration (molecules/µm³) at each ``r``.
        """
        r = np.asarray(r, dtype=float)
        m = self.mu(C0)
        L = math.log(self.p.R / self.p.a)
        return C0 * (1.0 - m * np.log(r / self.p.a) / L)

    def concentration_at_R(self, C0: float) -> float:
        """Concentration at the outer surface: ``C(R) = C₀ · (1 − μ)``.

        This is the concentration directly at the receptor patches.  It equals
        ``C₀`` when ``μ = 0`` (no absorption) and drops to 0 when ``μ = 1``
        (fully absorbing wall).
        """
        return C0 * (1.0 - self.mu(C0))

    def flux(self, C0: float) -> float:
        """Total uptake rate ``J = μ · k_max · C₀`` (particles/s).

        This is the number of molecules absorbed per second by all receptors
        combined at steady state.
        """
        return self.mu(C0) * self.k_max * C0

    def flux_ratio(self, C0: float) -> float:
        """Normalised flux ``J/J_max = μ(C₀)``.  Alias for ``mu(C0)``."""
        return self.mu(C0)

    # ------------------------------------------------------------------
    # Receptor occupancy
    # ------------------------------------------------------------------

    def bound_fraction(self, C0: float) -> float:
        """Fraction of receptors occupied at steady state: ``θ = C(R) / (C(R) + Kd)``.

        Physical meaning
        ----------------
        This is a Langmuir-type binding isotherm evaluated at the local
        concentration ``C(R)`` at the outer wall (not the bulk concentration
        at ``r = a``).  The receptor sees a depleted concentration ``C(R) < C₀``
        because diffusion cannot keep up with the uptake rate.

        Returns a value in ``[0, 1]``.
        """
        C_R = max(self.concentration_at_R(C0), 0.0)
        Kd = self.kd
        return C_R / (C_R + Kd)

    def bound_percent(self, C0: float) -> float:
        """Receptor occupancy as a percentage (0–100).

        Equivalent to ``100 * bound_fraction(C0)``.
        """
        return 100.0 * self.bound_fraction(C0)

    def n_bound(self, C0: float) -> float:
        """Expected number of bound receptors at steady state.

        Computed as ``n_patches * n_sites_per_patch * bound_fraction(C0)``.
        """
        return self.p.n_patches * self.p.n_sites_per_patch * self.bound_fraction(C0)

    # ------------------------------------------------------------------
    # Particle counts
    # ------------------------------------------------------------------

    def n_bulk(self, C0: float) -> float:
        """Number of free (unbound) molecules in the annular volume at steady state.

        Physical meaning
        ----------------
        Integrates the logarithmic concentration profile ``C(r)`` over the
        cylindrical annulus:

        ``N_bulk = 2πH ∫_a^R C(r) r dr``

        Substituting ``C(r) = C₀·[1 − μ·ln(r/a)/ln(R/a)]`` and evaluating::

            N_bulk = 2πH·C₀ · [(R²−a²)/2
                     − μ/(4·ln(R/a)) · ((2·ln(R/a)−1)·R² + a²)]

        This is used together with ``n_bound`` to compute the total particle
        count, which in turn determines how many particles to place in the
        simulation to achieve a given target concentration ``C₀``.
        """
        p = self.p
        L = math.log(p.R / p.a)
        m = self.mu(C0)
        bracket = (p.R**2 - p.a**2) / 2.0 - (m / (4.0 * L)) * ((2.0 * L - 1.0) * p.R**2 + p.a**2)
        if bracket <= 0:
            return float('nan')
        return 2.0 * math.pi * p.H * C0 * bracket

    def n_total(self, C0: float) -> float:
        """Total particle count: free in bulk + bound to receptors.

        Used to set the integer particle count ``N`` in the simulation so that
        the actual simulated concentration matches the target ``C₀``:

            ``N_total = N_bulk(C₀) + N_bound(C₀)``

        Raises ``ValueError`` if ``N_bulk`` is non-finite (indicates
        parameters outside the valid physical range).
        """
        nb = self.n_bulk(C0)
        if not np.isfinite(nb):
            raise ValueError(
                "Non-finite N_bulk from mu(C0) integral. Check parameters/units."
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
        """Inverse solver: find the bulk concentration ``C₀`` that gives a total
        particle count of ``N``.

        Solves ``n_total(C₀) = N`` by bisection.  The function is
        monotonically increasing in ``C₀``, so a unique root always exists.

        Parameters
        ----------
        N : float
            Target total particle count (free + bound).
        c0_min : float
            Lower bound on the search bracket for ``C₀`` (molecules/µm³).
        c0_max : float
            Initial upper bound; automatically expanded if ``n_total(c0_max) < N``.
        rtol : float
            Relative tolerance for bisection convergence.
        max_iter : int
            Maximum number of bisection iterations.

        Returns
        -------
        float
            Bulk concentration ``C₀`` (molecules/µm³) such that
            ``n_total(C₀) ≈ N``.

        Raises
        ------
        ValueError
            If ``N <= 0`` or if a valid bracket cannot be found.
        """
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
        # Expand upper bracket if n_total(c0_max) < N
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
