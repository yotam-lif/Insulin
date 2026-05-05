"""Analytical transient solutions: eigenvalue problems for cylindrical diffusion."""
import numpy as np
from scipy.special import j0, y0, j1, y1
from scipy.optimize import brentq
from scipy.integrate import quad
from classes.params import SimulationParams


class TransientAnalytical:
    """Eigenfunction expansion solution for transient diffusion in a cylindrical annulus.

    Physical picture
    ----------------
    At ``t = 0``, ``N₀`` molecules are distributed uniformly in the annulus
    ``a ≤ r ≤ R``.  They diffuse and are absorbed when they reach ``r = R``
    (or a subset of the wall if using patches).  We want ``N(t)/N₀`` — the
    fraction surviving at time ``t``.

    Governing equation
    ------------------
    The probability density ``p(r, t)`` satisfies the cylindrical diffusion
    equation::

        ∂p/∂t = D · (1/r) · ∂/∂r (r · ∂p/∂r)

    with:
    * Neumann BC at ``r = a``: ``∂p/∂r = 0`` (no-flux, reflecting inner wall)
    * Either Dirichlet (``p = 0``) or Robin (``∂p/∂r = -(k/R)·p``) at ``r = R``

    The survival fraction is obtained by integrating over the domain::

        N(t)/N₀ = 2π·H ∫_a^R p(r,t) r dr

    Eigenfunction expansion
    -----------------------
    The solution is a sum of decaying modes::

        p(r, t) = Σₙ Aₙ · φₙ(r) · exp(−D λₙ² t)

    where each eigenfunction is a linear combination of Bessel functions::

        φ(r; λ) = J₀(λr) + β·Y₀(λr)

    with ``β = −J₁(λa) / Y₁(λa)`` chosen to enforce the Neumann BC at ``r = a``.

    The eigenvalues ``λₙ`` are roots of the BC equation at ``r = R``:
    * **Dirichlet**: ``φ(R; λ) = 0``
    * **Robin**: ``φ'(R; λ) + (k/R)·φ(R; λ) = 0``

    Expansion coefficients ``Aₙ`` and radial integrals ``Iₙ``
    -----------------------------------------------------------
    The survival fraction is::

        N(t)/N₀ = Σₙ Aₙ · Iₙ · exp(−D λₙ² t)

    where::

        Aₙ = ∫_a^R p₀(r) φₙ(r) r dr  /  ∫_a^R φₙ(r)² r dr
        Iₙ = ∫_a^R φₙ(r) dr

    and ``p₀(r)`` is the initial probability density (uniform in volume or
    uniform in a thin shell near ``r = a``).
    """

    def __init__(self, params: SimulationParams, n_modes: int = 50):
        """
        Parameters
        ----------
        params : SimulationParams
            Geometry and diffusion coefficient.  Only ``a``, ``R``, ``D`` are
            used; kinetic parameters are ignored.
        n_modes : int
            Number of eigenfunction modes to include in the expansion.
            More modes improve short-time accuracy at the cost of more
            root-finding calls.  50–100 is typically sufficient for
            ``t > 0.01 * (R-a)² / D``.
        """
        self.p = params
        self.n_modes = n_modes

    # ------------------------------------------------------------------
    # Eigenvalue solvers
    # ------------------------------------------------------------------

    def eigenvalues_dirichlet(self) -> np.ndarray:
        """Compute eigenvalues for Neumann BC at ``r=a``, Dirichlet BC at ``r=R``.

        The Dirichlet condition ``φ(R; λ) = 0`` gives the root equation::

            J₀(λR) · Y₁(λa) − Y₀(λR) · J₁(λa) = 0

        This is derived by substituting ``φ(r) = J₀(λr) + β·Y₀(λr)`` with
        ``β = −J₁(λa)/Y₁(λa)`` (from the Neumann BC at ``r = a``) into
        ``φ(R) = 0`` and rearranging.

        Returns
        -------
        numpy array of shape ``(n_modes,)``
            Sorted eigenvalues ``λₙ > 0`` (units: µm⁻¹).
        """
        def f(lam):
            return j0(lam * self.p.R) * y1(lam * self.p.a) - y0(lam * self.p.R) * j1(lam * self.p.a)
        return self._find_roots(f)

    def eigenvalues_robin(self, k: float) -> np.ndarray:
        """Compute eigenvalues for Neumann BC at ``r=a``, Robin BC at ``r=R``.

        The Robin condition ``φ'(R) + (k/R)·φ(R) = 0`` accounts for partial
        absorption at the outer wall — this models the effect of receptor
        patches as an effective uniform resistance to absorption.

        The effective Robin parameter is ``k = k_eff = 2·n_patches·s / (π·H)``,
        computed from the patch geometry.  When ``k → ∞`` this reduces to the
        Dirichlet (fully absorbing) case; when ``k → 0`` it reduces to
        Neumann (perfectly reflecting).

        The root equation is::

            −λ·[J₁(λR)·Y₁(λa) − Y₁(λR)·J₁(λa)]
            + (k/R)·[J₀(λR)·Y₁(λa) − Y₀(λR)·J₁(λa)] = 0

        Parameters
        ----------
        k : float
            Robin parameter (dimensionless or µm⁻¹ depending on convention).
            Use ``k_eff = 2·n_patches·s / (π·H)`` for the patchy-absorption
            approximation.

        Returns
        -------
        numpy array of shape ``(n_modes,)``
            Sorted eigenvalues ``λₙ > 0`` (units: µm⁻¹).
        """
        a, R = self.p.a, self.p.R

        def f(lam):
            t1 = -lam * (j1(lam * R) * y1(lam * a) - y1(lam * R) * j1(lam * a))
            t2 = (k / R) * (j0(lam * R) * y1(lam * a) - y0(lam * R) * j1(lam * a))
            return t1 + t2
        return self._find_roots(f)

    def _find_roots(self, f) -> np.ndarray:
        """Find the first ``n_modes`` positive roots of the characteristic equation ``f(λ) = 0``.

        Strategy
        --------
        1. Evaluate ``f`` on a fine grid from near zero up to
           ``λ_max = (n_modes + 5) · π / (R - a)``.  The spacing between
           successive eigenvalues is approximately ``π / (R - a)``, so this
           grid is guaranteed to have at least one grid point between each pair
           of roots.
        2. Find sign changes between adjacent grid points — each sign change
           brackets a root.
        3. Refine each bracket to machine precision using ``scipy.optimize.brentq``.
        4. Skip intervals where ``Y₁(λa)`` is numerically unstable (near its
           poles, the eigenfunction representation breaks down).

        Parameters
        ----------
        f : callable
            The characteristic equation ``f(λ)`` whose positive roots are
            the eigenvalues.

        Returns
        -------
        numpy array
            Up to ``n_modes`` eigenvalues, sorted ascending.
        """
        a, R = self.p.a, self.p.R
        lam_max = (self.n_modes + 5) * np.pi / max(R - a, 1e-12)
        grid = np.linspace(1e-8, lam_max, 20000)
        fg = f(grid)

        # Y₁(λa) appears in β = −J₁(λa)/Y₁(λa).  Near a pole of Y₁ the
        # eigenfunction representation is numerically unreliable.
        fg[np.abs(y1(grid * a)) < 1e-10] = np.nan

        vals = []
        for i in range(len(grid) - 1):
            if np.isnan(fg[i]) or np.isnan(fg[i + 1]):
                continue
            if fg[i] == 0.0:
                vals.append(grid[i])
            elif fg[i] * fg[i + 1] < 0:
                # Sign change: a root lies between grid[i] and grid[i+1]
                vals.append(brentq(f, grid[i], grid[i + 1], maxiter=200))
            if len(vals) >= self.n_modes:
                break

        return np.array(vals, float)

    # ------------------------------------------------------------------
    # Survival fraction
    # ------------------------------------------------------------------

    def survival_fraction(
        self,
        t: np.ndarray,
        bc: str = "dirichlet",
        k: float | None = None,
        p0: str = "uniform_volume",
        eps: float = 0.0,
    ) -> np.ndarray:
        """Compute ``N(t)/N₀`` via the eigenfunction expansion.

        The survival fraction is::

            N(t)/N₀ = Σₙ Aₙ · Iₙ · exp(−D λₙ² t)

        where:

        * ``Aₙ`` is the projection of the initial density ``p₀(r)`` onto
          eigenfunction ``φₙ(r)``, weighted by the norm of ``φₙ``.
        * ``Iₙ = ∫_a^R φₙ(r) dr`` is the spatial integral of the eigenfunction
          (converts a probability density to a particle count).
        * The exponential factor decays faster for higher modes (larger ``λₙ``),
          so at long times only the fundamental mode (``n=0``) survives.

        Parameters
        ----------
        t : array-like
            Time points at which to evaluate the survival fraction (seconds).
        bc : {"dirichlet", "robin"}
            Boundary condition at ``r = R``:
            - ``"dirichlet"``: fully absorbing wall (``C = 0``).
            - ``"robin"``: partial absorption with parameter ``k``.
        k : float or None
            Robin parameter (required when ``bc="robin"``).  Typically
            ``k_eff = 2·n_patches·s / (π·H)``.
        p0 : {"uniform_volume", "thin_shell"}
            Initial particle distribution:
            - ``"uniform_volume"``: uniform over the entire annulus ``[a, R]``.
            - ``"thin_shell"``: uniform in a thin shell ``[a, a+eps]`` near the
              inner wall.
        eps : float
            Shell thickness for ``p0="thin_shell"`` (µm).  Ignored for
            ``"uniform_volume"``.

        Returns
        -------
        surv : numpy array, same shape as ``t``
            Survival fraction in ``[0, 1]``, clipped to avoid small negative
            values from eigenfunction truncation error.
        """
        a, R, D = self.p.a, self.p.R, self.p.D

        if bc == "dirichlet":
            lams = self.eigenvalues_dirichlet()
        elif bc == "robin":
            if k is None:
                raise ValueError("For bc='robin', must supply k.")
            lams = self.eigenvalues_robin(k)
        else:
            raise ValueError("bc must be 'dirichlet' or 'robin'")

        # Initial condition: support interval [r1, r2]
        if p0 == "uniform_volume":
            r1, r2 = a, R
        elif p0 == "thin_shell":
            r1, r2 = a, min(a + eps, R)
            if r2 <= r1:
                raise ValueError("For p0='thin_shell', need eps > 0 so a < a+eps <= R.")
        else:
            raise ValueError("p0 must be 'uniform_volume' or 'thin_shell'")

        # Normalisation factor for the uniform initial distribution in [r1, r2]
        # p₀(r) = r / Z  (the r factor comes from the 2-D area element r dr dθ)
        Z = (r2**2 - r1**2) / 2.0

        def p_init(r):
            """Normalized initial probability density (area-weighted uniform)."""
            if r < r1 or r > r2:
                return 0.0
            return r / Z

        def phi(lam, r):
            """Eigenfunction φ(r; λ) = J₀(λr) + β·Y₀(λr)."""
            beta = -j1(lam * a) / y1(lam * a)
            return j0(lam * r) + beta * y0(lam * r)

        # Compute expansion coefficients Aₙ and radial integrals Iₙ for each mode
        A = np.zeros_like(lams)
        I = np.zeros_like(lams)

        for i, lam in enumerate(lams):
            # ||φₙ||² = ∫_a^R φₙ(r)² r dr  (norm in the L²(r dr) inner product)
            norm = quad(lambda r: phi(lam, r)**2 * r, a, R, limit=200)[0]
            # Aₙ = <p₀, φₙ> / ||φₙ||²
            num = quad(lambda r: p_init(r) * phi(lam, r) * r, a, R, limit=200)[0]
            A[i] = num / norm
            # Iₙ = ∫_a^R φₙ(r) dr  (integrates probability density to particle fraction)
            I[i] = quad(lambda r: phi(lam, r), a, R, limit=200)[0]

        t = np.asarray(t, float)
        # Vectorised over both modes and time points: shape (n_modes, n_times)
        expf = np.exp(-D * (lams[:, None]**2) * t[None, :])
        surv = (A[:, None] * I[:, None] * expf).sum(axis=0)
        return np.clip(surv, 0.0, 1.0)
