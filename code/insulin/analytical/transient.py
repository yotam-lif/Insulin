"""Analytical transient solutions: eigenvalue problems for cylindrical diffusion."""
import numpy as np
from scipy.special import j0, y0, j1, y1
from scipy.optimize import brentq
from scipy.integrate import quad
from ..params import SimulationParams


class TransientAnalytical:
    """Eigenfunction expansion for transient diffusion in a cylindrical annulus.

    Supports:
    - Dirichlet (fully absorbing) at r=R
    - Robin (effective patchy absorption) at r=R
    Both with Neumann (reflecting) at r=a.
    """

    def __init__(self, params: SimulationParams, n_modes: int = 50):
        self.p = params
        self.n_modes = n_modes

    def eigenvalues_dirichlet(self) -> np.ndarray:
        """Eigenvalues for Neumann at a, Dirichlet at R.

        Root equation: J0(lam*R)*Y1(lam*a) - Y0(lam*R)*J1(lam*a) = 0
        """
        a, R = self.p.a, self.p.R

        def f(lam):
            return j0(lam * R) * y1(lam * a) - y0(lam * R) * j1(lam * a)

        return self._find_roots(f, a)

    def eigenvalues_robin(self, k: float) -> np.ndarray:
        """Eigenvalues for Neumann at a, Robin at R.

        Robin BC: phi'(R) + (k/R)*phi(R) = 0
        """
        a, R = self.p.a, self.p.R

        def f(lam):
            term1 = -lam * (j1(lam * R) * y1(lam * a) - y1(lam * R) * j1(lam * a))
            term2 = (k / R) * (j0(lam * R) * y1(lam * a) - y0(lam * R) * j1(lam * a))
            return term1 + term2

        return self._find_roots(f, a)

    def _find_roots(self, f, a) -> np.ndarray:
        R = self.p.R
        n_modes = self.n_modes
        lam_max = (n_modes + 5) * np.pi / max(R - a, 1e-12)
        grid = np.linspace(1e-8, lam_max, 20000)
        fg = f(grid)

        bad = np.abs(y1(grid * a)) < 1e-10
        fg[bad] = np.nan

        vals = []
        for i in range(len(grid) - 1):
            if np.isnan(fg[i]) or np.isnan(fg[i + 1]):
                continue
            if fg[i] == 0.0:
                vals.append(grid[i])
            elif fg[i] * fg[i + 1] < 0:
                lam = brentq(f, grid[i], grid[i + 1], maxiter=200)
                vals.append(lam)
            if len(vals) >= n_modes:
                break
        return np.array(vals, float)

    def survival_fraction(
        self,
        t: np.ndarray,
        bc: str = "dirichlet",
        k: float | None = None,
        p0: str = "thin_shell",
        eps: float = 0.0,
    ) -> np.ndarray:
        """Compute N(t)/N0 via eigenfunction expansion.

        Parameters
        ----------
        t : array
            Time points.
        bc : "dirichlet" or "robin"
        k : Robin parameter (required if bc="robin")
        p0 : Initial condition, "uniform_volume" or "thin_shell"
        eps : Shell thickness for p0="thin_shell"
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

        # Initial condition support
        if p0 == "uniform_volume":
            r1, r2 = a, R
        elif p0 == "thin_shell":
            r1, r2 = a, min(a + eps, R)
            if r2 <= r1:
                raise ValueError("For p0='thin_shell', need eps > 0 so that a < a+eps <= R.")
        else:
            raise ValueError("p0 must be 'uniform_volume' or 'thin_shell'")

        Z = (r2**2 - r1**2) / 2.0

        def p_init(r):
            if r < r1 or r > r2:
                return 0.0
            return r / Z

        def phi(lam, r):
            beta = -j1(lam * a) / y1(lam * a)
            return j0(lam * r) + beta * y0(lam * r)

        # Expansion coefficients and integrals
        A = np.zeros_like(lams)
        I = np.zeros_like(lams)

        for i, lam in enumerate(lams):
            norm = quad(lambda r: phi(lam, r)**2 * r, a, R, limit=200)[0]
            num = quad(lambda r: p_init(r) * phi(lam, r) * r, a, R, limit=200)[0]
            A[i] = num / norm
            I[i] = quad(lambda r: phi(lam, r), a, R, limit=200)[0]

        t = np.asarray(t, float)
        expf = np.exp(-D * (lams[:, None]**2) * t[None, :])
        surv = (A[:, None] * I[:, None] * expf).sum(axis=0)
        return np.clip(surv, 0.0, 1.0)
