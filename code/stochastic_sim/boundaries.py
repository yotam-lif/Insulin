"""Boundary condition handlers: reflection at inner/outer cylinder, periodic z."""
import numpy as np


def reflect_inner(X, a):
    """Reflect particles back into the annulus at the inner cylinder ``r = a``.

    Physical picture
    ----------------
    The inner cylinder (radius ``a``) is impenetrable — it represents the
    capillary wall or cell nucleus.  If a Gaussian diffusion step carries a
    particle to ``r < a``, we apply *specular reflection*: the component of
    the displacement that points radially inward is reversed.

    Because the displacement can be large relative to the shell width, we use
    the final position ``X`` (not the trajectory) to compute reflection:

    1. Find particles with ``r < a``.
    2. Compute the outward unit normal at the (incorrect) position:
       ``n̂ = (x, y) / ||(x, y)||``.
    3. The particle overshot the wall by ``d = a - r`` in the radial direction.
       Move it ``2d`` outward along ``n̂``, placing it at ``r = a + d``.

    This is equivalent to reflecting the position about the inner cylinder
    surface.

    Parameters
    ----------
    X : (M, 3) float array
        Current positions of ``M`` particles (modified in place).
    a : float
        Inner cylinder radius (µm).

    Returns
    -------
    X : (M, 3) float array
        Same array with penetrating particles reflected.  Non-penetrating
        particles are unchanged.
    """
    eps = 1e-300
    rho = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    mask = rho < a
    if not np.any(mask):
        return X

    X_in = X[mask].copy()
    # Outward radial unit normal at the (wrong) position
    n_norm = np.sqrt(X_in[:, 0]**2 + X_in[:, 1]**2)[:, None] + eps
    n_xy = X_in[:, :2] / n_norm

    # Radial penetration depth: how far inside the inner wall the particle went
    d = (a - rho[mask])[:, None]
    # Specular reflection: push back by 2*d outward
    X_in[:, :2] += 2.0 * d * n_xy

    X[mask] = X_in
    return X


def reflect_outer(S_hit, dX, t_hit, R):
    """Reflect the remaining displacement specularly off the outer cylinder ``r = R``.

    Physical picture
    ----------------
    When a particle crosses the outer wall at time ``t = t_hit * dt`` during
    its step, only the *remaining* fraction ``(1 - t_hit)`` of the displacement
    needs to be reflected.  The part of the step before the wall hit,
    ``t_hit * dX``, already ended at the wall surface point ``S_hit``.

    Specular (mirror) reflection works as follows:

    1. Compute the remaining displacement:
       ``d_rem = (1 - t_hit) * dX``
    2. Compute the outward unit normal to the cylinder at the hit point:
       ``n̂ = (S_hit.x / R, S_hit.y / R, 0)``
       (the normal points radially outward; z-component is zero because the
       cylinder wall is vertical)
    3. Remove the component of ``d_rem`` along ``n̂`` and add it back reversed:
       ``d_reflected = d_rem - 2 * (d_rem · n̂) * n̂``
    4. The new position is ``S_hit + d_reflected``.

    Note: this function does *not* check whether the reflected endpoint lands
    inside the annulus.  The calling code should follow up with
    ``reflect_inner`` if the step is large relative to the shell width.

    Parameters
    ----------
    S_hit : (M, 3) float array
        Hit points on the outer cylinder surface (``sqrt(x²+y²) = R``).
    dX : (M, 3) float array
        Original full displacement vectors for these ``M`` particles.
    t_hit : (M,) float array
        Crossing parameters in ``[0, 1]`` (fraction of step at which the
        particle reached the wall).
    R : float
        Outer cylinder radius (µm), used to normalise the surface normal.

    Returns
    -------
    X_new : (M, 3) float array
        Positions after reflection, one per input particle.
    """
    # Remaining displacement after the wall contact point
    d_rem = (1.0 - t_hit)[:, None] * dX

    # Outward unit normal at the hit point (cylinder wall is vertical, so z=0)
    nx = S_hit[:, 0] / R
    ny = S_hit[:, 1] / R
    n = np.stack([nx, ny, np.zeros_like(nx)], axis=1)

    # Reflect d_rem about the plane perpendicular to n̂
    dot = np.sum(d_rem * n, axis=1)[:, None]
    return S_hit + d_rem - 2.0 * dot * n


def wrap_periodic_z(X, H):
    """Apply periodic boundary conditions in the z direction.

    Physical picture
    ----------------
    The cylinder has finite height ``H``, but the simulation treats it as one
    tile in an infinite periodic stack.  A particle that exits through the top
    (``z > H/2``) immediately re-enters through the bottom (``z = -H/2 + overshoot``),
    and vice versa.

    This models an infinitely long cylinder without needing to simulate the
    full length — equivalent to tiling the cylinder and assuming the
    concentration gradient is the same in every tile.

    Mathematically: ``z ← ((z + H/2) mod H) - H/2``.

    * Example: if ``H = 10`` (so z ∈ [-5, 5]) and a particle reaches ``z = 7``,
      it wraps to ``z = 7 - 10 = -3``.
    * Example: if a particle reaches ``z = -8``, it wraps to ``z = -8 + 10 = 2``.

    Parameters
    ----------
    X : (M, 3) float array
        Positions to wrap (modified in place).
    H : float
        Cylinder height (µm).  The valid z-range is ``[-H/2, H/2)``.

    Returns
    -------
    X : (M, 3) float array
        Same array with z-coordinates wrapped into ``[-H/2, H/2)``.
    """
    halfH = 0.5 * H
    X[:, 2] = (X[:, 2] + halfH) % H - halfH
    return X
