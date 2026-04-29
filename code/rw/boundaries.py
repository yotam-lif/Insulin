import numpy as np

def reflect_from_inner_cylinder(X, a):
    """
    Specular reflection at inner cylinder of radius a
    for endpoints that end up inside (rho < a).

    Parameters
    ----------
    X : (M, 3) array
        Endpoints after a step.
    a : float
        Inner cylinder radius.

    Returns
    -------
    X : (M, 3) array
        Updated positions with points inside r < a reflected
        back across the cylindrical surface r = a.
    """
    eps = 1e-300

    # Radial distance in xy-plane
    x = X[:, 0]
    y = X[:, 1]
    rho = np.sqrt(x**2 + y**2)

    # Points that penetrated inside the inner cylinder
    mask = rho < a
    if not np.any(mask):
        return X

    X_in = X[mask].copy()
    rho_in = rho[mask][:, None]  # shape (m, 1)

    # Outward normal in xy-plane at each point (direction from axis)
    # n_xy = (x, y, 0) / rho
    n_xy = np.zeros_like(X_in)
    n_norm = np.sqrt(X_in[:, 0]**2 + X_in[:, 1]**2)[:, None] + eps
    n_xy[:, 0:2] = X_in[:, 0:2] / n_norm  # z-component stays 0

    # Distance inside the cylinder
    d = (a - rho_in)  # shape (m,1), positive since rho < a

    # Reflect across the surface: move outward by 2d along n_xy
    X_in[:, 0:2] = X_in[:, 0:2] + 2.0 * d * n_xy[:, 0:2]
    # z stays unchanged

    X[mask] = X_in
    return X



def reflect_after_outer_hit_specular_cyl(S_hit, dX, t_hit, R):
    """
    Specular reflection from the outer cylinder r = R.

    This reflects ONLY the remaining displacement after the first hit,
    preserving tangential and z components.

    Parameters
    ----------
    S_hit : (M,3) array
        First-hit points on r = R.
    dX : (M,3) array
        Full attempted displacement (X_trial - X_start).
    t_hit : (M,) array
        Fraction in [0,1] of the step at which the hit occurred.
    R : float
        Cylinder radius.

    Returns
    -------
    X_new : (M,3) array
        Reflected final positions.
    """
    # remaining displacement after reaching the boundary
    d_rem = (1.0 - t_hit)[:, None] * dX

    # outward normal at the hit point (radial direction)
    nx = S_hit[:, 0] / R
    ny = S_hit[:, 1] / R
    n = np.stack([nx, ny, np.zeros_like(nx)], axis=1)  # (M,3)

    # specular reflection: v' = v - 2 (v·n) n
    dot = np.sum(d_rem * n, axis=1)[:, None]
    d_ref = d_rem - 2.0 * dot * n

    return S_hit + d_ref


def apply_periodic_caps(X, H):
    """
    Apply periodic boundary conditions in z:

      z ∈ [-H/2, H/2) with wrapping.

    Parameters
    ----------
    X : (M, 3) array
        Positions.
    H : float
        Cylinder height.

    Returns
    -------
    X : (M, 3) array
        Positions with z wrapped periodically.
    """
    halfH = 0.5 * H
    z = X[:, 2]

    # Map z to [-H/2, H/2)
    z_wrapped = (z + halfH) % H - halfH
    X[:, 2] = z_wrapped

    return X


