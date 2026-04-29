"""Boundary condition implementations: reflection at inner cylinder,
specular reflection at outer cylinder, periodic wrapping in z."""
import numpy as np


def reflect_inner(X, a):
    """Specular reflection at inner cylinder r=a for particles that penetrated inside."""
    eps = 1e-300
    x, y = X[:, 0], X[:, 1]
    rho = np.sqrt(x**2 + y**2)
    mask = rho < a
    if not np.any(mask):
        return X

    X_in = X[mask].copy()
    rho_in = rho[mask][:, None]

    n_xy = np.zeros_like(X_in)
    n_norm = np.sqrt(X_in[:, 0]**2 + X_in[:, 1]**2)[:, None] + eps
    n_xy[:, 0:2] = X_in[:, 0:2] / n_norm

    d = a - rho_in
    X_in[:, 0:2] = X_in[:, 0:2] + 2.0 * d * n_xy[:, 0:2]

    X[mask] = X_in
    return X


def reflect_outer(S_hit, dX, t_hit, R):
    """Specular reflection from the outer cylinder r=R.

    Reflects only the remaining displacement after the first hit.
    """
    d_rem = (1.0 - t_hit)[:, None] * dX

    nx = S_hit[:, 0] / R
    ny = S_hit[:, 1] / R
    n = np.stack([nx, ny, np.zeros_like(nx)], axis=1)

    dot = np.sum(d_rem * n, axis=1)[:, None]
    d_ref = d_rem - 2.0 * dot * n

    return S_hit + d_ref


def wrap_periodic_z(X, H):
    """Periodic boundary conditions in z: z in [-H/2, H/2)."""
    halfH = 0.5 * H
    X[:, 2] = (X[:, 2] + halfH) % H - halfH
    return X
