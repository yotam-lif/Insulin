import numpy as np



def rand_points_in_thin_cyl_volume(n, a, R, H, eps, rng, where="inner"):
    """
    Uniform in volume in a *thin* cylindrical shell.

    Parameters
    ----------
    n : int
        Number of points.
    a : float
        Inner cylinder radius.
    R : float
        Outer cylinder radius.
    H : float
        Cylinder height (z in [-H/2, H/2]).
    eps : float
        Shell thickness.
    rng : np.random.Generator
        Random generator.
    where : str
        "inner" -> sample in a <= r <= a+eps
        "outer" -> sample in R-eps <= r <= R

    Returns
    -------
    X : (n, 3) ndarray
        Points [x, y, z].
    """
    where = where.lower().strip()
    if where not in {"inner", "outer"}:
        raise ValueError("where must be 'inner' or 'outer'")

    a = float(a); R = float(R); H = float(H); eps = float(eps)
    if n <= 0:
        return np.empty((0, 3), dtype=float)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    if where == "inner":
        r0, r1 = a, a + eps
        if r0 < 0.0:
            raise ValueError("a must be >= 0")
        if r1 > R:
            raise ValueError("inner shell exceeds outer radius: a+eps > R")
    else:  # "outer"
        r0, r1 = R - eps, R
        if r0 < a:
            raise ValueError("outer shell overlaps inner radius: R-eps < a")
        if R <= 0.0:
            raise ValueError("R must be > 0")

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z     = rng.uniform(-H / 2.0, H / 2.0, size=n)

    # Uniform in area => sample r^2 uniformly on [r0^2, r1^2]
    u = rng.uniform(0.0, 1.0, size=n)
    r = np.sqrt(r0*r0 + u * (r1*r1 - r0*r0))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

def segment_hits_cylinder(X, dX, R, *, snap=True):
    """
    For each segment x(t) = X + t dX, t in [0,1], solve
        (x + t dx)^2 + (y + t dy)^2 = R^2

    i.e. intersection with the infinite cylinder of radius R around the z-axis.

    Parameters
    ----------
    X : (M, 3) array
        Starting points of segments.
    dX : (M, 3) array
        Displacements (end - start).
    R : float
        Cylinder radius.
    snap : bool, default True
        If True, snap intersection points exactly onto r = R
        in the xy-plane (z is left as given by the parametric t).

    Returns
    -------
    hit_mask : (M,) bool
        True if there is an intersection with t in [0,1].
    t_hit : (M,) float
        Earliest forward root in [0,1]; np.inf if no hit.
    S : (M, 3)
        Final positions:
          - for hits: intersection point on radius R
          - for non-hits: plain endpoint X + dX
    """
    eps = 1e-300
    M = X.shape[0]

    # Default: no hit, endpoint
    S = X + dX
    t_hit = np.full(M, np.inf, dtype=float)

    # Use only x,y components for the cylinder intersection
    X_xy  = X[:, :2]          # (x, y)
    dX_xy = dX[:, :2]         # (dx, dy)

    # Quadratic coefficients for (x + t dx)^2 + (y + t dy)^2 = R^2
    # => (dX_xy·dX_xy) t^2 + 2 (X_xy·dX_xy) t + (X_xy·X_xy - R^2) = 0
    aa = np.sum(dX_xy * dX_xy, axis=1)            # |dX_xy|^2
    bb = 2.0 * np.sum(X_xy * dX_xy, axis=1)       # 2 X_xy·dX_xy
    cc = np.sum(X_xy * X_xy, axis=1) - R * R      # |X_xy|^2 - R^2

    disc = bb * bb - 4.0 * aa * cc

    valid_quad = (disc >= 0.0) & (aa > 0.0)
    if np.any(valid_quad):
        sd = np.zeros_like(disc)
        sd[valid_quad] = np.sqrt(disc[valid_quad])

        denom = 2.0 * aa[valid_quad] + eps
        t1 = (-bb[valid_quad] - sd[valid_quad]) / denom
        t2 = (-bb[valid_quad] + sd[valid_quad]) / denom

        t1_valid = (t1 >= 0.0) & (t1 <= 1.0)
        t2_valid = (t2 >= 0.0) & (t2 <= 1.0)

        # earliest forward root in [0,1]
        t_local = np.full_like(t1, np.inf)
        t_local[t1_valid] = t1[t1_valid]
        use_t2 = (~t1_valid) & t2_valid
        t_local[use_t2] = t2[use_t2]

        t_hit[valid_quad] = t_local

    # Hits: finite t in [0,1]
    hit_mask = np.isfinite(t_hit) & (t_hit >= 0.0) & (t_hit <= 1.0)

    if np.any(hit_mask):
        # Intersection before snapping
        S_hit = X[hit_mask] + t_hit[hit_mask, None] * dX[hit_mask]

        if snap:
            # Snap only the xy-plane radius to exactly R; keep z as is
            x_hit = S_hit[:, 0]
            y_hit = S_hit[:, 1]
            rho = np.sqrt(x_hit**2 + y_hit**2) + eps

            scale = R / rho
            x_hit *= scale
            y_hit *= scale

            S_hit[:, 0] = x_hit
            S_hit[:, 1] = y_hit

        S[hit_mask] = S_hit

    return hit_mask, t_hit, S
