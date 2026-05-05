"""Geometric utilities: uniform particle sampling and ray-cylinder intersection."""
import numpy as np


def rand_points_in_shell(n, a, R, H, eps, rng, where="inner"):
    """Sample ``n`` uniformly-distributed random points inside a thin cylindrical shell.

    Physical picture
    ----------------
    The geometry is a cylindrical annulus of inner radius ``a``, outer radius
    ``R``, and height ``H``.  This function places particles uniformly in a
    thin sub-shell near one of the two walls:

    * ``where="inner"``  → shell between ``r = a`` and ``r = a + eps``
      (used to respawn internalized particles near the capillary wall)
    * ``where="outer"``  → shell between ``r = R - eps`` and ``r = R``
      (used to respawn dissociated particles near the outer receptor wall)

    Why naive sampling (``r = rng.uniform(r0, r1)``) would be wrong
    ---------------------------------------------------------------
    In 2-D polar coordinates, equal-area annular rings grow as ``r dr``, so
    thin rings at large ``r`` contain more area than thin rings at small ``r``.
    Sampling ``r`` uniformly in ``[r0, r1]`` would over-populate small radii
    and under-populate large radii.

    The correct approach uses the cumulative distribution of the annular area:
    the area up to radius ``r`` is proportional to ``r² - r0²``, so drawing
    ``u ~ Uniform(0,1)`` and inverting gives::

        r = sqrt(r0² + u * (r1² - r0²))

    This ensures each unit of area on the annulus has equal probability.

    Parameters
    ----------
    n : int
        Number of points to sample.  Returns an empty (0,3) array if ``n<=0``.
    a : float
        Inner cylinder radius (µm).
    R : float
        Outer cylinder radius (µm).
    H : float
        Cylinder height (µm).  The z-axis runs from ``-H/2`` to ``+H/2``.
    eps : float
        Shell thickness (µm).  Must be > 0.
    rng : numpy.random.Generator
        Random number generator (e.g. ``np.random.default_rng(seed)``).
    where : {"inner", "outer"}
        Which wall the shell is adjacent to.  Case-insensitive.

    Returns
    -------
    X : (n, 3) float array
        Rows are [x, y, z] Cartesian coordinates of each sampled point.
        The radial distance of each row satisfies:
        * ``where="inner"``: ``a <= sqrt(x²+y²) <= a + eps``
        * ``where="outer"``: ``R - eps <= sqrt(x²+y²) <= R``

    Raises
    ------
    ValueError
        If ``where`` is not 'inner' or 'outer', ``eps <= 0``, or the
        requested shell falls outside the annulus.
    """
    where = where.lower().strip()
    if where not in {"inner", "outer"}:
        raise ValueError("where must be 'inner' or 'outer'")

    a, R, H, eps = float(a), float(R), float(H), float(eps)
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
    else:
        r0, r1 = R - eps, R
        if r0 < a:
            raise ValueError("outer shell overlaps inner radius: R-eps < a")

    # Uniform azimuthal angle and z (both truly uniform in these coordinates)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.uniform(-H / 2.0, H / 2.0, size=n)

    # Area-weighted radial sampling: invert the CDF of r to get uniform density
    # on the annular cross-section.  Naive r ~ Uniform(r0, r1) would give
    # too many particles near the axis.
    u = rng.uniform(0.0, 1.0, size=n)
    r = np.sqrt(r0 * r0 + u * (r1 * r1 - r0 * r0))

    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def segment_hits_cylinder(X, dX, R, *, snap=True):
    """Find where line segments cross the cylindrical surface ``r = R``.

    Physical picture
    ----------------
    During each time step a diffusing particle moves from position ``X`` by a
    Gaussian displacement ``dX``.  If the trial endpoint ``X + dX`` lies
    outside the cylinder (``r > R``), the particle crossed the wall mid-step.
    We need to find *exactly where* it crossed — the hit point ``S`` — so we
    can decide whether to absorb or reflect the particle at that location.

    The computation is a standard ray-cylinder intersection:

    A point on the segment is ``P(t) = X + t * dX``, ``t ∈ [0, 1]``.  We
    want the smallest ``t`` in ``[0, 1]`` such that ``P(t)`` lies on ``r = R``,
    i.e. ``Px(t)² + Py(t)² = R²`` (the z-coordinate plays no role because the
    cylinder is infinite in z — z-periodicity is handled separately).

    Expanding gives the quadratic::

        (dx²+dy²) t² + 2(x·dx + y·dy) t + (x²+y² - R²) = 0

    where ``(x,y) = X[:2]`` and ``(dx,dy) = dX[:2]``.  With discriminant
    ``disc = b² - 4ac``:

    * ``disc < 0``: no real intersection — the step stays inside or outside.
    * ``t1 <= t2``: the two roots give the entry and exit points of the
      infinite line through the cylinder.  We take the smaller *positive*
      root ``t ∈ [0, 1]`` as the first crossing.

    ``snap=True`` corrects floating-point drift
    -------------------------------------------
    After computing ``S = X + t*dX``, round-off means ``sqrt(S[0]²+S[1]²)``
    is not *exactly* ``R``.  With ``snap=True`` we rescale the x,y components
    so the hit point lies exactly on the surface, preventing particles from
    slowly drifting outside ``r = R`` over many steps.

    Parameters
    ----------
    X : (M, 3) float array
        Starting positions of ``M`` particles.
    dX : (M, 3) float array
        Displacement vectors (same shape as ``X``).
    R : float
        Cylinder radius.
    snap : bool, optional
        If True (default), rescale hit-point x,y so ``sqrt(x²+y²)`` equals
        exactly ``R`` after the intersection calculation.

    Returns
    -------
    hit_mask : (M,) bool array
        True for every particle whose segment crosses ``r = R`` within
        ``t ∈ [0, 1]``.
    t_hit : (M,) float array
        The crossing parameter ``t`` for each hit particle (``inf`` if no hit).
    S : (M, 3) float array
        For hit particles: the point on the cylinder surface where the
        crossing occurred.  For non-hit particles: the trial endpoint
        ``X + dX``.
    """
    eps = 1e-300
    M = X.shape[0]

    S = X + dX
    t_hit = np.full(M, np.inf, dtype=float)

    # Only the xy-plane projection matters for the radial distance check
    X_xy, dX_xy = X[:, :2], dX[:, :2]
    aa = np.sum(dX_xy * dX_xy, axis=1)          # coefficient of t²
    bb = 2.0 * np.sum(X_xy * dX_xy, axis=1)     # coefficient of t
    cc = np.sum(X_xy * X_xy, axis=1) - R * R    # constant term

    disc = bb * bb - 4.0 * aa * cc
    valid = (disc >= 0.0) & (aa > 0.0)          # real solutions exist

    if np.any(valid):
        sd = np.zeros_like(disc)
        sd[valid] = np.sqrt(disc[valid])

        denom = 2.0 * aa[valid] + eps            # avoid division by zero
        t1 = (-bb[valid] - sd[valid]) / denom    # earlier crossing
        t2 = (-bb[valid] + sd[valid]) / denom    # later crossing

        # Accept whichever root is in [0, 1]; prefer the smaller one (first hit)
        t1_ok = (t1 >= 0.0) & (t1 <= 1.0)
        t2_ok = (t2 >= 0.0) & (t2 <= 1.0)

        t_local = np.full_like(t1, np.inf)
        t_local[t1_ok] = t1[t1_ok]
        # Use t2 only when t1 is out of range (particle starting near wall)
        t_local[(~t1_ok) & t2_ok] = t2[(~t1_ok) & t2_ok]

        t_hit[valid] = t_local

    hit_mask = np.isfinite(t_hit) & (t_hit >= 0.0) & (t_hit <= 1.0)

    if np.any(hit_mask):
        S_hit = X[hit_mask] + t_hit[hit_mask, None] * dX[hit_mask]
        if snap:
            # Force the hit point exactly onto r=R to remove floating-point drift
            rho = np.sqrt(S_hit[:, 0]**2 + S_hit[:, 1]**2) + eps
            scale = R / rho
            S_hit[:, 0] *= scale
            S_hit[:, 1] *= scale
        S[hit_mask] = S_hit

    return hit_mask, t_hit, S
