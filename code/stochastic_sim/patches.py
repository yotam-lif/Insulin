"""Receptor patches on the outer cylinder surface.

Patches are small circular absorption sites sitting on the outer wall r=R.
A particle that hits the wall during diffusion is absorbed only if it strikes
a patch; otherwise it bounces back.  The class also models the subsequent
fate of a bound particle: it can be released back into the bulk (dissociation,
rate kb) or internalized by the cell (rate kc).
"""
import numpy as np


class PatchSet:
    """Circular receptor patches placed on the outer cylinder r=R.

    Physical picture
    ----------------
    The outer cylinder surface is 'unrolled' into a flat 2-D rectangle of
    width L = 2*pi*R (circumference) and height H.  Each patch is a disc of
    radius `patch_radius` on this rectangle.  Because the rectangle wraps in
    both directions (θ is periodic with period 2π; z is periodic with period H),
    the geometry is topologically a torus.

    A particle that hits the outer wall at some 3-D point S is absorbed only
    if S falls inside one of the patches.  If S lands on open wall between
    patches it is reflected specularly and continues diffusing.

    Binding kinetics
    ----------------
    Bound particles leave the receptor with total rate k_off = kb + kc.
    - With probability kb / (kb + kc) they are released back to the bulk
      near r=R (dissociation).
    - With probability kc / (kb + kc) they are internalized and counted as
      absorbed (they are respawned near r=a to keep particle count constant).

    Spatial lookup
    --------------
    Naively checking every patch for every wall-hit would be O(n_hits * n_patches).
    Instead, the unrolled surface is divided into a grid of square cells (cell size
    = patch_radius).  Each patch registers itself in all grid cells it overlaps.
    A wall-hit point is assigned to its cell in O(1) and only the patches in that
    cell and its neighbors are checked, giving O(1) average lookup.

    Parameters
    ----------
    R : float
        Outer cylinder radius (µm).
    H : float
        Cylinder height (µm).
    n_patches : int
        Number of receptor patches to place.
    patch_radius : float or None
        Radius of each patch in the unrolled (arc-length, z) coordinate system
        (µm).  A particle hitting the wall within this radius of a patch centre
        is considered 'on' that patch.
    rng : np.random.Generator
        Random number generator used for patch placement.
    ka : float
        Intrinsic association rate (µm³/s).  Controls how likely a particle
        that hits a patch is to bind.  math.inf means every hit leads to binding
        (perfectly absorbing patch); smaller values model a finite on-rate via
        the Erban-Chapman correction applied in the simulation loop.
    kb : float
        Dissociation rate (s⁻¹).  A bound particle leaves the receptor and
        re-enters the bulk near r=R at this rate.
    kc : float
        Internalization rate (s⁻¹).  A bound particle is absorbed by the cell
        at this rate (it re-enters the simulation near r=a).
    enabled : bool
        If False (or if n_patches == 0 or patch_radius is None/zero), the
        PatchSet is inactive: all outer-wall hits are treated as fully absorbing.
    n_sites_per_patch : int
        Maximum number of particles that can be simultaneously bound to a single
        patch.  Default 1 (one receptor site per patch).
    enforce_nonoverlap : bool
        If True, patches are placed via Random Sequential Adsorption so they do
        not overlap.  If False, patch centres are drawn independently (Poisson
        process), which allows overlaps.
    max_attempts : int
        Maximum total placement attempts in the RSA algorithm before raising
        RuntimeError.  Increase this if placing a large number of patches close
        to the jamming limit.

    Internal state set after construction
    --------------------------------------
    theta_centers : (n_patches,) array -- angular position of each patch center (rad)
    z_centers     : (n_patches,) array -- axial position of each patch center (µm)
    radius        : float              -- patch radius (same as patch_radius)
    radius2       : float              -- radius squared (cached for distance tests)
    occ           : (n_patches,) int32 -- current occupancy of each patch (0 .. n_sites_per_patch)
    n_sites_per_patch : int            -- capacity per patch
    n_sites_total     : int            -- total receptor sites = n_patches * n_sites_per_patch
    L    : float       -- cylinder circumference 2*pi*R (µm), width of unrolled rectangle
    cell : float       -- hash grid cell size = patch_radius (µm)
    nx   : int         -- number of grid columns along the circumference
    ny   : int         -- number of grid rows along the height
    u_centers : (n_patches,) float -- patch centres in unrolled u = R*theta coords (µm)
    v_centers : (n_patches,) float -- patch centres in unrolled v = z coords (µm)
    buckets   : dict (col, row) -> int32 array of patch indices in that cell
    """

    def __init__(self, R, H, n_patches, patch_radius, rng, ka, kb, kc,
                 enabled=True, n_sites_per_patch=1,
                 enforce_nonoverlap=True, max_attempts=2_000_000):
        # Kinetic rates stored so release_captured can use them
        self.ka = ka   # association rate (µm³/s)
        self.kb = kb   # dissociation rate (s⁻¹)
        self.kc = kc   # internalization rate (s⁻¹)

        self.R = float(R)
        self.H = float(H)
        self.patch_radius = patch_radius
        self.n_patches = max(0, int(n_patches))

        # occ[j] tracks how many particles are currently bound to patch j.
        # Binding is refused when occ[j] == n_sites_per_patch (patch full).
        self.occ = np.zeros(self.n_patches, dtype=np.int32)

        # Disable if there are no valid patches
        self.enabled = (
            bool(enabled)
            and (self.n_patches > 0)
            and (patch_radius is not None)
            and (float(patch_radius) > 0.0)
        )

        if not self.enabled:
            self._init_empty(n_sites_per_patch)
            return

        self.radius = float(patch_radius)
        self.radius2 = self.radius * self.radius  # cached for fast distance comparisons

        # Place patch centres on the cylinder surface
        if (not enforce_nonoverlap) or (self.n_patches <= 1):
            # Independent uniform placement (Poisson process) -- overlaps allowed
            self.theta_centers = rng.uniform(0.0, 2.0 * np.pi, size=self.n_patches)
            self.z_centers = rng.uniform(-0.5 * self.H, 0.5 * self.H, size=self.n_patches)
        else:
            # Random Sequential Adsorption -- no two patches overlap
            self.theta_centers, self.z_centers = self._place_nonoverlapping(
                rng, self.n_patches, self.radius, max_attempts
            )

        self.n_sites_per_patch = int(n_sites_per_patch)
        self.n_sites_total = self.n_patches * self.n_sites_per_patch

        self._build_spatial_hash()

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def _init_empty(self, n_sites_per_patch):
        """Set up a null (disabled) patch set with no centres and zero capacity."""
        self.theta_centers = np.array([], dtype=float)
        self.z_centers = np.array([], dtype=float)
        self.radius = 0.0
        self.radius2 = 0.0
        self.n_sites_per_patch = int(n_sites_per_patch)
        self.n_sites_total = 0

    def _wrap_dtheta(self, dtheta):
        """Map an angular difference to the signed shorter arc in [-pi, pi].

        The cylinder is periodic in theta: angle 0 and angle 2*pi refer to the
        same physical point.  When measuring the arc-length distance between
        two patches at angles theta1 and theta2, there are always two arcs
        (the 'short way' and the 'long way' around the cylinder).  We always
        want the shorter one.

        For example, if dtheta = 5*pi/3 (≈300°), going that way is equivalent
        to going -pi/3 (≈60°) in the other direction -- so the returned value
        is -pi/3, giving arc length R * pi/3 instead of R * 5*pi/3.
        """
        return (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    def _wrap_dz(self, dz):
        """Map a z-displacement to the shortest signed path through the periodic z-boundary.

        The simulation box has periodic boundary conditions in z: a particle
        that exits through the top cap at z = H/2 re-enters at the bottom cap
        z = -H/2.  Similarly, two patches separated by a z-gap larger than H/2
        are actually closer to each other going 'through' the boundary.

        This function maps any raw displacement dz to the equivalent displacement
        in [-H/2, H/2] that uses the shortest path (possibly wrapping once).
        """
        return (dz + 0.5 * self.H) % self.H - 0.5 * self.H

    def _place_nonoverlapping(self, rng, M, min_dist, max_attempts):
        """Place M patch centres on the cylinder surface without any two overlapping.

        Uses Random Sequential Adsorption (RSA): propose a random location,
        accept it only if it is at least `min_dist` away from every already-placed patch, and repeat until all M patches are placed.

        Distances are measured on the unrolled rectangle with torus topology:
            d² = (R * delta_theta_shortest)² + (delta_z_shortest)²
        where `_wrap_dtheta` and `_wrap_dz` give the shortest-path differences.

        This is O(M²) in the worst case, but typical receptor densities are
        well below the jamming limit (~54 % coverage) so it converges quickly.

        Parameters
        ----------
        rng         : random generator
        M           : number of patches to place
        min_dist    : minimum centre-to-centre surface distance (µm);
                      set to patch_radius so patch discs do not touch
        max_attempts: hard limit on proposal count before raising RuntimeError

        Returns
        -------
        thetas : (M,) array of angular centres (rad)
        zs     : (M,) array of axial centres (µm)
        """
        min_dist2 = min_dist ** 2
        thetas = np.zeros(M)
        zs = np.zeros(M)
        count = 0
        attempts = 0

        while count < M:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to place {M} non-overlapping patches after {max_attempts} attempts. "
                    "Density may be near the RSA jamming limit (~54 % coverage). "
                    "Reduce n_patches, increase patch_radius, or raise max_attempts."
                )

            th_new = rng.uniform(0.0, 2.0 * np.pi)
            z_new = rng.uniform(-0.5 * self.H, 0.5 * self.H)

            if count == 0:
                thetas[0] = th_new
                zs[0] = z_new
                count += 1
                continue

            # Shortest-path distance on the torus from the candidate to every
            # already-placed patch
            dth = self._wrap_dtheta(th_new - thetas[:count])
            dz = self._wrap_dz(z_new - zs[:count])
            dist2 = (self.R * dth) ** 2 + dz ** 2

            if np.all(dist2 >= min_dist2):
                thetas[count] = th_new
                zs[count] = z_new
                count += 1

        return thetas, zs

    # ------------------------------------------------------------------
    # Spatial hash construction
    # ------------------------------------------------------------------

    def _build_spatial_hash(self):
        """Build the 2-D grid that maps unrolled-surface cells to patch lists.

        Motivation
        ----------
        Each time-step, every particle that hits the outer wall must be tested
        against all patches to see whether it landed on one.  Doing this naively
        is O(n_hits * n_patches).  A spatial hash reduces this to O(n_hits) on
        average by pre-grouping patches into grid cells.

        Construction
        ------------
        The cylinder surface is unrolled into a rectangle:
            u = R * theta   in [0, L),   L = 2*pi*R   (arc length along circumference)
            v = z           in [-H/2, H/2)

        The rectangle is divided into a grid of square cells of side = patch_radius.
        Each patch is registered in every cell that its disc overlaps (including
        cells in neighbouring rows and columns, and wrapping at u = 0 / u = L).
        At query time, a hit point is mapped to its cell in O(1), and only the
        patches registered in that cell need to be distance-tested.

        Attributes set
        --------------
        L         : float  -- circumference 2*pi*R
        cell      : float  -- grid cell side length (= patch_radius)
        nx, ny    : int    -- grid dimensions (columns × rows)
        u_centers : array  -- patch u-coordinates in the unrolled frame
        v_centers : array  -- patch v-coordinates in the unrolled frame
        buckets   : dict (col, row) -> int32 array of patch indices
        """
        self.L = 2.0 * np.pi * self.R       # circumference (µm)
        self.cell = float(self.radius)       # grid cell size = patch radius
        self.nx = max(1, int(np.floor(self.L / self.cell)))  # columns
        self.ny = max(1, int(np.floor(self.H / self.cell)))  # rows

        # Convert patch centres to unrolled (u, v) coordinates
        u = (self.R * self.theta_centers) % self.L
        v = self.z_centers.copy()
        self.u_centers = u.astype(np.float64)
        self.v_centers = v.astype(np.float64)

        # Register each patch in all cells that its disc overlaps
        buckets = {}
        r = self.radius
        for j in range(self.n_patches):
            cx = int(self.u_centers[j] / self.cell)           # centre column
            cy = int((self.v_centers[j] + 0.5 * self.H) / self.cell)  # centre row
            # Number of cells the disc can span in each direction
            dx = dy = int(np.ceil(r / self.cell))
            for ix in range(cx - dx, cx + dx + 1):
                ixw = ix % self.nx   # wrap column index for periodicity in u
                for iy in range(cy - dy, cy + dy + 1):
                    if 0 <= iy < self.ny:
                        buckets.setdefault((ixw, iy), []).append(j)

        # Store as numpy arrays for fast indexing during lookup
        self.buckets = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def patch_index_of_hits(self, S: np.ndarray) -> np.ndarray:
        """Find which patch (if any) each outer-wall hit point lands on.

        A particle that crossed the outer wall at 3-D point S = (x, y, z) is
        converted to unrolled coordinates (u, v) on the cylinder surface:
            u = R * (arctan2(y, x) mod 2π)   -- arc-length along circumference
            v = z                              -- height

        The cell containing (u, v) is looked up in the spatial hash.  Only the
        patches registered in that cell need to be distance-tested, reducing the
        average work from O(n_patches) to O(1).

        A hit point is 'on' patch j if its shortest-path distance to the patch
        centre (accounting for the periodic torus topology) is <= patch_radius.

        Parameters
        ----------
        S : (M, 3) array
            3-D positions of wall-hit points, all at radius R.

        Returns
        -------
        out : (M,) int32 array
            out[i] = index of the patch hit point i landed on, or -1 if it
            hit the open wall between patches.
        """
        M = S.shape[0]
        if (not self.enabled) or (self.n_patches == 0) or (M == 0):
            return -np.ones(M, dtype=np.int32)

        # Convert 3-D hit points to unrolled 2-D surface coordinates
        theta = np.arctan2(S[:, 1], S[:, 0])            # angle in [-pi, pi]
        u = (self.R * (theta % (2 * np.pi))) % self.L   # arc-length in [0, L)
        v = ((S[:, 2] + 0.5 * self.H) % self.H) - 0.5 * self.H  # z wrapped to [-H/2, H/2)

        # Grid cell index for each hit point
        cx = (u / self.cell).astype(np.int32)
        cy = ((v + 0.5 * self.H) / self.cell).astype(np.int32)

        out = -np.ones(M, dtype=np.int32)
        r2 = self.radius2

        for i in range(M):
            # Retrieve the candidate patches registered in this grid cell
            cand = self.buckets.get((int(cx[i]) % self.nx, int(cy[i])))
            if cand is None:
                continue   # no patch centre anywhere near this cell

            # Shortest-path distances from hit point to each candidate patch
            # (must use periodic wrap in u and v, not raw difference)
            du = u[i] - self.u_centers[cand]
            du = (du + 0.5 * self.L) % self.L - 0.5 * self.L   # wrap in u

            dv = v[i] - self.v_centers[cand]
            dv = (dv + 0.5 * self.H) % self.H - 0.5 * self.H   # wrap in v

            dist2 = du * du + dv * dv
            jmin = np.argmin(dist2)
            if dist2[jmin] <= r2:
                out[i] = cand[jmin]   # assign to the nearest patch within radius

        return out

    def is_absorbing(self, S: np.ndarray) -> np.ndarray:
        """Return a boolean mask: True for each hit point that landed on a patch.

        Convenience wrapper around patch_index_of_hits.  Used in the transient
        simulation where we only need to know whether each hit is absorbed, not
        which specific patch index it belongs to.

        Parameters
        ----------
        S : (M, 3) array of wall-hit points

        Returns
        -------
        (M,) bool array -- True if the hit is on a patch, False if on open wall
        """
        return self.patch_index_of_hits(S) >= 0

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    def select_binders(self, idx_hit, patch_idx_hit, t_hit, dt, c_outer_shell, rng):
        """Choose which particles actually bind among those that hit a patch this step.

        This method is called after ka-gating (Erban-Chapman probability) has
        already been applied in the simulation loop, so every particle arriving
        here has 'won' the binding lottery.  The remaining constraint is receptor
        capacity: if a patch is already at full occupancy, additional particles
        that arrive in the same time step cannot bind.

        Tie-breaking rule: when more particles hit the same patch than there are
        free sites, the ones with the smallest t_hit (earliest arrival within the
        time step) are preferred.  This approximates the physical notion that the
        first molecule to actually reach the receptor surface binds, and later
        arrivals find it occupied.

        After binding, occ[patch_idx] is incremented for each new binder.  The
        corresponding particles are removed from free diffusion by the simulation
        loop (their positions are set to NaN).

        Parameters
        ----------
        idx_hit       : int array -- global particle indices that hit a patch
        patch_idx_hit : int array -- corresponding patch index for each particle
        t_hit         : float array (length = total particles) -- fractional time
                        within the step at which each particle hit the wall;
                        indexed by global particle index
        dt            : float -- time step size (used only for guard check)
        c_outer_shell : float -- concentration near r=R (not currently used;
                        reserved for future concentration-dependent binding)
        rng           : random generator (not used; reserved for future stochastic
                        sub-step selection)

        Returns
        -------
        (K, 2) int64 array
            Each row is [particle_idx, patch_idx] for a newly bound particle.
            K <= len(idx_hit).  Empty array if nothing binds.
        """
        if (not self.enabled) or (dt <= 0.0):
            return np.empty((0, 2), dtype=np.int64)

        on = (patch_idx_hit >= 0)
        if not np.any(on):
            return np.empty((0, 2), dtype=np.int64)

        p_idx = idx_hit[on].astype(np.int64, copy=False)
        q_idx = patch_idx_hit[on].astype(np.int32, copy=False)
        thit = t_hit[p_idx]  # hit time for each candidate, indexed by global particle id

        # Sort so that within each patch the earliest-arriving particle comes first
        order = np.lexsort((thit, q_idx))
        p_idx = p_idx[order]
        q_idx = q_idx[order]

        cap = int(self.n_sites_per_patch)
        out = []
        i = 0
        n = p_idx.size
        while i < n:
            q = int(q_idx[i])
            # Find the run of particles targeting the same patch q
            j = i + 1
            while j < n and int(q_idx[j]) == q:
                j += 1
            # Only the first `free` particles in this run can bind
            free = cap - int(self.occ[q])
            if free > 0:
                take = p_idx[i:j][:free]
                if take.size:
                    self.occ[q] += int(take.size)
                    out.extend((int(pi), q) for pi in take)
            i = j

        if not out:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(out, dtype=np.int64)

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release_captured(self, idx_captured, dt, rng):
        """Stochastically release bound particles at the end of each time step.

        Each bound particle has a total unbinding rate k_off = kb + kc.  In a
        time step dt, the probability of unbinding is:

            p_rel = 1 - exp(-(kb + kc) * dt)

        This is the exact discrete-time probability for a Poisson process, so
        the exponential ensures there is no systematic bias regardless of dt.

        A particle that unbinds is then routed either back to the outer bulk
        (dissociation, probability kb / (kb + kc)) or toward the inner cylinder
        as an internalized/absorbed particle (probability kc / (kb + kc)).

        After release the patch occupancy occ[patch_idx] is decremented for
        every released particle so the site becomes available for future binding.

        Parameters
        ----------
        idx_captured : (M, 2) int64 array
            Currently bound particles.  Each row is [particle_idx, patch_idx].
        dt  : float -- time step (s)
        rng : random generator

        Returns
        -------
        still_captured : (?, 2) int64 -- particles that remain bound this step
        released_outer : (?, 2) int64 -- particles released to bulk near r=R
                         (will be respawned in a thin shell just inside r=R)
        released_inner : (?, 2) int64 -- particles internalized, counted as
                         absorbed (will be respawned near r=a)
        """
        idx_captured = np.asarray(idx_captured, dtype=np.int64)
        M = idx_captured.shape[0]
        if (M == 0) or (dt <= 0.0):
            empty = np.empty((0, 2), dtype=np.int64)
            return idx_captured, empty, empty

        kb = float(max(0.0, self.kb))
        kc = float(max(0.0, self.kc))
        ksum = kb + kc
        if ksum <= 0.0:
            # Irreversible binding: nothing is ever released
            empty = np.empty((0, 2), dtype=np.int64)
            return idx_captured, empty, empty

        # Each bound particle releases independently with the same probability
        p_rel = float(np.clip(1.0 - np.exp(-ksum * dt), 0.0, 1.0))
        rel_mask = rng.random(M) < p_rel

        if not np.any(rel_mask):
            empty = np.empty((0, 2), dtype=np.int64)
            return idx_captured, empty, empty

        rel = idx_captured[rel_mask]
        still = idx_captured[~rel_mask]

        # Split released particles: fraction kb/(kb+kc) go outer, rest go inner
        outer_mask = rng.random(rel.shape[0]) < (kb / ksum)
        rel_outer = rel[outer_mask]   # will re-enter diffusion near r=R
        rel_inner = rel[~outer_mask]  # will be counted as absorbed

        # Free the receptor sites so new particles can bind
        released_patch_idx = rel[:, 1].astype(np.int64, copy=False)
        if released_patch_idx.size:
            np.add.at(self.occ, released_patch_idx, -1)
            self.occ[:] = np.maximum(self.occ, 0)  # safety clamp against negatives

        return still, rel_outer, rel_inner
