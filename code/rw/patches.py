import numpy as np


class PatchSetCyl:
    """
    Disk-like patches on the OUTER cylinder r = R.

    Each patch is a circular region on the cylinder surface, defined by a center
    (theta_c, z_c) and a surface radius 'patch_radius' measured on the
    unrolled (R*theta, z) plane:

        (R * Δθ)^2 + (z - z_c)^2 <= patch_radius^2

    Supports an optional 'mode_2d' where patches are infinite stripes in z
    and only differ by theta.

    Now also supports finite receptor capacity + kon/koff:
      - Total receptor sites = n_patches * n_sites_per_patch.
      - At any time, n_bound of them are occupied.
      - A hit on a patch binds with probability p_bind if there are free sites.
      - Bound receptors unbind with rate koff.
    """

    def __init__(self, R, H, n_patches, patch_radius, rng,ka,kb,kc,
                 enabled=True,
                 chunk_size: int = 512,
                 n_sites_per_patch: int = 1,
                 enforce_nonoverlap: bool = True,
                 max_attempts: int = 2_000_000):
        self.ka=ka
        self.kb=kb
        self.kc=kc
        self.R = float(R)
        self.H = float(H)
        self.patch_radius = patch_radius
        self.chunk_size = int(chunk_size)

        self.n_patches = max(0, int(n_patches))
        self.occ = np.zeros(self.n_patches, dtype=np.int32)

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
        self.radius2 = self.radius * self.radius

        # ---- Place centers ----
        if (not enforce_nonoverlap) or (self.n_patches <= 1):
            # Pure Poisson distribution (overlapping allowed)
            self.theta_centers = rng.uniform(0.0, 2.0 * np.pi, size=self.n_patches)
            self.z_centers = rng.uniform(-0.5 * self.H, 0.5 * self.H, size=self.n_patches)
        else:
            # Random Non-Overlapping (RSA)
            self.theta_centers, self.z_centers = self._place_random_nonoverlapping(
                rng=rng,
                M=self.n_patches,
                min_dist= self.radius,
                max_attempts=max_attempts
            )

        # ---- Occupancy ----
        self.n_sites_per_patch = int(n_sites_per_patch)
        self.n_sites_total = self.n_patches * self.n_sites_per_patch
        self._build_spatial_hash()

    def _init_empty(self, n_sites_per_patch):
        self.theta_centers = np.array([], dtype=float)
        self.z_centers = np.array([], dtype=float)
        self.radius = 0.0
        self.radius2 = 0.0
        self.n_sites_per_patch = int(n_sites_per_patch)
        self.n_sites_total = 0

    def _wrap_dtheta(self, dtheta):
        """Shortest angular distance on circle [-pi, pi]"""
        return (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    def _wrap_dz(self, dz):
        """Shortest vertical distance on periodic cylinder [-H/2, H/2]"""
        return (dz + 0.5 * self.H) % self.H - 0.5 * self.H

    def _place_random_nonoverlapping(self, rng, M, min_dist, max_attempts):
        """
        Rejection sampling with Minimum Image Convention (Torus topology).
        """
        R = self.R
        H = self.H
        min_dist2 = min_dist ** 2

        thetas = np.zeros(M)
        zs = np.zeros(M)

        count = 0
        attempts = 0

        while count < M:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to place {M} random patches after {max_attempts} attempts. "
                    "The requested density might be near the jamming limit."
                )

            # 1. Propose a random position
            th_new = rng.uniform(0.0, 2.0 * np.pi)
            z_new = rng.uniform(-0.5 * H, 0.5 * H)

            if count == 0:
                thetas[count] = th_new
                zs[count] = z_new
                count += 1
                continue

            # 2. Check against all existing patches using Periodic Distance
            dth = self._wrap_dtheta(th_new - thetas[:count])
            dz = self._wrap_dz(z_new - zs[:count])

            # Squared distance in the unrolled periodic plane
            dist2 = (R * dth) ** 2 + dz ** 2

            if np.all(dist2 >= min_dist2):
                thetas[count] = th_new
                zs[count] = z_new
                count += 1

        return thetas, zs

    def patch_index_of_hits(self, S: np.ndarray) -> np.ndarray:
        M = S.shape[0]
        if (not self.enabled) or (self.n_patches == 0) or (M == 0):
            return -np.ones(M, dtype=np.int32)

        # Convert hit points to unrolled coords
        x = S[:, 0]
        y = S[:, 1]
        z = S[:, 2]

        theta = np.arctan2(y, x)  # [-pi, pi]
        u = (self.R * (theta % (2 * np.pi))) % self.L
        v = ((z + 0.5 * self.H) % self.H) - 0.5 * self.H

        cx = (u / self.cell).astype(np.int32)
        cy = ((v + 0.5 * self.H) / self.cell).astype(np.int32)

        out = -np.ones(M, dtype=np.int32)
        r2 = self.radius2

        # Loop over hits (fast because few candidates each). This is still Python,
        # but it’s M_hits not M_hits*n_patches.
        for i in range(M):
            key = (int(cx[i]) % self.nx, int(cy[i]))
            cand = self.buckets.get(key, None)
            if cand is None:
                continue

            # compute periodic distance in unrolled u (wrap on [0,L))
            du = u[i] - self.u_centers[cand]
            du = (du + 0.5 * self.L) % self.L - 0.5 * self.L

            dv = v[i] - self.v_centers[cand]
            dv = (dv + 0.5 * self.H) % self.H - 0.5 * self.H

            dist2 = du * du + dv * dv
            jmin = np.argmin(dist2)
            if dist2[jmin] <= r2:
                out[i] = cand[jmin]

        return out

    def select_binders_with_occupancy(self, idx_hit, patch_idx_hit, t_hit, dt, c_outer_shell, rng):
        """
        Occupancy-only: choose earliest hits per patch up to free capacity.
        Assumes ka-gating already happened upstream.
        """
        if (not self.enabled) or (dt <= 0.0):
            return np.empty((0, 2), dtype=np.int64)

        on = (patch_idx_hit >= 0)
        if not np.any(on):
            return np.empty((0, 2), dtype=np.int64)

        p_idx = idx_hit[on].astype(np.int64, copy=False)  # global particle indices
        q_idx = patch_idx_hit[on].astype(np.int32, copy=False)  # patch indices aligned with p_idx
        thit = t_hit[p_idx]  # hit-times per particle (global t_hit)

        # sort by patch then by hit time (earliest first within patch)
        order = np.lexsort((thit, q_idx))
        p_idx = p_idx[order]
        q_idx = q_idx[order]

        cap = int(self.n_sites_per_patch)
        out = []

        i = 0
        n = p_idx.size
        while i < n:
            q = int(q_idx[i])
            j = i + 1
            while j < n and int(q_idx[j]) == q:
                j += 1

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




    def split_captured_by_release(
            self,
            idx_captured: np.ndarray,  # shape (M,2): [particle_idx, patch_idx]
            dt: float,
            rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (still, rel_outer, rel_inner), each shape (?,2).
        Also decrements self.occ for released patch indices.
        """
        idx_captured = np.asarray(idx_captured, dtype=np.int64)
        M = idx_captured.shape[0]
        if (M == 0) or (dt <= 0.0):
            return idx_captured, np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        kb = float(max(0.0, self.kb))
        kc = float(max(0.0, self.kc))
        ksum = kb + kc
        if ksum <= 0.0:
            return idx_captured, np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        p_rel = float(1.0 - np.exp(-ksum * dt))
        p_rel = float(np.clip(p_rel, 0.0, 1.0))

        rel_mask = (rng.random(M) < p_rel)
        if not np.any(rel_mask):
            return idx_captured, np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        rel = idx_captured[rel_mask]
        still = idx_captured[~rel_mask]

        p_outer = kb / ksum
        outer_mask = (rng.random(rel.shape[0]) < p_outer)
        rel_outer = rel[outer_mask]
        rel_inner = rel[~outer_mask]

        # FREE the patches corresponding to released binders
        rel_patch_idx = rel[:, 1].astype(np.int64, copy=False)
        if rel_patch_idx.size:
            # decrement one per released particle
            np.add.at(self.occ, rel_patch_idx, -1)
            # safety clamp
            self.occ[:] = np.maximum(self.occ, 0)

        return still, rel_outer, rel_inner

    def _build_spatial_hash(self):
        # Unrolled dimensions
        self.L = 2.0 * np.pi * self.R
        self.cell = float(self.radius)  # cell size; can try radius/2 for fewer false positives
        self.nx = max(1, int(np.floor(self.L / self.cell)))
        self.ny = max(1, int(np.floor(self.H / self.cell)))

        # Precompute patch centers in unrolled coords
        u = (self.R * self.theta_centers) % self.L
        v = self.z_centers.copy()  # already in [-H/2, H/2]

        self.u_centers = u.astype(np.float64, copy=False)
        self.v_centers = v.astype(np.float64, copy=False)

        # Map from (cx,cy) -> list of patch indices
        buckets = {}
        r = self.radius
        for j in range(self.n_patches):
            cx = int(self.u_centers[j] / self.cell)
            cy = int((self.v_centers[j] + 0.5 * self.H) / self.cell)

            # Patch overlaps neighboring cells too
            # how many cells to cover radius
            dx = int(np.ceil(r / self.cell))
            dy = int(np.ceil(r / self.cell))

            for ix in range(cx - dx, cx + dx + 1):
                ixw = ix % self.nx
                for iy in range(cy - dy, cy + dy + 1):
                    if 0 <= iy < self.ny:
                        buckets.setdefault((ixw, iy), []).append(j)

        # Store as dict of numpy arrays for speed
        self.buckets = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}