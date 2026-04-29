"""Receptor patches on the outer cylinder surface.

PatchSet manages placement, spatial lookup, occupancy tracking,
and binding/release kinetics for discrete receptor patches.
"""
import numpy as np


class PatchSet:
    """Disk-like receptor patches on the outer cylinder r=R.

    Each patch is a circular region on the unrolled cylinder surface
    (R*theta, z), with spatial hashing for fast hit-to-patch lookup.

    Tracks per-patch occupancy and implements release kinetics (kb, kc).
    """

    def __init__(self, R, H, n_patches, patch_radius, rng, ka, kb, kc,
                 enabled=True, n_sites_per_patch=1,
                 enforce_nonoverlap=True, max_attempts=2_000_000):
        self.ka = ka
        self.kb = kb
        self.kc = kc
        self.R = float(R)
        self.H = float(H)
        self.patch_radius = patch_radius
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

        # Place centers via RSA or random
        if (not enforce_nonoverlap) or (self.n_patches <= 1):
            self.theta_centers = rng.uniform(0.0, 2.0 * np.pi, size=self.n_patches)
            self.z_centers = rng.uniform(-0.5 * self.H, 0.5 * self.H, size=self.n_patches)
        else:
            self.theta_centers, self.z_centers = self._place_nonoverlapping(
                rng, self.n_patches, self.radius, max_attempts
            )

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
        return (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    def _wrap_dz(self, dz):
        return (dz + 0.5 * self.H) % self.H - 0.5 * self.H

    def _place_nonoverlapping(self, rng, M, min_dist, max_attempts):
        """Random Sequential Adsorption with periodic (torus) topology."""
        R, H = self.R, self.H
        min_dist2 = min_dist ** 2
        thetas = np.zeros(M)
        zs = np.zeros(M)
        count = 0
        attempts = 0

        while count < M:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to place {M} patches after {max_attempts} attempts. "
                    "Density may be near the jamming limit."
                )

            th_new = rng.uniform(0.0, 2.0 * np.pi)
            z_new = rng.uniform(-0.5 * H, 0.5 * H)

            if count == 0:
                thetas[0] = th_new
                zs[0] = z_new
                count += 1
                continue

            dth = self._wrap_dtheta(th_new - thetas[:count])
            dz = self._wrap_dz(z_new - zs[:count])
            dist2 = (R * dth) ** 2 + dz ** 2

            if np.all(dist2 >= min_dist2):
                thetas[count] = th_new
                zs[count] = z_new
                count += 1

        return thetas, zs

    def patch_index_of_hits(self, S: np.ndarray) -> np.ndarray:
        """For each hit point, return the patch index it lands on (-1 if none)."""
        M = S.shape[0]
        if (not self.enabled) or (self.n_patches == 0) or (M == 0):
            return -np.ones(M, dtype=np.int32)

        theta = np.arctan2(S[:, 1], S[:, 0])
        u = (self.R * (theta % (2 * np.pi))) % self.L
        v = ((S[:, 2] + 0.5 * self.H) % self.H) - 0.5 * self.H

        cx = (u / self.cell).astype(np.int32)
        cy = ((v + 0.5 * self.H) / self.cell).astype(np.int32)

        out = -np.ones(M, dtype=np.int32)
        r2 = self.radius2

        for i in range(M):
            key = (int(cx[i]) % self.nx, int(cy[i]))
            cand = self.buckets.get(key)
            if cand is None:
                continue

            du = u[i] - self.u_centers[cand]
            du = (du + 0.5 * self.L) % self.L - 0.5 * self.L

            dv = v[i] - self.v_centers[cand]
            dv = (dv + 0.5 * self.H) % self.H - 0.5 * self.H

            dist2 = du * du + dv * dv
            jmin = np.argmin(dist2)
            if dist2[jmin] <= r2:
                out[i] = cand[jmin]

        return out

    def is_absorbing(self, S: np.ndarray) -> np.ndarray:
        """Boolean mask: True if hit point is on any patch."""
        return self.patch_index_of_hits(S) >= 0

    def select_binders(self, idx_hit, patch_idx_hit, t_hit, dt, c_outer_shell, rng):
        """Choose earliest hits per patch up to free capacity.

        Assumes ka-gating already happened upstream.
        Returns (N, 2) array of [particle_idx, patch_idx] pairs.
        """
        if (not self.enabled) or (dt <= 0.0):
            return np.empty((0, 2), dtype=np.int64)

        on = (patch_idx_hit >= 0)
        if not np.any(on):
            return np.empty((0, 2), dtype=np.int64)

        p_idx = idx_hit[on].astype(np.int64, copy=False)
        q_idx = patch_idx_hit[on].astype(np.int32, copy=False)
        thit = t_hit[p_idx]

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

    def release_captured(self, idx_captured, dt, rng):
        """Process release events for captured particles.

        Returns (still_captured, released_outer, released_inner),
        each shape (?, 2) of [particle_idx, patch_idx].
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

        p_rel = float(np.clip(1.0 - np.exp(-ksum * dt), 0.0, 1.0))

        rel_mask = rng.random(M) < p_rel
        if not np.any(rel_mask):
            return idx_captured, np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        rel = idx_captured[rel_mask]
        still = idx_captured[~rel_mask]

        p_outer = kb / ksum
        outer_mask = rng.random(rel.shape[0]) < p_outer
        rel_outer = rel[outer_mask]
        rel_inner = rel[~outer_mask]

        rel_patch_idx = rel[:, 1].astype(np.int64, copy=False)
        if rel_patch_idx.size:
            np.add.at(self.occ, rel_patch_idx, -1)
            self.occ[:] = np.maximum(self.occ, 0)

        return still, rel_outer, rel_inner

    def _build_spatial_hash(self):
        self.L = 2.0 * np.pi * self.R
        self.cell = float(self.radius)
        self.nx = max(1, int(np.floor(self.L / self.cell)))
        self.ny = max(1, int(np.floor(self.H / self.cell)))

        u = (self.R * self.theta_centers) % self.L
        v = self.z_centers.copy()
        self.u_centers = u.astype(np.float64, copy=False)
        self.v_centers = v.astype(np.float64, copy=False)

        buckets = {}
        r = self.radius
        for j in range(self.n_patches):
            cx = int(self.u_centers[j] / self.cell)
            cy = int((self.v_centers[j] + 0.5 * self.H) / self.cell)
            dx = int(np.ceil(r / self.cell))
            dy = int(np.ceil(r / self.cell))
            for ix in range(cx - dx, cx + dx + 1):
                ixw = ix % self.nx
                for iy in range(cy - dy, cy + dy + 1):
                    if 0 <= iy < self.ny:
                        buckets.setdefault((ixw, iy), []).append(j)

        self.buckets = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}
