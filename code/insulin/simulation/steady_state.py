"""Steady-state random walk simulation in a cylindrical annulus."""
import math
import numpy as np
from dataclasses import dataclass
from ..params import SimulationParams
from .geometry import rand_points_in_shell, segment_hits_cylinder
from .boundaries import reflect_inner, reflect_outer, wrap_periodic_z
from .patches import PatchSet


@dataclass
class SteadyStateResult:
    """Output of a steady-state simulation run."""
    c0_total: float        # global concentration from total particles
    c0_measured: float     # extrapolated concentration near r=a
    rate: float            # absorption rate (particles/s)
    rate_normalized: float # rate / c0_measured (um^3/s)
    r_centers: np.ndarray | None = None
    profile_times: np.ndarray | None = None
    concentrations: np.ndarray | None = None


class SteadyStateSimulation:
    """Steady-state random walk in a cylindrical shell.

    Particles diffuse in the annulus a <= r <= R with periodic z-boundaries.
    Absorbed particles at r=R are respawned near r=a to maintain steady state.
    Supports patchy absorption with finite binding kinetics.
    """

    def __init__(self, params: SimulationParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng

    def run(self, concentration_profile=False,
            n_profile_times=10, n_r_bins=40) -> SteadyStateResult:
        p = self.p
        rng = self.rng

        N0 = int(p.N)
        D, a, R, H = float(p.D), float(p.a), float(p.R), float(p.H)
        dt, T = float(p.dt), float(p.T)
        eps_respawn = p.eps_respawn
        t_relax = float(p.t_relax)
        ka, kb, kc = float(p.ka), float(p.kb), float(p.kc)

        vol_shell = np.pi * H * (R**2 - a**2)
        c0_total = (N0 / vol_shell) if vol_shell > 0 else 0.0

        step_sigma = np.sqrt(2.0 * D * dt) if (D > 0 and dt > 0) else 0.0
        n_steps = int(np.ceil(T / dt)) if dt > 0 else 0

        n_relax_steps = int(t_relax / dt) if dt > 0 else 0
        n_relax_steps = min(max(n_relax_steps, 0), max(n_steps - 1, 0))

        absorbed_after_relax = 0
        measure_steps = 0

        # --- C0 extrapolation setup (two-shell linear fit near r=a) ---
        eps_c0 = float(p.eps_c0)
        if eps_c0 > 0 and R > a:
            eps_c0 = min(eps_c0, (R - a) / 3.0)

        if eps_c0 > 0:
            r0_minus, r0_plus = a, a + eps_c0
            r1_minus, r1_plus = a + 2 * eps_c0, a + 3 * eps_c0
            vol0 = np.pi * H * (r0_plus**2 - r0_minus**2)
            vol1 = np.pi * H * (r1_plus**2 - r1_minus**2)
        else:
            r0_minus = r0_plus = r1_minus = r1_plus = a
            vol0 = vol1 = 1.0

        c0_sum = c1_sum = 0.0
        c0_cnt = c1_cnt = 0

        def c0_extrapolate():
            if eps_c0 <= 0 or c0_cnt == 0 or c1_cnt == 0:
                return c0_total
            C0 = c0_sum / c0_cnt
            C1 = c1_sum / c1_cnt
            r0c = 0.5 * (r0_minus + r0_plus)
            r1c = 0.5 * (r1_minus + r1_plus)
            if r1c == r0c:
                return float(C0)
            m = (C1 - C0) / (r1c - r0c)
            b = C0 - m * r0c
            return float(m * a + b)

        # --- Outer shell concentration ---
        eps_out = float(p.eps_out)
        r_out_minus, r_out_plus = R - eps_out, R
        vol_out = np.pi * H * (r_out_plus**2 - r_out_minus**2)

        # --- Concentration profile snapshots ---
        if concentration_profile:
            r_edges = np.linspace(a, R, n_r_bins + 1)
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            shell_volumes = np.pi * H * (r_edges[1:]**2 - r_edges[:-1]**2)
            profile_steps = np.linspace(0, max(n_steps - 1, 0), n_profile_times, dtype=int)
            profile_times = profile_steps * dt
            concentrations_time = np.full((n_profile_times, n_r_bins), np.nan)
            prof_i = 0
        else:
            r_centers = profile_times = concentrations_time = None
            profile_steps = None
            prof_i = 0

        # --- Patches ---
        patches_enabled = (p.n_patches > 0 and p.patch_radius is not None)
        patches = PatchSet(
            enabled=patches_enabled,
            n_patches=p.n_patches,
            patch_radius=p.patch_radius,
            R=R, H=H, rng=rng,
            ka=ka, kb=kb, kc=kc,
        )

        # --- Initial positions ---
        X = rand_points_in_shell(N0, a, R, H, R - a, rng, "inner")
        idx_captured = np.empty((0, 2), dtype=np.int64)

        for k in range(n_steps):
            n_abs = 0
            idx_bind = np.empty(0, dtype=int)
            free_mask = np.isfinite(X[:, 0])

            # 1) Diffusion
            dX = np.zeros_like(X)
            dX[free_mask] = rng.normal(0.0, step_sigma, size=(free_mask.sum(), 3))
            X_next = X + dX

            # 2) Outer wall hits
            hit_mask = np.zeros(X.shape[0], dtype=bool)
            t_hit = np.zeros(X.shape[0], dtype=float)
            S = np.zeros_like(X)

            if np.any(free_mask):
                hit_sub, t_hit_sub, S_sub = segment_hits_cylinder(X[free_mask], dX[free_mask], R, snap=True)
                idx_free = np.flatnonzero(free_mask)
                hit_mask[idx_free] = hit_sub
                t_hit[idx_free] = t_hit_sub
                S[idx_free] = S_sub

            # 3) Binding + reflection at r=R
            absorbed_now_mask = np.zeros(X.shape[0], dtype=bool)

            X_free = X[free_mask]
            rho_free = np.sqrt(X_free[:, 0]**2 + X_free[:, 1]**2)
            mask_out = (rho_free >= r_out_minus) & (rho_free < r_out_plus)
            c_outer_shell = float(mask_out.sum()) / float(vol_out)

            if not patches_enabled:
                absorbed_now_mask = hit_mask.copy()
            else:
                if np.any(hit_mask):
                    idx_hit = np.where(hit_mask)[0]
                    S_hit = S[idx_hit]
                    patch_idx_hit = patches.patch_index_of_hits(S_hit)

                    on = (patch_idx_hit >= 0)
                    idx_on_patch = idx_hit[on]
                    patch_idx_on = patch_idx_hit[on]

                    pos_in_on = np.full(X.shape[0], -1, dtype=int)
                    pos_in_on[idx_on_patch] = np.arange(idx_on_patch.size)

                    # ka gating (Erban-Chapman)
                    if idx_on_patch.size and (ka != math.inf):
                        patch_area = np.pi * (patches.radius ** 2)
                        kappa = ka / patch_area
                        p_scale = np.sqrt(np.pi * dt / D)
                        p_bind = (kappa * p_scale) / (1 + (kappa * p_scale) / 2)
                        if p_bind > 1.0:
                            p_bind = 1.0
                        idx_ka = idx_on_patch[rng.random(idx_on_patch.size) < p_bind]
                    else:
                        idx_ka = idx_on_patch

                    if (kb > 0) or (kc != math.inf):
                        if idx_ka.size:
                            patch_idx_ka = patch_idx_on[pos_in_on[idx_ka]]
                            bind_pairs = patches.select_binders(
                                idx_hit=idx_ka, patch_idx_hit=patch_idx_ka,
                                t_hit=t_hit, dt=dt,
                                c_outer_shell=c_outer_shell, rng=rng,
                            )
                            idx_bind = bind_pairs[:, 0] if bind_pairs.size else np.empty(0, dtype=int)
                        else:
                            bind_pairs = np.empty((0, 2), dtype=int)
                            idx_bind = np.empty(0, dtype=int)

                        if idx_bind.size:
                            absorbed_now_mask[idx_bind] = True
                            X_next[idx_bind, :] = np.nan
                            idx_captured = np.vstack([idx_captured, bind_pairs])

                        idx_ref = np.setdiff1d(idx_hit, idx_bind, assume_unique=False)
                        if idx_ref.size:
                            X_next[idx_ref] = reflect_outer(S[idx_ref], dX[idx_ref], t_hit[idx_ref], R)
                    else:
                        idx_bind = idx_ka
                        if idx_bind.size:
                            absorbed_now_mask[idx_bind] = True
                            X_next[idx_bind, :] = np.nan

                        idx_ref = np.setdiff1d(idx_hit, idx_bind, assume_unique=False)
                        if idx_ref.size:
                            X_next[idx_ref] = reflect_outer(S[idx_ref], dX[idx_ref], t_hit[idx_ref], R)

            # 4) Inner reflection
            free_next = np.isfinite(X_next[:, 0])
            alive_mask = free_next & (~absorbed_now_mask)
            if np.any(alive_mask):
                X_next[alive_mask] = reflect_inner(X_next[alive_mask], a)

            # 5) Release captured
            if (kb > 0) or (kc != math.inf):
                idx_captured, relR_pairs, relA_pairs = patches.release_captured(idx_captured, dt, rng)
                idx_rel_R = relR_pairs[:, 0] if relR_pairs.size else np.empty(0, dtype=int)
                idx_rel_a = relA_pairs[:, 0] if relA_pairs.size else np.empty(0, dtype=int)

                if idx_rel_R.size:
                    X_next[idx_rel_R] = rand_points_in_shell(len(idx_rel_R), a, R, H, eps_out, rng, "outer")
                if idx_rel_a.size:
                    X_next[idx_rel_a] = rand_points_in_shell(len(idx_rel_a), a, R, H, eps_respawn, rng, "inner")
                n_abs = idx_rel_a.size
            else:
                n_abs = int(absorbed_now_mask.sum())
                if n_abs:
                    X_next[absorbed_now_mask] = rand_points_in_shell(n_abs, a, R, H, eps_respawn, rng, "inner")

            # 6) Periodic z
            free_next = np.isfinite(X_next[:, 0])
            if np.any(free_next):
                X_next[free_next] = wrap_periodic_z(X_next[free_next], H)

            X = X_next

            # Concentration profile snapshot
            if concentration_profile and (prof_i < len(profile_steps)) and (k == profile_steps[prof_i]):
                free_now = np.isfinite(X[:, 0])
                r = np.sqrt(X[free_now, 0]**2 + X[free_now, 1]**2)
                counts, _ = np.histogram(r, bins=r_edges)
                concentrations_time[prof_i, :] = counts / shell_volumes
                prof_i += 1

            # Post-relax measurements
            if k >= n_relax_steps:
                absorbed_after_relax += n_abs
                measure_steps += 1

                if eps_c0 > 0:
                    free_now = np.isfinite(X[:, 0])
                    r = np.sqrt(X[free_now, 0]**2 + X[free_now, 1]**2)
                    mask0 = (r >= r0_minus) & (r < r0_plus)
                    mask1 = (r >= r1_minus) & (r < r1_plus)
                    c0_sum += mask0.sum() / vol0
                    c1_sum += mask1.sum() / vol1
                    c0_cnt += 1
                    c1_cnt += 1

        c0_measured = max(float(c0_extrapolate()), 1e-300)

        if measure_steps > 0:
            T_meas = measure_steps * dt
            rate_est = absorbed_after_relax / T_meas
        else:
            rate_est = 0.0

        print(f"c0_measured={c0_measured:.4g}  c0_input={float(p.C0):.4g}  "
              f"n_patches={p.n_patches}  ka={ka}  kb={kb}  kc={kc}")

        return SteadyStateResult(
            c0_total=float(c0_total),
            c0_measured=float(c0_measured),
            rate=float(rate_est),
            rate_normalized=float(rate_est / c0_measured),
            r_centers=r_centers,
            profile_times=profile_times,
            concentrations=concentrations_time,
        )
