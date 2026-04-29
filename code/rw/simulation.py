import math

import numpy as np
from .geometry import (
    segment_hits_cylinder,rand_points_in_thin_cyl_volume        # intersection with r = R (outer cylinder)
)
from .boundaries import (
    reflect_from_inner_cylinder,  # fix any r < a
    apply_periodic_caps, reflect_after_outer_hit_specular_cyl       # periodic in z
)
from .params import RWCylParams
from .patches import PatchSetCyl


def simulate_flux_cyl(params: RWCylParams,
                      rng: np.random.Generator,
                      concentration_check: bool = False,
                      sample_stride: int | None = None,
                      n_profile_times: int = 10,
                      n_r_bins: int = 40):
    """
    Steady-state RW in a cylindrical shell a<=r<=R, periodic in z.
    Absorbed particles at r=R are respawned near r=a (thin shell).

    Optional: concentration_profile=True -> record radial concentration profiles
    at n_profile_times evenly spaced times between 0 and T.

    Returns dict with usual outputs + (if concentration_check):
      - r_centers
      - profile_times
      - concentrations_time  (shape: [n_profile_times, n_r_bins])
    """
    N0 = int(params.N)
    D  = float(params.D)
    a  = float(params.a)
    R  = float(params.R)
    H  = float(params.H)
    dt = float(params.dt)
    T  = float(params.T)
    C0_final=float(params.C0)
    eps_respawn = params.eps_respawn
    t_relax = float(params.t_relax)



    # --- global concentration from total particles ---
    vol_shell = np.pi * H * (R**2 - a**2)
    c0_total  = (N0 / vol_shell) if vol_shell > 0 else 0.0

    ka = float(getattr(params, "ka", math.inf))  # binding coefficient
    kb = float(getattr(params, "kb", 0.0))  # release to outer
    kc = float(getattr(params, "kc", math.inf))  # release to inner




    step_sigma = np.sqrt(2.0 * D * dt) if (D > 0 and dt > 0) else 0.0
    n_steps = int(np.ceil(T / dt)) if dt > 0 else 0

    # --- relaxation (for measuring steady-state flux / c0) ---
    n_relax_steps = int(t_relax / dt) if dt > 0 else 0
    n_relax_steps = min(max(n_relax_steps, 0), max(n_steps - 1, 0))

    absorbed_total = 0
    absorbed_after_relax = 0
    measure_steps = 0


    # --- c0 estimation near inner cylinder (two-shell linear extrapolation) ---
    eps_c0 = float(getattr(params, "eps_c0", 0.0))
    if eps_c0 > 0 and R > a:
        eps_c0 = min(eps_c0, (R - a) / 3.0)

    if eps_c0 > 0:
        r0_minus, r0_plus = a, a + eps_c0
        r1_minus, r1_plus = a + 2*eps_c0, a + 3*eps_c0
        vol0 = np.pi * H * (r0_plus**2 - r0_minus**2)
        vol1 = np.pi * H * (r1_plus**2 - r1_minus**2)
    else:
        r0_minus = r0_plus = r1_minus = r1_plus = a
        vol0 = vol1 = 1.0

    c0_sum = c1_sum = 0.0
    c0_cnt = c1_cnt = 0

    def c0_extrapolate_so_far():
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

    # ---c_out estimation near outer cylinder (two-shell linear extrapolation) ---

    eps_out = float(getattr(params, "eps_out", 0.1))
    r_out_minus, r_out_plus = R - eps_out, R
    vol_out = np.pi * H * (r_out_plus ** 2 - r_out_minus ** 2)

    # ------------------------------------------------------------
    # Radial concentration snapshots at 10 times (optional)
    # ------------------------------------------------------------
    if concentration_check:
        r_edges = np.linspace(a, R, n_r_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_volumes = np.pi * H * (r_edges[1:]**2 - r_edges[:-1]**2)

        n_profile_times = int(max(1, n_profile_times))
        profile_steps = np.linspace(0, max(n_steps - 1, 0), n_profile_times, dtype=int)
        profile_times = profile_steps * dt

        concentrations_time = np.full((n_profile_times, n_r_bins), np.nan, dtype=float)
        prof_i = 0
    else:
        r_centers = None
        profile_times = None
        concentrations_time = None
        profile_steps = None
        prof_i = 0

    patches_enabled = (getattr(params, "n_patches", 0) > 0 and
                       getattr(params, "patch_radius", None) is not None)

    patches = PatchSetCyl(
        enabled=patches_enabled,
        n_patches=getattr(params, "n_patches", 0),
        patch_radius=getattr(params, "patch_radius", None),
        R=R,
        H=H,
        rng=rng, ka=ka,kb=kb,kc=kc,
    )

    # Initial positions
    X = rand_points_in_thin_cyl_volume(N0, a,R, H, R - a, rng, "inner")


    t = 0.0
    idx_captured = np.empty((0, 2), dtype=np.int64)  # [particle_idx, patch_idx]

    for k in range(n_steps):
        t += dt
        n_abs = 0  # ALWAYS define each step
        idx_bind = np.empty(0, dtype=int)

        # -------------------------
        # Masks: free vs captured
        # -------------------------
        # If you use NaN to mark captured, this is the most reliable "free" test
        free_mask = np.isfinite(X[:, 0])  # captured/out-of-bulk are NaN


        # -------------------------
        # 1) Diffusion step (free only)
        # -------------------------
        dX = np.zeros_like(X)
        dX[free_mask] = rng.normal(0.0, step_sigma, size=(free_mask.sum(), 3))

        # Trial endpoints
        X_next = X + dX

        # -------------------------
        # 2) Outer wall hits (free only)
        # -------------------------
        hit_mask = np.zeros(X.shape[0], dtype=bool)
        t_hit = np.zeros(X.shape[0], dtype=float)
        S = np.zeros_like(X)

        if np.any(free_mask):
            hit_sub, t_hit_sub, S_sub = segment_hits_cylinder(X[free_mask], dX[free_mask], R, snap=True)
            idx_free = np.flatnonzero(free_mask)

            hit_mask[idx_free] = hit_sub
            t_hit[idx_free] = t_hit_sub
            S[idx_free] = S_sub

        # -------------------------
        # 3) Binding + reflection at r=R
        # -------------------------
        absorbed_now_mask = np.zeros(X.shape[0], dtype=bool)

        # Estimate outer-shell concentration from FREE particles only
        X_free = X[free_mask]
        rho_free = np.sqrt(X_free[:, 0] ** 2 + X_free[:, 1] ** 2)
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

                # Build robust mapping: global particle idx -> patch idx (only for idx_on_patch set)
                pos_in_on = np.full(X.shape[0], -1, dtype=int)
                pos_in_on[idx_on_patch] = np.arange(idx_on_patch.size)

                # ---- ka gating (finite ka) ----
                if idx_on_patch.size and (ka != math.inf):


                    # 1. Define Intrinsic Rate (Kappa)
                    # If 'ka' is the target macroscopic rate for ONE patch:
                    patch_area = np.pi * (patches.patch_radius ** 2)
                    kappa = ka / patch_area

                    # 2. Calculate P_bind using Erban-Chapman approximation
                    # This scales with sqrt(dt) to compensate for re-collisions
                    p_scale = np.sqrt(np.pi * dt / D)
                    p_bind = kappa * p_scale

                    # Optional: A slightly more accurate formula for larger P (prevents P > 1)
                    p_bind = (kappa * p_scale) / (1 + (kappa * p_scale) / 2)

                    # 3. Apply
                    # Cap at 1.0 just in case
                    if p_bind > 1.0:
                        p_bind = 1.0

                    idx_ka = idx_on_patch[rng.random(idx_on_patch.size) < p_bind]

                else:
                    idx_ka = idx_on_patch

                # Ensure idx_ka is valid subset (debug safety)
                # assert np.all(pos_in_on[idx_ka] >= 0)

                if (kb > 0) or (kc != math.inf):
                    if idx_ka.size:
                        patch_idx_ka = patch_idx_on[pos_in_on[idx_ka]]

                        bind_pairs = patches.select_binders_with_occupancy(
                            idx_hit=idx_ka,
                            patch_idx_hit=patch_idx_ka,
                            t_hit=t_hit,  # global, ONLY if occupancy indexes global t_hit[p_idx]
                            dt=dt,
                            c_outer_shell=c_outer_shell,
                            rng=rng,
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
                        X_next[idx_ref] = reflect_after_outer_hit_specular_cyl(
                            S_hit=S[idx_ref], dX=dX[idx_ref], t_hit=t_hit[idx_ref], R=R
                        )

                else:
                    idx_bind = idx_ka
                    if idx_bind.size:
                        absorbed_now_mask[idx_bind] = True
                        X_next[idx_bind, :] = np.nan

                    idx_ref = np.setdiff1d(idx_hit, idx_bind, assume_unique=False)
                    if idx_ref.size:
                        X_next[idx_ref] = reflect_after_outer_hit_specular_cyl(
                            S_hit=S[idx_ref], dX=dX[idx_ref], t_hit=t_hit[idx_ref], R=R
                        )

        # -------------------------
        # 4) Inner reflection for FREE + NOT-just-bound
        # -------------------------
        free_next = np.isfinite(X_next[:, 0])
        alive_mask = free_next & (~absorbed_now_mask)  # absorbed_now are bound this step (NaN anyway)

        if np.any(alive_mask):
            X_next[alive_mask] = reflect_from_inner_cylinder(X_next[alive_mask], a)

        # -------------------------
        # 5) Release captured particles (respawn)
        # -------------------------

        if (kb > 0) or (kc != math.inf):
            idx_captured, relR_pairs, relA_pairs = patches.split_captured_by_release(
                idx_captured=idx_captured, dt=dt, rng=rng
            )

            idx_rel_R = relR_pairs[:, 0] if relR_pairs.size else np.empty(0, dtype=int)
            idx_rel_a = relA_pairs[:, 0] if relA_pairs.size else np.empty(0, dtype=int)

            if idx_rel_R.size:
                X_next[idx_rel_R] = rand_points_in_thin_cyl_volume(
                    len(idx_rel_R), a, R, H, eps_out, rng, "outer"
                )
            if idx_rel_a.size:
                X_next[idx_rel_a] = rand_points_in_thin_cyl_volume(
                    len(idx_rel_a), a, R, H, eps_respawn, rng, "inner"
                )

            n_abs = idx_rel_a.size



        else:
            n_abs = int(absorbed_now_mask.sum())

            if n_abs:
                X_next[absorbed_now_mask] = rand_points_in_thin_cyl_volume(
                    n_abs, a, R, H, eps_respawn, rng, "inner"
                )

        # -------------------------
        # 6) Periodic z for FREE particles only
        # -------------------------
        free_next = np.isfinite(X_next[:, 0])
        if np.any(free_next):
            X_next[free_next] = apply_periodic_caps(X_next[free_next], H)

        X = X_next

        # -------------------------
        # Concentration profile (FREE only)
        # -------------------------
        if concentration_check and (prof_i < len(profile_steps)) and (k == profile_steps[prof_i]):
            free_now = np.isfinite(X[:, 0])
            r = np.sqrt(X[free_now, 0] ** 2 + X[free_now, 1] ** 2)
            counts, _ = np.histogram(r, bins=r_edges)
            concentrations_time[prof_i, :] = counts / shell_volumes
            prof_i += 1

        # -------------------------
        # Post-relax measurements (FREE only)
        # -------------------------
        if k >= n_relax_steps:
            absorbed_after_relax += n_abs
            measure_steps += 1

            if eps_c0 > 0:
                free_now = np.isfinite(X[:, 0])
                r = np.sqrt(X[free_now, 0] ** 2 + X[free_now, 1] ** 2)
                mask0 = (r >= r0_minus) & (r < r0_plus)
                mask1 = (r >= r1_minus) & (r < r1_plus)
                c0_sum += ((mask0.sum()) / vol0)
                c1_sum += (mask1.sum() / vol1)
                c0_cnt += 1
                c1_cnt += 1

    # After loop
    c0_measured = max(float(c0_extrapolate_so_far()), 1e-300)

    if measure_steps > 0:
        T_meas = measure_steps * dt
        rate_est = absorbed_after_relax / T_meas
    else:
        rate_est = 0.0

    print(f"This is the measurd {c0_measured} this is the caluclated {C0_final} for n patches {params.n_patches} and for ka,kb,kc {ka},{kb},{kc}")

    rate_normalized = rate_est / c0_measured
    out = {
        "c0_total": float(c0_total),
        "c0_measured": float(c0_measured),
        "rate_est": float(rate_est),
        "rate_normalized": float(rate_normalized),
    }

    if concentration_check:
        out.update({
            "r_centers": r_centers,
            "profile_times": profile_times,
            "concentrations_time": concentrations_time
        })

    return out
