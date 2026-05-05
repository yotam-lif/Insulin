"""Steady-state random walk simulation in a cylindrical annulus."""
import math
import numpy as np
from dataclasses import dataclass
from classes.params import SimulationParams
from .geometry import rand_points_in_shell, segment_hits_cylinder
from .boundaries import reflect_inner, reflect_outer, wrap_periodic_z
from .patches import PatchSet


@dataclass
class SteadyStateResult:
    """Output of a completed steady-state simulation run.

    Fields
    ------
    c0_total : float
        Global bulk concentration computed from the total particle count and
        shell volume: ``C0 = N / (pi * H * (R² - a²))``.  This is the
        "nominal" concentration — what you would get if particles were spread
        uniformly.  It does not account for the depletion gradient near the
        receptors.

    c0_measured : float
        Concentration extrapolated to ``r = a`` by a linear fit through two
        thin shells near the inner wall.  This is the effective "source"
        concentration seen by the receptors in the steady-state diffusion
        gradient.  Used as ``C0`` when computing the analytical ``mu(C0)``
        for comparison.

    rate : float
        Mean absorption rate averaged over the measurement window
        (particles/s).  Computed as ``(absorptions after relaxation) / T_meas``
        where ``T_meas = (n_steps - n_relax_steps) * dt``.

    rate_normalized : float
        Rate constant ``k = rate / c0_measured`` (µm³/s).  Dividing by the
        measured concentration makes this comparable to the analytical
        ``k_max = 2*pi*H*D / ln(R/a)`` regardless of particle count.

    r_centers : ndarray or None
        Bin centres (µm) for the radial concentration profile.  Only set when
        ``run(concentration_profile=True)`` is called.

    profile_times : ndarray or None
        Time points (s) at which concentration snapshots were recorded.

    concentrations : ndarray or None
        Shape ``(n_profile_times, n_r_bins)`` array of instantaneous
        concentration values (particles/µm³) in each radial bin at each
        snapshot time.
    """

    c0_total: float         # global concentration from total particles (particles/µm³)
    c0_measured: float      # extrapolated concentration near r=a
    rate: float             # absorption rate (particles/s)
    rate_normalized: float  # rate / c0_measured (µm³/s) -- analogous to k
    r_centers: np.ndarray | None = None
    profile_times: np.ndarray | None = None
    concentrations: np.ndarray | None = None


class SteadyStateSimulation:
    """Steady-state random walk in the cylindrical annulus ``a <= r <= R``.

    Simulation concept
    ------------------
    Particles diffuse by Brownian motion.  Whenever a particle is absorbed
    (bound and internalized, or — if patches are absent — simply hits the outer
    wall), it is immediately *respawned* near the inner wall ``r ≈ a``.  This
    maintains a constant total particle count ``N``, which drives a persistent
    diffusion current from the inner wall toward the outer receptors — the
    steady state.

    After an initial relaxation phase (``t < t_relax``) the concentration
    profile settles into the logarithmic steady-state form
    ``C(r) ∝ ln(r/a)``.  Only absorptions after ``t_relax`` are counted toward
    the flux estimate.

    Per-timestep algorithm
    ----------------------
    1. **Diffuse** free particles: each coordinate gets a Gaussian displacement
       with standard deviation ``sigma = sqrt(2*D*dt)``.
    2. **Detect outer-wall hits** via ray-cylinder intersection
       (``segment_hits_cylinder``).
    3. **Bind** particles that hit a receptor patch:
       - If ``ka < inf``: apply Erban-Chapman probability
         ``p_bind = (κ√(πdt/D)) / (1 + κ√(πdt/D)/2)`` where ``κ = ka/(πs²)``.
       - Call ``PatchSet.select_binders`` to enforce patch capacity and
         record captured particles.
       - Captured particles get ``X = nan`` (removed from the free pool).
    4. **Reflect** non-binding outer-wall hits specularly (``reflect_outer``).
    5. **Reflect** particles that penetrated the inner wall (``reflect_inner``).
    6. **Release** bound particles stochastically (``PatchSet.release_captured``):
       - ``kb``: particle returns to outer bulk near ``r ≈ R``.
       - ``kc``: particle is internalized — transported to inner bulk near ``r ≈ a``.
    7. **Respawn** internalized particles near ``r = a``.
    8. **Wrap** z periodically (``wrap_periodic_z``).
    """

    def __init__(self, params: SimulationParams, rng: np.random.Generator):
        """
        Parameters
        ----------
        params : SimulationParams
            All physical and numerical parameters for this run.
        rng : numpy.random.Generator
            Seeded random number generator.  Pass a fresh generator per worker
            to ensure reproducible parallel runs.
        """
        self.p = params
        self.rng = rng

    def run(self, concentration_profile=False,
            n_profile_times=10, n_r_bins=40) -> SteadyStateResult:
        """Execute the simulation and return flux and concentration estimates.

        Parameters
        ----------
        concentration_profile : bool
            If True, record snapshots of the radial concentration profile at
            ``n_profile_times`` evenly spaced times throughout the run.
            Useful for visualising approach to steady state.
        n_profile_times : int
            Number of profile snapshots to record (ignored if
            ``concentration_profile=False``).
        n_r_bins : int
            Number of radial bins for the concentration histogram
            (ignored if ``concentration_profile=False``).

        Returns
        -------
        SteadyStateResult
            See class docstring for field descriptions.
        """
        p = self.p
        rng = self.rng

        N0 = int(p.N)
        D, a, R, H = float(p.D), float(p.a), float(p.R), float(p.H)
        dt, T = float(p.dt), float(p.T)
        eps_respawn = p.eps_respawn
        t_relax = float(p.t_relax)
        ka, kb, kc = float(p.ka), float(p.kb), float(p.kc)

        # Nominal concentration: assumes particles spread uniformly in shell
        vol_shell = np.pi * H * (R**2 - a**2)
        c0_total = (N0 / vol_shell) if vol_shell > 0 else 0.0

        step_sigma = np.sqrt(2.0 * D * dt) if (D > 0 and dt > 0) else 0.0
        n_steps = int(np.ceil(T / dt)) if dt > 0 else 0
        n_relax_steps = min(max(int(t_relax / dt), 0), max(n_steps - 1, 0))

        absorbed_after_relax = 0
        measure_steps = 0

        # ------------------------------------------------------------------
        # C0 extrapolation setup
        # ------------------------------------------------------------------
        # We estimate the local concentration at r=a by measuring <C> in two
        # thin shells and extrapolating linearly to r=a.
        # Shell 0: [a,   a+eps_c0]  (innermost)
        # Shell 1: [a+2*eps_c0, a+3*eps_c0]  (one step further out)
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
            """Linear extrapolation of C(r) to r=a from the two inner shells."""
            if eps_c0 <= 0 or c0_cnt == 0 or c1_cnt == 0:
                return c0_total
            C0_ = c0_sum / c0_cnt     # mean concentration in inner shell
            C1_ = c1_sum / c1_cnt     # mean concentration in outer shell
            r0c = 0.5 * (r0_minus + r0_plus)   # centre radius of inner shell
            r1c = 0.5 * (r1_minus + r1_plus)   # centre radius of outer shell
            if r1c == r0c:
                return float(C0_)
            # Slope dC/dr between the two shell centres
            m = (C1_ - C0_) / (r1c - r0c)
            # Extrapolate back to r=a
            return float(m * (a - r0c) + C0_)

        # Thin shell near outer wall for tracking local concentration
        eps_out = float(p.eps_out)
        r_out_minus = R - eps_out
        vol_out = np.pi * H * (R**2 - r_out_minus**2)

        # ------------------------------------------------------------------
        # Optional concentration profile setup
        # ------------------------------------------------------------------
        if concentration_profile:
            r_edges = np.linspace(a, R, n_r_bins + 1)
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            shell_volumes = np.pi * H * (r_edges[1:]**2 - r_edges[:-1]**2)
            profile_steps = np.linspace(0, max(n_steps - 1, 0), n_profile_times, dtype=int)
            profile_times = profile_steps * dt
            concentrations_time = np.full((n_profile_times, n_r_bins), np.nan)
            prof_i = 0
        else:
            r_centers = profile_times = concentrations_time = profile_steps = None
            prof_i = 0

        # ------------------------------------------------------------------
        # Receptor patches
        # ------------------------------------------------------------------
        patches_enabled = (p.n_patches > 0 and p.patch_radius is not None)
        patches = PatchSet(
            enabled=patches_enabled,
            n_patches=p.n_patches,
            patch_radius=p.patch_radius,
            R=R, H=H, rng=rng,
            ka=ka, kb=kb, kc=kc,
        )

        # Place all N particles uniformly across the full shell to start
        X = rand_points_in_shell(N0, a, R, H, R - a, rng, "inner")
        # idx_captured: (K, 2) array of [particle_index, patch_index] for bound particles
        idx_captured = np.empty((0, 2), dtype=np.int64)

        # ------------------------------------------------------------------
        # Main time loop
        # ------------------------------------------------------------------
        for k in range(n_steps):
            n_abs = 0
            idx_bind = np.empty(0, dtype=int)
            # Free particles have finite coordinates; bound/internalized have NaN
            free_mask = np.isfinite(X[:, 0])

            # Step 1: Diffuse free particles
            dX = np.zeros_like(X)
            dX[free_mask] = rng.normal(0.0, step_sigma, size=(free_mask.sum(), 3))
            X_next = X + dX

            # Step 2: Detect which free particles crossed the outer wall r=R
            hit_mask = np.zeros(X.shape[0], dtype=bool)
            t_hit = np.zeros(X.shape[0], dtype=float)
            S = np.zeros_like(X)

            if np.any(free_mask):
                hit_sub, t_hit_sub, S_sub = segment_hits_cylinder(X[free_mask], dX[free_mask], R, snap=True)
                idx_free = np.flatnonzero(free_mask)
                hit_mask[idx_free] = hit_sub
                t_hit[idx_free] = t_hit_sub
                S[idx_free] = S_sub

            # Step 3: Binding and reflection at r=R
            absorbed_now_mask = np.zeros(X.shape[0], dtype=bool)

            # Local concentration near the outer wall (used by PatchSet.select_binders
            # to track how many free particles are near the receptors)
            X_free = X[free_mask]
            rho_free = np.sqrt(X_free[:, 0]**2 + X_free[:, 1]**2)
            c_outer_shell = float(((rho_free >= r_out_minus)).sum()) / float(vol_out)

            if not patches_enabled:
                # No patches: fully absorbing wall — every outer-wall hit is absorbed
                absorbed_now_mask = hit_mask.copy()
            else:
                if np.any(hit_mask):
                    idx_hit_arr = np.where(hit_mask)[0]
                    # Map each hitting particle to its patch (or -1 if it missed all patches)
                    patch_idx_hit = patches.patch_index_of_hits(S[idx_hit_arr])

                    on = (patch_idx_hit >= 0)
                    idx_on_patch = idx_hit_arr[on]
                    patch_idx_on = patch_idx_hit[on]

                    # Lookup table: given a global particle index, what position is it in
                    # the "on-patch hitters" list?  Needed to look up patch indices later.
                    pos_in_on = np.full(X.shape[0], -1, dtype=int)
                    pos_in_on[idx_on_patch] = np.arange(idx_on_patch.size)

                    # Erban-Chapman finite-ka gating:
                    # Even though the particle touched the patch, it only binds with
                    # probability p_bind that accounts for finite association kinetics.
                    if idx_on_patch.size and (ka != math.inf):
                        kappa = ka / (np.pi * patches.radius**2)
                        p_scale = np.sqrt(np.pi * dt / D)
                        p_bind = (kappa * p_scale) / (1.0 + (kappa * p_scale) / 2.0)
                        p_bind = min(p_bind, 1.0)
                        idx_ka = idx_on_patch[rng.random(idx_on_patch.size) < p_bind]
                    else:
                        # ka=inf: every patch contact leads to binding
                        idx_ka = idx_on_patch

                    if (kb > 0) or (kc != math.inf):
                        # Finite kinetics: particles can be captured temporarily
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
                            X_next[idx_bind] = np.nan     # park them as "captured"
                            idx_captured = np.vstack([idx_captured, bind_pairs])

                        # Reflect any outer-wall hits that did not lead to binding
                        idx_ref = np.setdiff1d(idx_hit_arr, idx_bind)
                        if idx_ref.size:
                            X_next[idx_ref] = reflect_outer(S[idx_ref], dX[idx_ref], t_hit[idx_ref], R)
                    else:
                        # Perfectly absorbing patches (kc=inf, kb=0):
                        # immediately treat bound particles as absorbed
                        idx_bind = idx_ka
                        if idx_bind.size:
                            absorbed_now_mask[idx_bind] = True
                            X_next[idx_bind] = np.nan

                        idx_ref = np.setdiff1d(idx_hit_arr, idx_bind)
                        if idx_ref.size:
                            X_next[idx_ref] = reflect_outer(S[idx_ref], dX[idx_ref], t_hit[idx_ref], R)

            # Step 4: Reflect inner wall penetrations
            alive_mask = np.isfinite(X_next[:, 0]) & (~absorbed_now_mask)
            if np.any(alive_mask):
                X_next[alive_mask] = reflect_inner(X_next[alive_mask], a)

            # Step 5: Release bound particles stochastically
            if (kb > 0) or (kc != math.inf):
                # release_captured returns:
                #   idx_captured : updated list (bindings that did NOT release this step)
                #   relR_pairs   : particles released to outer bulk (dissociation, kb)
                #   relA_pairs   : particles internalized (kc), to be respawned near r=a
                idx_captured, relR_pairs, relA_pairs = patches.release_captured(idx_captured, dt, rng)
                idx_rel_R = relR_pairs[:, 0] if relR_pairs.size else np.empty(0, dtype=int)
                idx_rel_a = relA_pairs[:, 0] if relA_pairs.size else np.empty(0, dtype=int)

                if idx_rel_R.size:
                    # Dissociated: return to outer bulk near r=R
                    X_next[idx_rel_R] = rand_points_in_shell(len(idx_rel_R), a, R, H, eps_out, rng, "outer")
                if idx_rel_a.size:
                    # Internalized: respawn near the inner wall r=a
                    X_next[idx_rel_a] = rand_points_in_shell(len(idx_rel_a), a, R, H, eps_respawn, rng, "inner")
                n_abs = idx_rel_a.size   # only internalized particles count as absorbed
            else:
                # Perfectly absorbing: all hits are instant absorptions
                n_abs = int(absorbed_now_mask.sum())
                if n_abs:
                    X_next[absorbed_now_mask] = rand_points_in_shell(n_abs, a, R, H, eps_respawn, rng, "inner")

            # Step 6: Wrap z periodically
            free_next = np.isfinite(X_next[:, 0])
            if np.any(free_next):
                X_next[free_next] = wrap_periodic_z(X_next[free_next], H)

            X = X_next

            # Optionally record a concentration profile snapshot
            if concentration_profile and profile_steps is not None:
                if prof_i < len(profile_steps) and k == profile_steps[prof_i]:
                    free_now = np.isfinite(X[:, 0])
                    r = np.sqrt(X[free_now, 0]**2 + X[free_now, 1]**2)
                    counts, _ = np.histogram(r, bins=r_edges)
                    concentrations_time[prof_i] = counts / shell_volumes
                    prof_i += 1

            # After relaxation: accumulate absorption count and C0 measurements
            if k >= n_relax_steps:
                absorbed_after_relax += n_abs
                measure_steps += 1

                if eps_c0 > 0:
                    free_now = np.isfinite(X[:, 0])
                    r = np.sqrt(X[free_now, 0]**2 + X[free_now, 1]**2)
                    # Count particles in each thin measurement shell
                    c0_sum += (r >= r0_minus).sum() / vol0
                    c1_sum += ((r >= r1_minus) & (r < r1_plus)).sum() / vol1
                    c0_cnt += 1
                    c1_cnt += 1

        c0_measured = max(float(c0_extrapolate()), 1e-300)
        T_meas = measure_steps * dt if measure_steps > 0 else 0.0
        rate_est = absorbed_after_relax / T_meas if T_meas > 0 else 0.0

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
