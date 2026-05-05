"""Transient random walk simulation: tracks particle survival over time."""
import numpy as np
from classes.params import SimulationParams
from .geometry import rand_points_in_shell, segment_hits_cylinder
from .boundaries import reflect_inner, reflect_outer, wrap_periodic_z
from .patches import PatchSet


class TransientSimulation:
    """Transient absorption in the cylindrical annulus ``a <= r <= R``.

    Simulation concept
    ------------------
    At time ``t = 0``, ``N`` particles are placed uniformly inside the annulus.
    Particles diffuse by Brownian motion and are *permanently removed* when they
    are absorbed.  There is no respawning.  The survival fraction ``N(t)/N₀``
    decays from 1 toward 0 as particles are absorbed one by one.

    This is the transient (time-dependent) counterpart to the steady-state
    simulation.  The result is compared with the analytical eigenfunction
    expansion ``N(t)/N₀ = Σ Aₙ · Iₙ · exp(−D λₙ² t)`` computed by
    ``TransientAnalytical``.

    Absorption rules
    ----------------
    * **No patches** (``n_patches = 0``): the outer wall is *fully absorbing* —
      any particle that hits ``r = R`` is removed.  This is the Dirichlet
      boundary condition.
    * **With patches** (``n_patches > 0``): only hits on a receptor patch are
      absorbed; hits on the open wall are reflected specularly.  This
      approximates a Robin (mixed) boundary condition with effective parameter
      ``k_eff = 2 * n_patches * s / (pi * H)``.

    Note: finite kinetics (``ka``, ``kb``, ``kc``) are technically accepted by
    the ``PatchSet`` constructor but the transient simulation uses
    ``patches.is_absorbing()`` which treats every patch contact as an instant
    absorption.  The kinetics parameters are therefore ignored in practice.
    """

    def __init__(self, params: SimulationParams, rng: np.random.Generator):
        """
        Parameters
        ----------
        params : SimulationParams
            All physical and numerical parameters.  Key fields:
            ``N``, ``D``, ``a``, ``R``, ``H``, ``dt``, ``T``,
            ``n_patches``, ``patch_radius``.
        rng : numpy.random.Generator
            Seeded random number generator for reproducibility.
        """
        self.p = params
        self.rng = rng

    def run(self, record_stride: int | None = None) -> dict:
        """Execute the transient simulation and return the survival curve.

        Parameters
        ----------
        record_stride : int or None
            Record the particle count every ``record_stride`` steps.  If None,
            defaults to ``max(1, n_steps // 500)`` so that approximately 500
            data points are collected regardless of the total number of steps.

        Returns
        -------
        dict with keys:

        ``t`` : (T,) float array
            Simulation times at which the particle count was recorded (seconds).
            Always starts with ``t=0``.

        ``N_alive`` : (T,) int array
            Number of surviving (not yet absorbed) particles at each recorded
            time.

        ``absorbed`` : (T,) int array
            Cumulative number of absorbed particles up to each recorded time.

        ``frac_alive`` : (T,) float array
            Survival fraction ``N_alive / N₀``.  Starts at 1.0 and decreases
            monotonically toward 0.

        ``t_end`` : float
            The actual final time reached by the simulation (may be slightly
            less than ``T`` if all particles are absorbed early).
        """
        p = self.p
        rng = self.rng

        N0 = int(p.N)
        D, a, R, H = float(p.D), float(p.a), float(p.R), float(p.H)
        dt, T = float(p.dt), float(p.T)

        patches_enabled = (p.n_patches > 0 and p.patch_radius is not None)
        patches = PatchSet(
            enabled=patches_enabled,
            n_patches=p.n_patches,
            patch_radius=p.patch_radius,
            R=R, H=H, rng=rng,
            ka=p.ka, kb=p.kb, kc=p.kc,
        )

        # Place all particles uniformly across the entire annular volume
        X = rand_points_in_shell(N0, a, R, H, R - a, rng, "inner")

        step_sigma = np.sqrt(2.0 * D * dt)
        n_steps = int(np.ceil(T / dt)) if dt > 0 else 0

        # Default: ~500 recorded points over the full run
        if record_stride is None or record_stride < 1:
            record_stride = max(1, n_steps // 500)

        # Record initial state at t=0
        t_list = [0.0]
        N_list = [X.shape[0]]
        abs_list = [0]
        absorbed_total = 0
        t = 0.0

        for k in range(n_steps):
            if X.shape[0] == 0:
                # All particles absorbed — simulation ends early
                break
            t += dt

            # Diffuse all surviving particles
            dX = rng.normal(0.0, step_sigma, size=(X.shape[0], 3))
            X_trial = X + dX
            hit_mask, t_hit, S = segment_hits_cylinder(X, dX, R, snap=True)
            X_next = X_trial.copy()

            if not patches_enabled:
                # Fully absorbing wall: every hit removes the particle
                absorbed_total += int(hit_mask.sum())
                X_surv = X_next[~hit_mask]
            else:
                if np.any(hit_mask):
                    idx_hit = np.where(hit_mask)[0]
                    # Wrap hit points z-periodically before patch lookup
                    S[idx_hit] = wrap_periodic_z(S[idx_hit], H)

                    # is_absorbing returns True if the hit point lies on a patch
                    on_patch = patches.is_absorbing(S[idx_hit])
                    idx_abs = idx_hit[on_patch]     # absorbed: on a patch
                    idx_ref = idx_hit[~on_patch]    # reflected: open wall

                    absorbed_total += int(idx_abs.size)

                    # Reflect particles that hit the open wall specularly
                    if idx_ref.size:
                        X_next[idx_ref] = reflect_outer(S[idx_ref], dX[idx_ref], t_hit[idx_ref], R)

                    # Remove absorbed particles from the array
                    keep = np.ones(X_next.shape[0], dtype=bool)
                    keep[idx_abs] = False
                    X_surv = X_next[keep]
                else:
                    X_surv = X_next

            # Reflect any particles that penetrated the inner wall
            X_surv = reflect_inner(X_surv, a)
            # Apply periodic z boundary
            X_surv = wrap_periodic_z(X_surv, H)
            X = X_surv

            # Record particle count at the specified stride
            if (k + 1) % record_stride == 0:
                t_list.append(t)
                N_list.append(X.shape[0])
                abs_list.append(absorbed_total)

        t_arr = np.asarray(t_list, float)
        N_arr = np.asarray(N_list, int)
        abs_arr = np.asarray(abs_list, int)

        return {
            "t": t_arr,
            "N_alive": N_arr,
            "absorbed": abs_arr,
            "frac_alive": N_arr / float(N0),
            "t_end": float(t_arr[-1]),
        }
