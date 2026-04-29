"""Transient random walk simulation: tracks particle survival over time."""
import numpy as np
from ..params import SimulationParams
from .geometry import rand_points_in_shell, segment_hits_cylinder
from .boundaries import reflect_inner, reflect_outer, wrap_periodic_z
from .patches import PatchSet


class TransientSimulation:
    """Transient absorption in a 3D cylindrical shell.

    Particles are placed initially and absorbed over time (no respawning).
    Records survival fraction N(t)/N0 for comparison with analytical solutions.
    """

    def __init__(self, params: SimulationParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng

    def run(self, record_stride=None):
        """Run transient simulation.

        Returns dict with t, N_alive, absorbed, frac_alive, t_end.
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

        X = rand_points_in_shell(N0, a, R, H, R - a, rng, "inner")

        step_sigma = np.sqrt(2.0 * D * dt)
        n_steps = int(np.ceil(T / dt)) if dt > 0 else 0

        if record_stride is None or record_stride < 1:
            record_stride = max(1, n_steps // 500)

        t_list = [0.0]
        N_list = [X.shape[0]]
        abs_list = [0]
        absorbed_total = 0
        t = 0.0

        for k in range(n_steps):
            if X.shape[0] == 0:
                break
            t += dt

            dX = rng.normal(0.0, step_sigma, size=(X.shape[0], 3))
            X_trial = X + dX

            hit_mask, t_hit, S = segment_hits_cylinder(X, dX, R, snap=True)
            X_next = X_trial.copy()

            if not patches_enabled:
                absorbed_total += int(hit_mask.sum())
                X_surv = X_next[~hit_mask]
            else:
                if np.any(hit_mask):
                    idx_hit = np.where(hit_mask)[0]
                    S[idx_hit] = wrap_periodic_z(S[idx_hit], H)

                    on_patch = patches.is_absorbing(S[idx_hit])
                    idx_abs = idx_hit[on_patch]
                    idx_ref = idx_hit[~on_patch]

                    absorbed_total += int(idx_abs.size)

                    if idx_ref.size > 0:
                        X_next[idx_ref] = reflect_outer(S[idx_ref], dX[idx_ref], t_hit[idx_ref], R)

                    keep = np.ones(X_next.shape[0], dtype=bool)
                    keep[idx_abs] = False
                    X_surv = X_next[keep]
                else:
                    X_surv = X_next

            X_surv = reflect_inner(X_surv, a)
            X_surv = wrap_periodic_z(X_surv, H)
            X = X_surv

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
