from rw.geometry import (
    segment_hits_cylinder,rand_points_in_thin_cyl_volume        # intersection with r = R (outer cylinder)
)
from .boundaries import (reflect_from_inner_cylinder,
    apply_periodic_caps, reflect_after_outer_hit_specular_cyl)
from rw.patches import PatchSetCyl
import numpy as np
from scipy.special import j0, y0, j1, y1
from scipy.optimize import brentq
from scipy.integrate import quad


def annulus_eigenvalues_NeumannRobin(a, R, k, n_modes=50):
    """
    Find eigenvalues λ_n > 0 for:
      φ'(a)=0                  (Neumann at a)
      φ'(R) + (k/R) φ(R) = 0   (Robin at R; effective patch absorption)

    with φ(r)=J0(λr) + β Y0(λr), β=-J1(λa)/Y1(λa).

    Root equation (β eliminated):
      f(λ)= -λ[ J1(λR)Y1(λa) - Y1(λR)J1(λa) ]
            + (k/R)[ J0(λR)Y1(λa) - Y0(λR)J1(λa) ] = 0
    """
    def f(lam):
        term1 = -lam * (j1(lam*R)*y1(lam*a) - y1(lam*R)*j1(lam*a))
        term2 = (k / R) * (j0(lam*R)*y1(lam*a) - y0(lam*R)*j1(lam*a))
        return term1 + term2

    vals = []
    lam_max = (n_modes + 5) * np.pi / max(R - a, 1e-12)

    grid = np.linspace(1e-8, lam_max, 20000)
    fg = f(grid)

    # avoid numerical trouble where Y1(lam*a) is near 0
    bad = np.abs(y1(grid*a)) < 1e-10
    fg[bad] = np.nan

    for i in range(len(grid) - 1):
        if np.isnan(fg[i]) or np.isnan(fg[i+1]):
            continue
        if fg[i] == 0.0:
            vals.append(grid[i])
        elif fg[i] * fg[i+1] < 0:
            lam = brentq(f, grid[i], grid[i+1], maxiter=200)
            vals.append(lam)
        if len(vals) >= n_modes:
            break

    return np.array(vals, float)
def annulus_eigenvalues_NeumannDirichlet(a, R, n_modes=50):
    """
    Find eigenvalues λ_n > 0 for:
      φ'(a)=0  (Neumann at a)
      φ(R)=0   (Dirichlet at R)
    with φ(r)=J0(λr) + β Y0(λr), β=-J1(λa)/Y1(λa).

    Root equation:
      f(λ) = J0(λR) Y1(λa) - Y0(λR) J1(λa) = 0
    """
    def f(lam):
        return j0(lam*R)*y1(lam*a) - y0(lam*R)*j1(lam*a)

    vals = []
    # heuristic scan range; increase if you need more modes
    lam_max = (n_modes + 5) * np.pi / max(R - a, 1e-12)

    # scan for sign changes
    grid = np.linspace(1e-8, lam_max, 20000)
    fg = f(grid)

    for i in range(len(grid)-1):
        if np.isnan(fg[i]) or np.isnan(fg[i+1]):
            continue
        if fg[i] == 0.0:
            vals.append(grid[i])
        elif fg[i] * fg[i+1] < 0:
            lam = brentq(f, grid[i], grid[i+1], maxiter=200)
            vals.append(lam)
        if len(vals) >= n_modes:
            break

    return np.array(vals, float)

def annulus_mode_beta(lam, a):
    # from Neumann at a: J1(λa) + β Y1(λa) = 0
    return -j1(lam*a) / y1(lam*a)

def annulus_phi(lam, a, r):
    beta = annulus_mode_beta(lam, a)
    return j0(lam*r) + beta*y0(lam*r)

def annulus_survival_fraction(t, a, R, D, n_modes=50, p0="thin_shell", eps=0.0,
                             bc="dirichlet", k=None):
    """
    Compute N(t)/N0 using eigen-expansion.

    bc:
      - "dirichlet": φ(R)=0  (full absorbing)
      - "robin":     φ'(R) + (k/R) φ(R)=0  (effective patch absorption)

    For bc="robin" you must pass k (dimensionless).
    """
    if bc == "dirichlet":
        lams = annulus_eigenvalues_NeumannDirichlet(a, R, n_modes=n_modes)
    elif bc == "robin":
        if k is None:
            raise ValueError("For bc='robin', you must supply k.")
        lams = annulus_eigenvalues_NeumannRobin(a, R, k, n_modes=n_modes)
    else:
        raise ValueError("bc must be 'dirichlet' or 'robin'")

    # --- your original p0 logic unchanged ---
    if p0 == "uniform_volume":
        r1, r2 = a, R
    elif p0 == "thin_shell":
        r1, r2 = a, min(a + eps, R)
        if r2 <= r1:
            raise ValueError("For p0='thin_shell', need eps>0 so that a < a+eps <= R.")
    else:
        raise ValueError("p0 must be 'uniform_volume' or 'thin_shell'")

    Z = (r2**2 - r1**2) / 2.0
    def p_init(r):
        if r < r1 or r > r2:
            return 0.0
        return r / Z

    A = np.zeros_like(lams)

    for i, lam in enumerate(lams):
        def phi(r): return annulus_phi(lam, a, r)

        norm = quad(lambda r: (phi(r)**2) * r, a, R, limit=200)[0]
        num  = quad(lambda r: p_init(r) * phi(r) * r, a, R, limit=200)[0]
        A[i] = num / norm

    # integral of φ over dr (your original approach)
    I = np.zeros_like(lams)
    for i, lam in enumerate(lams):
        def phi(r): return annulus_phi(lam, a, r)
        I[i] = quad(lambda r: phi(r), a, R, limit=200)[0]

    t = np.asarray(t, float)
    expf = np.exp(-D * (lams[:, None]**2) * t[None, :])
    surv = (A[:, None] * I[:, None] * expf).sum(axis=0)

    return np.clip(surv, 0.0, 1.0)

def simulate_transient_cyl_absorb(params, rng: np.random.Generator,
                                           record_stride: int | None = None):
    """
    Transient absorption in a 3D cylindrical shell a <= r <= R, |z| <= H/2.

    BCs:
      - Inner cylinder r=a: reflecting (handled by reflect_from_inner_cylinder)
      - Outer cylinder r=R:
          * patches disabled: fully absorbing (remove any hit)
          * patches enabled: absorb if hit-point is on a patch, else specularly reflect
      - z-caps: periodic (apply_periodic_caps)

    Returns dict:
      - t, N_alive, absorbed, frac_alive, t_end
    """
    N0 = int(params.N)
    D  = float(params.D)
    a  = float(params.a)
    R  = float(params.R)
    H  = float(params.H)
    dt = float(params.dt)
    T  = float(params.T)

    patches_enabled = (getattr(params, "n_patches", 0) > 0 and
                       getattr(params, "patch_radius", None) is not None)

    patches = PatchSetCyl(
        enabled=patches_enabled,
        n_patches=getattr(params, "n_patches", 0),
        patch_radius=getattr(params, "patch_radius", None),
        R=R,
        H=H,
        rng=rng
    )

    # Initial positions
    X = rand_points_in_thin_cyl_volume(N0, a,R, H, R - a, rng,"inner")

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


        # ---- Outer boundary: segment hits on r=R ----
        hit_mask, t_hit, S = segment_hits_cylinder(X, dX, R, snap=True)



        X_next = X_trial.copy()


        if not patches_enabled:
            absorbed_total += int(hit_mask.sum())
            X_surv = X_next[~hit_mask]

        else:
            if np.any(hit_mask):
                idx_hit = np.where(hit_mask)[0]
                S[idx_hit] = apply_periodic_caps(S[idx_hit], H)

                on_patch = patches.is_absorbing(S[idx_hit])

                idx_abs = idx_hit[on_patch]
                idx_ref = idx_hit[~on_patch]

                absorbed_total += int(idx_abs.size)

                if idx_ref.size > 0:
                    X_next[idx_ref] = reflect_after_outer_hit_specular_cyl(
                        S_hit=S[idx_ref],
                        dX=dX[idx_ref],
                        t_hit=t_hit[idx_ref],
                        R=R
                    )

                keep = np.ones(X_next.shape[0], dtype=bool)
                keep[idx_abs] = False
                X_surv = X_next[keep]
            else:
                X_surv = X_next

        # ---- Inner reflection + periodic caps on survivors ----
        X_surv = reflect_from_inner_cylinder(X_surv, a)
        X_surv = apply_periodic_caps(X_surv, H)

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



