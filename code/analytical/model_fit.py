"""Empirical curve fitting for J/J_max vs receptor count.

Tries multiple functional forms and reports the best fit by R².
Useful for understanding the scaling of uptake efficiency with receptor density
and for checking whether the simulation follows the Berg-Purcell prediction.

The natural variable is ``x = n_patches * patch_radius`` — the product of
receptor count and patch radius, which is the "total receptor aperture".  The
Berg-Purcell formula predicts ``J/J_max = x / (x + b)`` for some constant
``b`` determined by the geometry and diffusion resistance.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ------------------------------------------------------------------
# Model functions
# All take (x, ...) where x = n_patches * r_patch
# ------------------------------------------------------------------

def f_ln(x, a, b):
    """a * ln(x) + b  — logarithmic growth."""
    return a * np.log(x) + b

def f_inv_ln(x, a, b):
    """a / (ln(x) + b)  — inverse logarithm; expected for diffusion-limited capture."""
    return a / (np.log(x) + b)

def f_ln_invx(x, a, b):
    """a * ln(1/x) + b  — logarithm of the inverse."""
    return a * np.log(1.0 / x) + b

def f_inv_ln_invx(x, a, b):
    """a / (ln(1/x) + b)."""
    return a / (np.log(1.0 / x) + b)

def f_one_over_1_plus_ln(x, a, b):
    """a / (1 + b*ln(x))  — logarithmic saturation."""
    return a / (1.0 + b * np.log(x))

def f_one_over_1_plus_ln_invx(x, a, b):
    """a / (1 + b*ln(1/x))."""
    return a / (1.0 + b * np.log(1.0 / x))

def f_a_over_b_plus_x(x, a, b):
    """a / (b + x)  — hyperbolic decay."""
    return a / (b + x)

def f_x_over_b_plus_x(x, a, b):
    """a * x / (b + x)  — Michaelis-Menten / Berg-Purcell saturation.

    This is the expected functional form from the analytical model.
    At small x (few receptors): J/J_max ≈ (a/b)*x — linear in receptor count.
    At large x (many receptors): J/J_max → a — saturates at full efficiency.
    ``b`` has units of the total receptor aperture and encodes the geometric
    diffusion resistance.
    """
    return a * x / (b + x)

def f_power(x, a, b):
    """a * x^b  — power law."""
    return a * x**b

def f_linear(x, a, b):
    """a * x + b  — linear baseline."""
    return a * x + b

def f_logistic_like(x, a, b, c):
    """a * x^c / (b + x^c)  — generalised Hill equation.

    Reduces to ``f_x_over_b_plus_x`` when ``c = 1``.  The Hill coefficient
    ``c > 1`` indicates cooperative behaviour; ``c < 1`` indicates sub-linear
    saturation.
    """
    return a * x**c / (b + x**c)


# Map of model names to (function, initial parameters)
MODELS = {
    "ln(x)":           (f_ln,                      [1.0, 0.0]),
    "1/ln(x)":         (f_inv_ln,                  [1.0, 1.0]),
    "ln(1/x)":         (f_ln_invx,                 [1.0, 1.0]),
    "1/ln(1/x)":       (f_inv_ln_invx,             [1.0, 1.0]),
    "1/(1+ln(x))":     (f_one_over_1_plus_ln,      [1.0, 1.0]),
    "1/(1+ln(1/x))":   (f_one_over_1_plus_ln_invx, [1.0, 1.0]),
    "a/(b+x)":         (f_a_over_b_plus_x,         [1.0, 0.01]),
    "x/(b+x)":         (f_x_over_b_plus_x,         [1.0, 0.01]),
    "power":           (f_power,                   [1.0, 1.0]),
    "linear":          (f_linear,                  [1.0, 0.0]),
    "logistic":        (f_logistic_like,            [1.0, 1e4, 1.0]),
}


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _fit_single(func, x, y, p0):
    """Fit one model to the data and return fit statistics.

    Parameters
    ----------
    func : callable
        Model function ``func(x, *params)``.
    x : array
        Independent variable values (must be > 0 for log-based models).
    y : array
        Observed ``J/J_max`` values.
    p0 : list
        Initial parameter guesses for ``scipy.optimize.curve_fit``.

    Returns
    -------
    dict with keys ``params``, ``r2``, ``model``, ``y_pred_full``
    or ``None`` if the fit failed to converge.
    """
    mask = x > 0
    x_fit, y_fit = x[mask], y[mask]
    try:
        popt, _ = curve_fit(func, x_fit, y_fit, p0=p0, maxfev=10000)
    except RuntimeError:
        return None

    y_pred = func(x_fit, *popt)
    ss_res = np.sum((y_fit - y_pred)**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"params": popt, "r2": r2, "model": func, "y_pred_full": func(x, *popt)}


# ------------------------------------------------------------------
# Public fitting functions
# ------------------------------------------------------------------

def fit_J_over_Jmax(n_patches, sim_ratios, r_patch, make_plot=True):
    """Fit ``J/J_max = f(x)`` where ``x = n_patches * r_patch``.

    Tries all 11 model forms in ``MODELS`` and prints the R² for each.  The
    best-fitting models are plotted if ``make_plot=True``.

    Why ``x = n_patches * r_patch``?
    ---------------------------------
    The Berg-Purcell formula is ``μ = Ns / (Ns + b)`` where ``N`` is patch
    count and ``s`` is patch radius.  The natural variable is therefore
    ``x = Ns`` — the total receptor "aperture length".  Fitting as a function
    of ``x`` rather than ``N`` alone makes the fit independent of the choice
    of patch size.

    Parameters
    ----------
    n_patches : array-like of int
        Patch counts tested in the simulation.
    sim_ratios : array-like of float
        Simulated ``J/J_max`` values corresponding to each patch count.
    r_patch : float
        Patch radius (µm).
    make_plot : bool
        If True, save a figure ``best_J_over_Jmax_fits.png`` with the three
        best-fitting models overlaid on the simulation data.

    Returns
    -------
    fit_results : dict
        Maps model name (str) to a dict with keys:
        - ``params``: fitted parameter array
        - ``r2``: coefficient of determination R²
        - ``model``: the model function
        - ``y_pred_full``: predicted values at all ``x`` points
    """
    x = np.asarray(n_patches, float) * float(r_patch)
    y = np.asarray(sim_ratios, float)

    fit_results = {}
    print("\n=== Fitting models for J/J_max ===")
    for name, (func, p0) in MODELS.items():
        res = _fit_single(func, x, y, p0)
        if res is not None:
            fit_results[name] = res
            print(f"  {name:20s} R²={res['r2']:.4f}  params={res['params']}")
        else:
            print(f"  {name:20s} FIT FAILED")

    if make_plot and fit_results:
        # Plot the three best-fitting models
        best_names = sorted(fit_results, key=lambda k: fit_results[k]["r2"], reverse=True)[:3]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y, "o", label="Simulation", markersize=6)
        for name in best_names:
            res = fit_results[name]
            ax.plot(x, res["y_pred_full"], "--", label=f"{name} (R²={res['r2']:.3f})")
        ax.set_xlabel("x = n_patches * r_patch")
        ax.set_ylabel("J / J_max")
        ax.set_title("Best fits for J/J_max")
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig("best_J_over_Jmax_fits.png", dpi=300)
        plt.close(fig)

    return fit_results


def fit_bp_shape(n_patches_sorted, sim_ratios, r_patch):
    """Fit the Berg-Purcell shape ``J/J_max = x / (b + x)`` to simulation data.

    This is a single-parameter fit (only ``b`` is free; the amplitude is fixed
    at 1 because ``J/J_max → 1`` as ``x → ∞``).  It directly tests whether the
    simulation follows the Berg-Purcell prediction.

    The fitted parameter ``b`` is the value of ``x = N·s`` at which
    ``J/J_max = 0.5`` — the "half-saturation" receptor aperture.  The
    analytical value is ``b = π·H / (2·ln(R/a))``.

    Parameters
    ----------
    n_patches_sorted : array-like of int
        Patch counts in ascending order.
    sim_ratios : array-like of float
        Corresponding ``J/J_max`` values from simulation.
    r_patch : float
        Patch radius (µm).

    Returns
    -------
    (b, r2)

    b : float
        Fitted half-saturation aperture (µm).  Compare to the analytical value
        ``π·H / (2·ln(R/a))``.
    r2 : float
        R² (coefficient of determination) of the fit.
    """
    x = np.asarray(n_patches_sorted, float) * float(r_patch)
    y = np.asarray(sim_ratios, float)

    def model(x, b):
        return x / (b + x)

    mask = x > 0
    popt, _ = curve_fit(model, x[mask], y[mask], p0=[np.median(x[mask])],
                        bounds=(0, np.inf), maxfev=10000)
    b = popt[0]

    y_pred = model(x[mask], b)
    ss_res = np.sum((y[mask] - y_pred)**2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask]))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return b, r2
