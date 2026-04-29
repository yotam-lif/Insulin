# fitting_models_static3d.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================
#  Model functions
# ============================
def f_ln(x, a, b):                return a * np.log(x) + b
def f_inv_ln(x, a, b):            return a / (np.log(x) + b)
def f_ln_invx(x, a, b):           return a * np.log(1.0 / x) + b
def f_inv_ln_invx(x, a, b):       return a / (np.log(1.0 / x) + b)
def f_one_over_1_plus_ln(x, a, b):       return a / (1.0 + b * np.log(x))
def f_one_over_1_plus_ln_invx(x, a, b):  return a / (1.0 + b * np.log(1.0 / x))
def f_a_over_b_plus_x(x, a, b):          return a / (b + x)
def f_x_over_b_plus_x(x, a, b):          return a * x / (b + x)
def f_power(x, a, b):             return a * x**b
def f_linear(x, a, b):            return a * x + b

def f_logistic_like(x, a, b, c):
    # saturating Hill / logistic-like shape
    return a * x**c / (b + x**c)


# ============================
#  Fitting helper
# ============================
def _fit_single_model(model_func, x, y, p0):
    # guard for log-based models
    mask = x > 0
    x_fit = x[mask]
    y_fit = y[mask]

    try:
        popt, _ = curve_fit(model_func, x_fit, y_fit, p0=p0, maxfev=10000)
    except RuntimeError:
        return None

    y_pred = model_func(x_fit, *popt)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "params": popt,
        "r2": r2,
        "model": model_func,
        "y_pred_full": model_func(x, *popt),
    }


# ============================
#  Main fitting interface
# ============================
def fit_J_over_Jmax_models(n_patches, sim_ratios, r_patch, make_plot=True):
    """
    Fits J/J_max = f(x) with x = n_patches * r_patch.
    Returns dict of fit results for all models.
    """

    # IMPORTANT: use x = n_patches * r_patch (not just r_patch)
    x = n_patches * r_patch
    y = sim_ratios

    models = {
        "ln(x)":                (f_ln,                 [1.0, 0.0]),
        "1/ln(x)":              (f_inv_ln,             [1.0, 1.0]),
        "ln(1/x)":              (f_ln_invx,            [1.0, 1.0]),
        "1/ln(1/x)":            (f_inv_ln_invx,        [1.0, 1.0]),
        "1/(1+ln(x))":          (f_one_over_1_plus_ln, [1.0, 1.0]),
        "1/(1+ln(1/x))":        (f_one_over_1_plus_ln_invx, [1.0, 1.0]),
        "a/(b+x)":              (f_a_over_b_plus_x,    [1.0, 0.01]),
        "x/(b+x)":              (f_x_over_b_plus_x,    [1.0, 0.01]),
        "power":                (f_power,              [1.0, 1.0]),
        "linear":               (f_linear,             [1.0, 0.0]),
        "logistic_like":        (f_logistic_like,      [1.0, 1e4, 1.0]),
    }

    fit_results = {}

    print("\n=== Fitting models for J/J_max ===")
    for name, (func, p0) in models.items():
        res = _fit_single_model(func, x, y, p0)
        if res is not None:
            fit_results[name] = res
            print(f"{name:15s} | R^2 = {res['r2']:.4f} | params = {res['params']}")
        else:
            print(f"{name:15s} | FIT FAILED")

    # Plot best 3 fits
    if make_plot and fit_results:
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, "o", label="Simulation (3D static patches)", markersize=6)

        best = sorted(
            fit_results.keys(),
            key=lambda k: fit_results[k]["r2"],
            reverse=True,
        )[:3]

        for name in best:
            res = fit_results[name]
            plt.plot(
                x,
                res["y_pred_full"],
                "--",
                label=f"{name} (R²={res['r2']:.3f})",
            )

        plt.xlabel("x = n_patches * r_patch")
        plt.ylabel("J / J_max")
        plt.title("Best analytic fits for 3D static-patch simulation")
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig("best_J_over_Jmax_fits_static3d.png", dpi=300)
        print("\nSaved: best_J_over_Jmax_fits_static3d.png\n")
        plt.show()

    return fit_results


def fit_b_from_sweep(n_patches_sorted, sim_ratios, r_patch):
    """
    Special fit for J/J_max ≈ x / (b + x), x = n_patches * r_patch.
    Returns (b, R^2).
    """
    x = n_patches_sorted * r_patch
    y = sim_ratios

    def model(x, b):
        return x / (b + x)

    mask = x > 0
    x_fit = x[mask]
    y_fit = y[mask]

    b0 = np.median(x_fit)
    popt, _ = curve_fit(
        model, x_fit, y_fit, p0=[b0], bounds=(0, np.inf), maxfev=10000
    )
    b = popt[0]

    y_pred = model(x_fit, b)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return b, r2


# ============================
#  Run directly on your data
# ============================
def model_fit(n_patches,J_over_Jmax,r_patch):
    # -------------------------
    # 1. Paste the data here
    # -------------------------


    # -------------------------
    # 2. Fit all models
    # -------------------------
    fit_results = fit_J_over_Jmax_models(
        n_patches=n_patches,
        sim_ratios=J_over_Jmax,
        r_patch=r_patch,
        make_plot=True,
    )

    # find and print the best model
    best_name = max(fit_results, key=lambda k: fit_results[k]["r2"])
    best = fit_results[best_name]
    print("\n>>> BEST MODEL <<<")
    print(f"Name   : {best_name}")
    print(f"R^2    : {best['r2']:.4f}")
    print(f"params : {best['params']}")