"""
SVI (Stochastic Volatility Inspired) parametric volatility smile fit.

SVI models TOTAL VARIANCE w(k) = IV^2 * T as a function of log-moneyness
k = ln(K/F) (F = forward price), one slice (one expiry) at a time:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

Parameters:
    a      -- overall level of variance (vertical shift)
    b      -- controls the overall steepness/wing slope (b >= 0)
    rho    -- controls the skew/tilt of the smile (-1 < rho < 1)
    m      -- horizontal shift of the smile's minimum
    sigma  -- controls the curvature at the minimum (sigma > 0)

Why fit this instead of just using the raw scattered IV points:
    - Gives a smooth, arbitrage-checked curve you can evaluate at ANY
      strike/expiry, not just the ones that happened to be listed
      (interpolation "for free")
    - The 5 fitted parameters ARE a compact description of the smile's
      shape (level, skew, curvature) -- comparing them directly across
      SPY and NIFTY is a cleaner comparison than eyeballing two scatter
      plots
    - It's the industry-standard model for this (Gatheral 2004), so
      using it demonstrates familiarity with how real trading desks
      actually represent vol surfaces, not just "solve BS and plot"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def svi_total_variance(k, a, b, rho, m, sigma):
    """w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_iv(k, T, a, b, rho, m, sigma):
    """Convert SVI total variance back to annualized IV: iv = sqrt(w/T)."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 1e-8) / T)


def _initial_guess(k, w):
    """
    Rough starting point from the data itself. SVI fits are sensitive to
    starting point -- in particular, b needs to be scaled to the actual
    magnitude of w's range, not a fixed constant. Total variance w is
    iv^2 * T, so for a short-dated or low-vol slice, w can be as small as
    1e-4; a fixed b0 like 0.1 is then 2-3 orders of magnitude too large,
    and the optimizer just drives b down to its lower bound instead of
    ever finding the right (much smaller) value -- exactly the failure
    mode that showed up fitting real SPY data during development (3 of 4
    expiries collapsed to a flat line before this fix).
    """
    a0 = max(w.min() * 0.9, 1e-6)
    k_range = max(k.max() - k.min(), 1e-4)
    b0 = max((w.max() - w.min()) / k_range, 1e-6)
    rho0 = 0.0
    m0 = k[np.argmin(w)]
    sigma0 = max(k_range / 4, 0.01)
    return [a0, b0, rho0, m0, sigma0]


def fit_svi_slice(k, w, weights=None):
    """
    Fits one SVI slice (one expiry) via constrained least squares.
    k: log-moneyness array, w: total variance array (both for the same
    expiry). Returns a dict of fitted params plus fit diagnostics
    (rmse in variance space and in IV-point terms).

    Bounds enforce the basic SVI sanity constraints (b >= 0, |rho| < 1,
    sigma > 0) but this is NOT a full no-static-arbitrage check (that
    requires checking the wider Gatheral-Jacquier conditions across the
    whole surface, not just one slice) -- fine for a portfolio project,
    but worth knowing if you take this further.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if weights is None:
        weights = np.ones_like(w)

    def objective(params):
        a, b, rho, m, sigma = params
        model_w = svi_total_variance(k, a, b, rho, m, sigma)
        return np.sum(weights * (model_w - w) ** 2)

    x0 = _initial_guess(k, w)
    bounds = [
        (1e-6, None),        # a >= 0
        (1e-6, 5.0),         # b >= 0
        (-0.999, 0.999),     # rho in (-1, 1)
        (k.min() - 1.0, k.max() + 1.0),  # m roughly within the observed range
        (1e-4, 5.0),         # sigma > 0
    ]

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    a, b, rho, m, sigma = result.x

    fitted_w = svi_total_variance(k, a, b, rho, m, sigma)
    rmse_w = np.sqrt(np.mean((fitted_w - w) ** 2))

    return {
        "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
        "rmse_variance": rmse_w,
        "converged": result.success,
        "n_points": len(k),
    }


def fit_svi_surface(surface_df: pd.DataFrame, spot: float, r: float, q: float = 0.0,
                     min_points: int = 10) -> pd.DataFrame:
    """
    Fits one SVI slice per expiry in `surface_df` (which must already
    have T, strike, iv columns -- i.e. the output of surface.build_surface).
    Returns one row per expiry with the fitted params and diagnostics.

    min_points: SVI has 5 free parameters, so a slice with only a handful
    of strikes is under-determined -- the optimizer will still "converge"
    but can land on a degenerate solution (e.g. b pinned to its lower
    bound) that fits those few points but doesn't represent a real smile
    shape. Slices below this threshold are skipped rather than returning
    an unreliable fit; 10 is a reasonable floor (2x the parameter count).
    """
    rows = []
    for expiry, group in surface_df.groupby("expiry"):
        T = group["T"].iloc[0]
        if T <= 0 or len(group) < min_points:
            continue

        F = spot * np.exp((r - q) * T)
        k = np.log(group["strike"].values / F)
        w = (group["iv"].values ** 2) * T

        fit = fit_svi_slice(k, w)
        fit["expiry"] = expiry
        fit["T"] = T
        # RMSE in IV-point terms is more interpretable than raw variance
        # units -- convert the fitted curve back to IV and compare.
        fitted_iv = svi_iv(k, T, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
        fit["rmse_iv"] = np.sqrt(np.mean((fitted_iv - group["iv"].values) ** 2))
        rows.append(fit)

    return pd.DataFrame(rows)


def svi_smile_curve(fit_row: pd.Series, spot: float, r: float, q: float = 0.0,
                     moneyness_range=(0.85, 1.15), n_points: int = 200) -> pd.DataFrame:
    """
    Evaluates a fitted SVI slice across a dense grid of strikes -- this
    is the "interpolation for free" part: you can read off the model IV
    at any strike, not just the ones that were actually listed.
    """
    T = fit_row["T"]
    F = spot * np.exp((r - q) * T)
    moneyness = np.linspace(*moneyness_range, n_points)
    strikes = moneyness * spot
    k = np.log(strikes / F)

    iv = svi_iv(k, T, fit_row["a"], fit_row["b"], fit_row["rho"],
                fit_row["m"], fit_row["sigma"])

    return pd.DataFrame({"moneyness": moneyness, "strike": strikes, "iv": iv})


def plot_svi_fits(surface_df: pd.DataFrame, svi_fits: pd.DataFrame, spot: float,
                   r: float, q: float = 0.0, title: str = "SVI fit vs market IV",
                   save_path=None):
    """
    One panel per fitted expiry: raw market IV points (scatter) overlaid
    with the fitted SVI curve (line), so you can see fit quality directly
    rather than trusting the RMSE number alone.
    """
    import matplotlib.pyplot as plt

    n = len(svi_fits)
    if n == 0:
        raise ValueError("No SVI fits to plot -- fit_svi_surface returned no rows "
                          "(likely every expiry had fewer than min_points)")

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, (_, fit_row) in enumerate(svi_fits.iterrows()):
        ax = axes[i // ncols][i % ncols]
        expiry = fit_row["expiry"]
        market = surface_df[surface_df["expiry"] == expiry]

        ax.scatter(market["moneyness"], market["iv"] * 100, s=15, alpha=0.6,
                   label="Market IV", color="tab:blue")

        curve = svi_smile_curve(fit_row, spot, r, q)
        ax.plot(curve["moneyness"], curve["iv"] * 100, color="tab:red",
                linewidth=1.5, label="SVI fit")

        ax.set_title(f"{expiry} (n={fit_row['n_points']}, "
                     f"rmse={fit_row['rmse_iv']*100:.1f}pt)", fontsize=10)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # hide any unused subplot axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig
