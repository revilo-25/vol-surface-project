"""
Black-Scholes pricing, Greeks, and implied volatility solver.

Conventions:
    S     : spot price
    K     : strike price
    T     : time to expiry, in years
    r     : risk-free rate (annualized, continuous compounding)
    q     : dividend / index yield (annualized, continuous compounding)
    sigma : volatility (annualized)
    option_type : 'call' or 'put'
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, q, sigma, option_type="call"):
    """Black-Scholes-Merton price with continuous dividend yield q."""
    if T <= 0:
        # At/after expiry -> intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def implied_vol(price, S, K, T, r, q, option_type="call",
                 lo=1e-4, hi=5.0, tol=1e-6):
    """
    Solve for sigma such that bs_price(...) == price, via Brent's method.
    Returns np.nan if no root is found in [lo, hi] (e.g. price outside
    no-arbitrage bounds, or too close to intrinsic value / zero).
    """
    if price is None or np.isnan(price) or price <= 0 or T <= 0:
        return np.nan

    # No-arbitrage sanity check: price must exceed intrinsic value bound
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if price < intrinsic * np.exp(-q * T) - 1e-8:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, q, sigma, option_type) - price

    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(objective, lo, hi, xtol=tol)
    except (ValueError, RuntimeError):
        return np.nan


def greeks(S, K, T, r, q, sigma, option_type="call"):
    """
    Returns a dict of delta, gamma, vega, theta, rho.
    - vega  : price change per 1.00 (100%) change in sigma -> divide by 100
              for a "per 1 vol point" number if you prefer that convention.
    - theta : price change per YEAR -> divide by 365 for per-day decay.
    """
    if T <= 0 or sigma <= 0:
        return {"delta": np.nan, "gamma": np.nan, "vega": np.nan,
                "theta": np.nan, "rho": np.nan}

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)  # per 1.00 change in sigma

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


if __name__ == "__main__":
    # Quick sanity check
    S, K, T, r, q, sigma = 100, 100, 0.5, 0.05, 0.01, 0.20
    price = bs_price(S, K, T, r, q, sigma, "call")
    print(f"Call price: {price:.4f}")
    iv = implied_vol(price, S, K, T, r, q, "call")
    print(f"Recovered IV: {iv:.4f} (should be close to {sigma})")
    print("Greeks:", greeks(S, K, T, r, q, sigma, "call"))
