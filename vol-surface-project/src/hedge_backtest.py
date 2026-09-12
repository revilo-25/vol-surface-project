"""
Delta-hedging backtest: simulate the P&L of writing (or buying) a
Black-Scholes-priced option and delta-hedging it daily over a REAL
historical underlying price path, using an assumed constant hedging
volatility.

This is deliberately not a "backtest" of a live/current option -- you
can't backtest something whose future price path hasn't happened yet.
Instead, it answers the classic textbook question: "if I had sold this
option T days ago and delta-hedged it daily using volatility sigma, what
P&L would I have made, given what actually happened to the underlying?"

The core result this demonstrates: a delta-hedged option seller's P&L is
driven by the gap between the volatility they hedged with (sigma_hedge)
and the volatility that was actually realized over the option's life.
Hedge with vol higher than what's realized -> seller profits (they
collected too much premium for the risk that materialized), and
vice versa. This is why the module reports both numbers side by side.
"""

import numpy as np
import pandas as pd

from black_scholes import bs_price, greeks


def realized_vol(prices, trading_days_per_year: int = 252) -> float:
    """Annualized realized volatility from a series of prices (close-to-close log returns)."""
    prices = np.asarray(prices, dtype=float)
    log_rets = np.diff(np.log(prices))
    return float(np.std(log_rets, ddof=1) * np.sqrt(trading_days_per_year))


def simulate_delta_hedge(prices, T_years: float, K: float, r: float, q: float,
                          sigma_hedge: float, option_type: str = "call",
                          position: str = "short") -> dict:
    """
    prices: array-like of underlying prices sampled at equal intervals from
            inception (index 0) to expiry (index -1), inclusive. Length
            N+1 implies N rebalancing steps of size dt = T_years / N.
    sigma_hedge: the constant volatility assumed when pricing the option
                 and computing the daily hedge ratio (delta). This is the
                 vol *used to hedge*, not necessarily what actually
                 happens to the underlying -- that's realized_vol(prices).
    position: 'short' (you wrote/sold the option, collected premium,
              hedge by holding +delta shares) or 'long' (you bought the
              option, hedge by holding -delta shares). The two are exact
              mirror images of each other under this simulation, so
              'long' is computed as -1 * the 'short' result.

    Returns a dict with:
        initial_premium, final_payoff, final_hedge_value,
        hedging_pnl, pnl_pct_of_premium, sigma_hedge, realized_vol,
        path -- a DataFrame with the full step-by-step simulation
    """
    prices = np.asarray(prices, dtype=float)
    N = len(prices) - 1
    if N < 1:
        raise ValueError("Need at least 2 price points (inception + 1 more) to simulate a hedge")
    dt = T_years / N

    S0 = prices[0]
    V0 = bs_price(S0, K, T_years, r, q, sigma_hedge, option_type)
    delta0 = greeks(S0, K, T_years, r, q, sigma_hedge, option_type)["delta"]

    shares = delta0
    # Writer receives V0 premium, spends delta0*S0 buying the initial
    # hedge; the remainder sits in a cash account earning/paying rate r.
    cash = V0 - shares * S0

    rows = [{"t": 0.0, "S": S0, "delta": delta0, "shares": shares,
             "cash": cash, "portfolio_value": cash + shares * S0}]

    for i in range(1, N + 1):
        cash *= np.exp(r * dt)  # cash account accrues at the risk-free rate
        S_i = prices[i]
        t_remaining = T_years - i * dt

        if t_remaining > 1e-8:
            delta_i = greeks(S_i, K, t_remaining, r, q, sigma_hedge, option_type)["delta"]
        else:
            # At expiry, delta collapses to the in/out-of-the-money indicator.
            if option_type == "call":
                delta_i = 1.0 if S_i > K else 0.0
            else:
                delta_i = -1.0 if S_i < K else 0.0

        d_shares = delta_i - shares
        cash -= d_shares * S_i  # buy/sell shares to rebalance, funded from cash
        shares = delta_i

        rows.append({"t": i * dt, "S": S_i, "delta": delta_i, "shares": shares,
                     "cash": cash, "portfolio_value": cash + shares * S_i})

    path = pd.DataFrame(rows)

    S_final = prices[-1]
    payoff = max(S_final - K, 0.0) if option_type == "call" else max(K - S_final, 0.0)
    final_hedge_value = path["portfolio_value"].iloc[-1]

    # Writer's P&L: what the hedge portfolio is worth, minus what's owed
    # to the option holder at expiry.
    hedging_pnl = final_hedge_value - payoff
    if position == "long":
        hedging_pnl = -hedging_pnl
    elif position != "short":
        raise ValueError("position must be 'short' or 'long'")

    return {
        "initial_premium": V0,
        "final_payoff": payoff,
        "final_hedge_value": final_hedge_value,
        "hedging_pnl": hedging_pnl,
        "pnl_pct_of_premium": hedging_pnl / V0 if V0 > 0 else np.nan,
        "sigma_hedge": sigma_hedge,
        "realized_vol": realized_vol(prices),
        "position": position,
        "path": path,
    }


def rolling_hedge_backtest(price_series: pd.Series, window_days: int,
                            vol_lookback_days: int, r: float, q: float = 0.0,
                            option_type: str = "call", strike_moneyness: float = 1.0,
                            position: str = "short", step_days: int = 5) -> pd.DataFrame:
    """
    Repeats simulate_delta_hedge over many overlapping historical windows
    to build a distribution of hedging outcomes, so you can see the
    sigma_hedge-vs-realized_vol relationship empirically rather than as a
    single data point.

    For each window:
        - sigma_hedge is estimated from realized vol over the
          `vol_lookback_days` immediately BEFORE the window starts (so
          the hedge vol is only using information available before the
          option's life begins -- no look-ahead bias)
        - the option's strike is set as strike_moneyness * S at the start
          of the window (e.g. 1.0 = at-the-money)
        - step_days controls how far each rolling window start advances
          (5 = roughly weekly windows, less overlap/autocorrelation than
          daily)

    Returns a DataFrame, one row per window, with sigma_hedge, realized_vol
    (during the window itself), hedging_pnl, pnl_pct_of_premium, and the
    window's start/end dates.
    """
    prices = price_series.values
    dates = price_series.index
    T_years = window_days / 252.0

    results = []
    start = vol_lookback_days
    while start + window_days < len(prices):
        lookback_prices = prices[start - vol_lookback_days: start + 1]
        sigma_hedge = realized_vol(lookback_prices)

        window_prices = prices[start: start + window_days + 1]
        K = window_prices[0] * strike_moneyness

        try:
            result = simulate_delta_hedge(window_prices, T_years, K, r, q,
                                           sigma_hedge, option_type, position)
        except Exception:
            start += step_days
            continue

        results.append({
            "start_date": dates[start],
            "end_date": dates[start + window_days],
            "sigma_hedge": result["sigma_hedge"],
            "realized_vol": result["realized_vol"],
            "vol_gap": result["sigma_hedge"] - result["realized_vol"],
            "initial_premium": result["initial_premium"],
            "hedging_pnl": result["hedging_pnl"],
            "pnl_pct_of_premium": result["pnl_pct_of_premium"],
        })
        start += step_days

    return pd.DataFrame(results)
