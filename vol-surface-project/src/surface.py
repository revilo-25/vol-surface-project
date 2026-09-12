"""
Turn a raw option chain (from data_us.py or data_nifty.py) into an IV +
Greeks surface, and plot it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)

from black_scholes import implied_vol, greeks


def build_surface(chain: pd.DataFrame, spot: float, r: float, q: float = 0.0,
                   option_type: str = "call",
                   moneyness_bounds=(0.85, 1.15),
                   min_open_interest: int = 10,
                   min_oi_percentile: float = 50.0,
                   max_spread_pct: float = 0.15,
                   max_iv: float = 1.5) -> pd.DataFrame:
    """
    Adds iv, delta, gamma, vega, theta, rho, moneyness columns to a copy
    of `chain`, filtered to one option_type, a moneyness band, a
    liquidity filter, and an IV sanity cap.

    The liquidity filter matters more than the moneyness band for smile
    quality: deep ITM/OTM strikes have very low vega, so a few cents of
    ordinary bid-ask noise (or a stale settlement price) translates into
    a large, spurious swing in the solved IV -- this shows up as sharp
    sawtooth spikes in a smile plot, not real information.

    An absolute open-interest floor isn't enough, though: an OI of 10-50
    contracts still produces visible noise for a name like SPY where the
    truly liquid strikes sit in the thousands-to-tens-of-thousands, while
    that same absolute number would wrongly discard perfectly good NIFTY
    strikes, whose OI runs in the hundreds-of-thousands to millions (a
    totally different scale). So the primary filter is RELATIVE:
    `min_oi_percentile` keeps only strikes at or above that percentile of
    open interest within their own expiry, which auto-adapts to whatever
    scale the underlying trades at. `min_open_interest` remains as a
    absolute floor underneath that (mostly relevant for very sparse
    chains where the percentile filter alone might not be enough).
    `max_spread_pct` catches wide bid-ask spreads on top of that (only
    applied when bid/ask columns are present, i.e. the US live-chain
    case).

    Even liquid, tight-spread strikes can still produce a nonsense IV,
    though -- e.g. a deep ITM American SPY call near expiry, priced only
    a few cents above intrinsic value. That's a real market price, not a
    bad quote, but it reflects American-exercise/dividend effects our
    European Black-Scholes model doesn't capture, not genuine implied
    volatility. Since extrinsic value there is nearly zero, vega is too,
    and the "IV" needed to match that price can be enormous. Tightening
    the default moneyness band to (0.85, 1.15) keeps clear of most of
    that region; `max_iv` is a final backstop that drops anything still
    priced by the solver to an implausible level (default cap: 150%).
    """
    df = chain[chain["option_type"] == option_type].copy()
    df["moneyness"] = df["strike"] / spot
    lo, hi = moneyness_bounds

    df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]
    df = df[df["mid_price"] > 0]

    if "open_interest" in df.columns:
        df = df[df["open_interest"].fillna(0) >= min_open_interest]
        if min_oi_percentile > 0 and "expiry" in df.columns and len(df) > 0:
            oi_thresholds = df.groupby("expiry")["open_interest"].transform(
                lambda x: x.quantile(min_oi_percentile / 100.0))
            df = df[df["open_interest"] >= oi_thresholds]

    if "bid" in df.columns and "ask" in df.columns:
        spread_pct = (df["ask"] - df["bid"]).abs() / df["mid_price"].replace(0, np.nan)
        df = df[spread_pct.isna() | (spread_pct <= max_spread_pct)]

    df = df.reset_index(drop=True)

    ivs, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []
    for _, row in df.iterrows():
        iv = implied_vol(row["mid_price"], spot, row["strike"], row["T"],
                          r, q, option_type)
        ivs.append(iv)
        if np.isnan(iv):
            deltas.append(np.nan); gammas.append(np.nan)
            vegas.append(np.nan); thetas.append(np.nan); rhos.append(np.nan)
        else:
            g = greeks(spot, row["strike"], row["T"], r, q, iv, option_type)
            deltas.append(g["delta"]); gammas.append(g["gamma"])
            vegas.append(g["vega"]); thetas.append(g["theta"]); rhos.append(g["rho"])

    df["iv"] = ivs
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas
    df["rho"] = rhos

    df = df.dropna(subset=["iv"])
    df = df[df["iv"] <= max_iv]
    return df.reset_index(drop=True)


def plot_surface(surface_df: pd.DataFrame, title: str = "Implied Volatility Surface",
                  save_path: str | None = None):
    """3D scatter/surface of moneyness x T x IV."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = surface_df["moneyness"]
    y = surface_df["T"] * 365  # display in days
    z = surface_df["iv"] * 100  # display in vol points

    sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=20)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title(title)
    fig.colorbar(sc, shrink=0.5, label="IV (%)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_smile_by_expiry(surface_df: pd.DataFrame, title: str = "Volatility Smile by Expiry",
                          save_path: str | None = None):
    """2D IV-vs-moneyness lines, one per expiry -- often the more readable view."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for exp, grp in surface_df.groupby("expiry"):
        grp = grp.sort_values("moneyness")
        ax.plot(grp["moneyness"], grp["iv"] * 100, marker="o", markersize=3, label=str(exp))

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig
