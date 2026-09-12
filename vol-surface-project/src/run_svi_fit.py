"""
Fit an SVI parametric curve to each expiry of a vol surface, for either
a US ticker (live yfinance chain) or NIFTY/BANKNIFTY (daily bhavcopy).

Usage:
    python run_svi_fit.py us SPY --expiries 8
    python run_svi_fit.py nifty NIFTY --expiries 6
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from surface import build_surface
from svi_fit import fit_svi_surface, plot_svi_fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("market", choices=["us", "nifty"])
    parser.add_argument("symbol")
    parser.add_argument("--expiries", type=int, default=8)
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--min-points", type=int, default=10,
                         help="Minimum strikes required to fit an expiry's SVI slice")
    parser.add_argument("--outdir", default="../output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.market == "us":
        from data_us import get_spot, get_risk_free_rate, get_option_chain
        spot = get_spot(args.symbol)
        r = get_risk_free_rate()
        chain = get_option_chain(args.symbol, max_expiries=args.expiries)
        q = 0.0
    else:
        from data_nifty import get_option_chain
        from data_us import get_spot as get_yf_spot
        chain = get_option_chain(args.symbol, max_expiries=args.expiries)
        spot_series = chain["underlying_spot"].dropna()
        spot = float(spot_series.iloc[0]) if not spot_series.empty else get_yf_spot("^NSEI")
        r = 0.065
        q = 0.01

    print(f"Spot: {spot:.2f}, r: {r:.4f}, q: {q:.4f}")
    surface = build_surface(chain, spot, r, q=q, option_type=args.option_type)
    print(f"{len(surface)} rows in the liquidity-filtered surface")

    fits = fit_svi_surface(surface, spot, r, q=q, min_points=args.min_points)
    if fits.empty:
        print(f"No expiry had at least {args.min_points} strikes -- try more expiries "
              f"or a lower --min-points.")
        return

    print("\nSVI fits per expiry:")
    print(fits[["expiry", "T", "n_points", "a", "b", "rho", "m", "sigma",
                "rmse_iv", "converged"]].to_string())

    csv_path = os.path.join(args.outdir, f"{args.symbol}_{args.option_type}_svi_fits.csv")
    fits.to_csv(csv_path, index=False)
    print(f"\nSaved fit params to {csv_path}")

    plot_svi_fits(surface, fits, spot, r, q=q,
                  title=f"{args.symbol}: SVI fit vs market IV ({args.option_type}s)",
                  save_path=os.path.join(args.outdir, f"{args.symbol}_{args.option_type}_svi_fit.png"))


if __name__ == "__main__":
    main()
