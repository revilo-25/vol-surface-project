"""
Run the full pipeline for NSE index options (Nifty / Bank Nifty).

Usage:
    python run_nifty.py NIFTY
    python run_nifty.py BANKNIFTY --expiries 4

Risk-free rate: we use a fixed proxy for the Indian short-term rate
(91-day T-bill ballpark) since there's no single free live-quote API
as convenient as yfinance's ^IRX for India. Override with --rate if
you have a current figure you trust more.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_nifty import get_option_chain
from data_us import get_spot as get_yf_spot  # fallback spot source
from surface import build_surface, plot_surface, plot_smile_by_expiry

DEFAULT_INDIA_RATE = 0.065  # update this periodically -- see docstring
YF_INDEX_TICKER = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", choices=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--expiries", type=int, default=6)
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--rate", type=float, default=DEFAULT_INDIA_RATE)
    parser.add_argument("--min-open-interest", type=int, default=10)
    parser.add_argument("--outdir", default="../output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Fetching {args.symbol} F&O bhavcopy ({args.expiries} expiries)...")
    chain = get_option_chain(args.symbol, max_expiries=args.expiries)
    if chain.empty:
        print("No data returned -- see the error above for what NSE's daily-reports "
              "API is currently offering, and check _FO_BHAVCOPY_FILEKEY_CANDIDATES "
              "in data_nifty.py against it.")
        return

    spot = chain["underlying_spot"].dropna()
    if not spot.empty:
        spot = float(spot.iloc[0])
    else:
        print(f"  Bhavcopy had no underlying price column -- falling back to "
              f"yfinance ({YF_INDEX_TICKER[args.symbol]})...")
        spot = get_yf_spot(YF_INDEX_TICKER[args.symbol])
    print(f"  spot = {spot:.2f}, using r = {args.rate:.4f}")
    print(f"  {len(chain)} raw rows")

    print(f"Computing implied vol + Greeks ({args.option_type}s)...")
    # Nifty index options -> treat as having a dividend-yield-like drag q
    # roughly equal to the index dividend yield; 0.01 (1%) is a reasonable
    # default for Nifty 50, adjust if you have a better estimate.
    surface = build_surface(chain, spot, args.rate, q=0.01, option_type=args.option_type,
                             min_open_interest=args.min_open_interest)
    print(f"  {len(surface)} rows with valid IV solves after liquidity filtering "
          f"(min_open_interest={args.min_open_interest})")

    csv_path = os.path.join(args.outdir, f"{args.symbol}_{args.option_type}_surface.csv")
    surface.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")

    plot_surface(surface, title=f"{args.symbol} IV Surface ({args.option_type}s)",
                 save_path=os.path.join(args.outdir, f"{args.symbol}_{args.option_type}_surface_3d.png"))
    plot_smile_by_expiry(surface, title=f"{args.symbol} Vol Smile ({args.option_type}s)",
                          save_path=os.path.join(args.outdir, f"{args.symbol}_{args.option_type}_smile.png"))

    print("\nSummary stats:")
    print(surface[["moneyness", "T", "iv", "delta", "gamma", "vega", "theta"]].describe())


if __name__ == "__main__":
    main()
