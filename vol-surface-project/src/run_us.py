"""
Run the full pipeline for a US ticker: fetch chain -> compute IV surface
and Greeks -> save CSV + plots.

Usage:
    python run_us.py SPY
    python run_us.py AAPL --expiries 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_us import get_spot, get_risk_free_rate, get_option_chain
from surface import build_surface, plot_surface, plot_smile_by_expiry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="e.g. SPY, QQQ, AAPL")
    parser.add_argument("--expiries", type=int, default=8)
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--min-open-interest", type=int, default=10)
    parser.add_argument("--max-spread-pct", type=float, default=0.15)
    parser.add_argument("--outdir", default="../output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Fetching spot and risk-free rate for {args.ticker}...")
    spot = get_spot(args.ticker)
    r = get_risk_free_rate()
    print(f"  spot = {spot:.2f}, r = {r:.4f}")

    print(f"Fetching option chain ({args.expiries} expiries)...")
    chain = get_option_chain(args.ticker, max_expiries=args.expiries)
    print(f"  {len(chain)} raw rows")

    print(f"Computing implied vol + Greeks ({args.option_type}s)...")
    surface = build_surface(chain, spot, r, q=0.0, option_type=args.option_type,
                             min_open_interest=args.min_open_interest,
                             max_spread_pct=args.max_spread_pct)
    print(f"  {len(surface)} rows with valid IV solves after liquidity filtering "
          f"(min_open_interest={args.min_open_interest}, max_spread_pct={args.max_spread_pct})")

    csv_path = os.path.join(args.outdir, f"{args.ticker}_{args.option_type}_surface.csv")
    surface.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")

    plot_surface(surface, title=f"{args.ticker} IV Surface ({args.option_type}s)",
                 save_path=os.path.join(args.outdir, f"{args.ticker}_{args.option_type}_surface_3d.png"))
    plot_smile_by_expiry(surface, title=f"{args.ticker} Vol Smile ({args.option_type}s)",
                          save_path=os.path.join(args.outdir, f"{args.ticker}_{args.option_type}_smile.png"))

    print("\nSummary stats:")
    print(surface[["moneyness", "T", "iv", "delta", "gamma", "vega", "theta"]].describe())


if __name__ == "__main__":
    main()
