"""
Run the delta-hedging backtest for a US ticker.

Two modes:
  --mode single   One window (the most recent `--window-days` trading
                  days), with a full equity-curve plot of the hedge P&L
                  over time.
  --mode rolling  Many overlapping historical windows, showing the
                  relationship between (assumed hedge vol - realized vol)
                  and the resulting P&L -- the core takeaway of the whole
                  exercise.

Usage:
    python run_hedge_backtest.py SPY --mode single --window-days 30
    python run_hedge_backtest.py SPY --mode rolling --window-days 20 --vol-lookback-days 20
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data_us import get_price_history, get_risk_free_rate
from hedge_backtest import simulate_delta_hedge, rolling_hedge_backtest


def plot_single_run(result, ticker, save_path=None):
    path = result["path"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(path["t"] * 252, path["S"], color="tab:blue")
    axes[0].set_ylabel("Underlying price")
    axes[0].set_title(f"{ticker}: underlying path over the hedge window")
    axes[0].grid(alpha=0.3)

    axes[1].plot(path["t"] * 252, path["portfolio_value"], color="tab:orange")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("Trading days")
    axes[1].set_ylabel("Hedge portfolio value")
    axes[1].set_title(
        f"Hedge portfolio value over time (sigma_hedge={result['sigma_hedge']:.3f}, "
        f"realized_vol={result['realized_vol']:.3f})"
    )
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_rolling_results(df, ticker, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["vol_gap"] * 100, df["pnl_pct_of_premium"] * 100, alpha=0.6)

    # simple linear fit line for visual reference
    import numpy as np
    if len(df) > 2:
        coeffs = np.polyfit(df["vol_gap"], df["pnl_pct_of_premium"], 1)
        xs = np.linspace(df["vol_gap"].min(), df["vol_gap"].max(), 50)
        ax.plot(xs * 100, (coeffs[0] * xs + coeffs[1]) * 100, color="red",
                linewidth=1.5, label=f"fit slope={coeffs[0]:.2f}")
        ax.legend()

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Vol gap: sigma_hedge - realized_vol (vol points, %)")
    ax.set_ylabel("Hedging P&L (% of initial premium)")
    ax.set_title(f"{ticker}: delta-hedge P&L vs vol forecast error "
                 f"({len(df)} rolling windows)")
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--mode", choices=["single", "rolling"], default="single")
    parser.add_argument("--window-days", type=int, default=30,
                         help="Option life in trading days")
    parser.add_argument("--vol-lookback-days", type=int, default=30,
                         help="Trading days of prior history used to estimate sigma_hedge")
    parser.add_argument("--strike-moneyness", type=float, default=1.0,
                         help="Strike as a multiple of spot at window start (1.0 = ATM)")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--position", choices=["short", "long"], default="short")
    parser.add_argument("--step-days", type=int, default=5,
                         help="[rolling mode] trading days between window starts")
    parser.add_argument("--outdir", default="../output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    r = get_risk_free_rate()
    print(f"Risk-free rate: {r:.4f}")

    if args.mode == "single":
        history_days_needed = args.window_days + args.vol_lookback_days + 5
        prices = get_price_history(args.ticker, period=f"{max(history_days_needed, 60)}d")
        print(f"Fetched {len(prices)} days of price history for {args.ticker}")

        lookback_prices = prices.values[-(args.window_days + args.vol_lookback_days + 1):-args.window_days]
        window_prices = prices.values[-(args.window_days + 1):]

        from hedge_backtest import realized_vol
        sigma_hedge = realized_vol(lookback_prices)
        K = window_prices[0] * args.strike_moneyness
        T_years = args.window_days / 252.0

        result = simulate_delta_hedge(window_prices, T_years, K, r, 0.0,
                                       sigma_hedge, args.option_type, args.position)

        print(f"\n--- Single-window delta-hedge backtest: {args.ticker} ---")
        print(f"  Window: last {args.window_days} trading days")
        print(f"  Strike: {K:.2f} ({args.strike_moneyness}x spot at window start)")
        print(f"  sigma_hedge (estimated from prior {args.vol_lookback_days}d): {sigma_hedge:.4f}")
        print(f"  realized_vol (during the window): {result['realized_vol']:.4f}")
        print(f"  Initial premium: {result['initial_premium']:.4f}")
        print(f"  Final payoff: {result['final_payoff']:.4f}")
        print(f"  Hedging P&L ({args.position}): {result['hedging_pnl']:.4f} "
              f"({result['pnl_pct_of_premium']*100:.1f}% of premium)")

        plot_single_run(result, args.ticker,
                         save_path=os.path.join(args.outdir, f"{args.ticker}_hedge_single_window.png"))

    else:  # rolling
        prices = get_price_history(args.ticker, period="2y")
        print(f"Fetched {len(prices)} days of price history for {args.ticker}")

        df = rolling_hedge_backtest(
            prices, window_days=args.window_days, vol_lookback_days=args.vol_lookback_days,
            r=r, q=0.0, option_type=args.option_type, strike_moneyness=args.strike_moneyness,
            position=args.position, step_days=args.step_days
        )

        if df.empty:
            print("No rolling windows produced -- try a shorter window/lookback "
                  "or fetch more history.")
            return

        csv_path = os.path.join(args.outdir, f"{args.ticker}_hedge_rolling.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rolling windows to {csv_path}")

        print("\nSummary:")
        print(df[["sigma_hedge", "realized_vol", "vol_gap", "pnl_pct_of_premium"]].describe())
        print(f"\nCorrelation(vol_gap, pnl_pct_of_premium): "
              f"{df['vol_gap'].corr(df['pnl_pct_of_premium']):.3f}")
        print(f"Win rate ({args.position} position): "
              f"{(df['hedging_pnl'] > 0).mean()*100:.1f}%")

        plot_rolling_results(df, args.ticker,
                              save_path=os.path.join(args.outdir, f"{args.ticker}_hedge_rolling.png"))


if __name__ == "__main__":
    main()
