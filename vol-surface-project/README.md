# Options Implied Volatility Surface, Greeks & Delta-Hedging Backtest (US + Nifty)

A dual-market options analytics toolkit built on Black-Scholes-Merton:
pulls live/EOD option chains for US equities/indices and NSE Nifty/Bank
Nifty, solves for implied volatility and Greeks, fits an industry-standard
parametric smile (SVI), and backtests delta-hedging P&L against real
historical price paths.

Companion to the [EVT tail-risk project](../evt-tail-risk) -- same
dual-market (S&P/Nifty) framing, this time looking at the market's own
forward-looking risk pricing (implied vol) rather than backward-looking
tail statistics.

## Features

- **IV surface + Greeks** -- Black-Scholes IV solver (Brent's method) and
  full Greeks (delta, gamma, vega, theta, rho) across the whole chain,
  for both US tickers (live yfinance chain) and Nifty/Bank Nifty (NSE's
  daily F&O Bhavcopy)
- **SVI parametric smile fit** -- fits Gatheral's SVI model per expiry,
  giving a smooth, arbitrage-checked curve you can evaluate at any
  strike, not just the ones listed
- **Delta-hedging backtest** -- simulates writing/buying an option and
  delta-hedging it daily over a real historical price path, demonstrating
  how hedging P&L is driven by the gap between assumed and realized
  volatility
- **Liquidity-aware data pipeline** -- a relative, per-expiry open-interest
  filter that auto-adapts across wildly different liquidity scales (SPY's
  thousands of contracts vs Nifty's millions), rather than a fragile
  hardcoded threshold

## Project structure

```
vol-surface-project/
├── src/
│   ├── black_scholes.py       # BS pricer, IV solver, Greeks -- pure math, no I/O
│   ├── data_us.py             # yfinance option chain + price history fetcher
│   ├── data_nifty.py          # NSE F&O Bhavcopy fetcher
│   ├── surface.py             # chain -> liquidity-filtered IV/Greeks surface, plotting
│   ├── svi_fit.py             # SVI parametric smile fit
│   ├── hedge_backtest.py      # delta-hedging simulation core
│   ├── run_us.py              # CLI: US ticker -> IV surface
│   ├── run_nifty.py           # CLI: Nifty/Bank Nifty -> IV surface
│   ├── run_svi_fit.py         # CLI: SVI fit for either market
│   └── run_hedge_backtest.py  # CLI: delta-hedging backtest
├── output/                    # CSVs + PNGs land here
└── requirements.txt
```

## Installation

```bash
git clone <this-repo>
cd vol-surface-project
pip install -r requirements.txt
cd src
```

## Usage

**IV surface & Greeks**
```bash
python run_us.py SPY --expiries 8
python run_nifty.py NIFTY --expiries 6
```
Writes a CSV of the full surface plus two plots (3D surface, 2D
smile-by-expiry) to `output/`.

**SVI parametric fit**
```bash
python run_svi_fit.py us SPY --expiries 8
python run_svi_fit.py nifty NIFTY --expiries 6
```
Fits one SVI curve per expiry and plots it against the raw market IV
points, so fit quality can be checked visually, not just by an RMSE
number.

**Delta-hedging backtest**
```bash
# Single window -- most recent 30 trading days, full equity-curve plot
python run_hedge_backtest.py SPY --mode single --window-days 30

# Rolling across ~2 years -- P&L vs vol-forecast-error scatter
python run_hedge_backtest.py SPY --mode rolling --window-days 20 --vol-lookback-days 20
```

## Results

Across ~2 years of rolling 20-day windows on SPY, delta-hedging P&L
correlates strongly (r ≈ 0.72) with the gap between the volatility used
to hedge and what actually got realized -- direct empirical confirmation
that a delta-hedged option seller's P&L is compensation for volatility
risk, not free money. Mean vol-forecast-error was ~0 (consistent with
efficient pricing on average), with the expected asymmetric P&L shape:
frequent small gains, occasional large losses.

The SVI fits track the market smile within 0.1-1 vol point of RMSE for
every expiry except the shortest-dated one in each market (SPY and
Nifty both show ~2-4 points there) -- a real, consistent property of
near-expiry smiles being harder for a smooth 5-parameter curve to
capture, not a data or fitting artifact.

## Engineering notes & known limitations

Every component below was validated against synthetic data with a known
ground truth (GBM price paths, options priced from known volatilities)
before being trusted on real market data -- several real bugs were only
caught this way, not by inspecting summary statistics alone.

- **Liquidity filtering matters more than moneyness bounds for smile
  quality.** Deep ITM/OTM strikes have very low vega, so a few cents of
  ordinary bid-ask noise (or a stale settlement print) translates into a
  large, spurious IV swing -- the cause of sharp sawtooth spikes in a raw
  smile plot. An absolute open-interest floor isn't enough on its own,
  though: SPY's genuinely liquid strikes sit in the thousands, while
  Nifty's run in the hundreds of thousands to millions. The filter uses
  `min_oi_percentile` (default 50th) -- strikes at or above that
  percentile of open interest *within their own expiry* -- so it
  auto-adapts to whatever scale the underlying trades at. Validated
  against real noisy data on both markets: cut a 32-point IV zigzag down
  to near-monotonic on SPY, and cut sign-reversals from 32 to 4 on Nifty
  while keeping 50+ strikes per expiry.
- **Even a liquid, tight-spread quote can give a nonsense IV.** SPY
  options are American-style on a dividend-paying underlying; a deep ITM
  call near expiry can trade only a few cents above intrinsic value
  (real market behavior), which European Black-Scholes can't capture
  correctly -- vega there is nearly zero, so matching that price forces
  the solver to an implausible IV. Found this in practice: a SPY 585
  call, OI=809, a tight 2% spread, still solved to a 74% IV with nothing
  to do with the rest of the smile. Fixed with a tighter default
  moneyness band (0.85-1.15) plus a `max_iv` sanity cap (150%) as a
  backstop.
- **SVI's individual parameters aren't uniquely identified.** Several
  different (a, b, rho, m, sigma) combinations can produce nearly
  identical curves -- compare the fitted curve (or derived quantities
  like ATM vol/skew) across markets, not raw parameter values directly.
  A real bug surfaced here during development: the initial guess for `b`
  needs to scale with the actual magnitude of total variance in the data
  (which itself scales with T and IV²) -- a fixed constant guess caused
  the optimizer to collapse to a flat line on 3 of 4 real SPY expiries,
  something the RMSE number alone didn't reveal but plotting the fit
  against market data immediately did.
- **Anti-bot / TLS fingerprinting.** Yahoo Finance blocks plain
  `requests`/`urllib3` traffic at the TLS-fingerprint level on some
  networks. `data_us.py` uses `curl_cffi` (`Session(impersonate="chrome")`)
  to work around this, and avoids yfinance's heavyweight `fast_info`/`info`
  calls (which secretly pull a full year of price history) in favor of a
  cheap 5-day history call.
- **NSE data source.** Live option-chain scraping (`option-chain-indices`)
  is reliably blocked by NSE's anti-bot layer regardless of headers or
  TLS fingerprinting. `data_nifty.py` instead pulls the **daily F&O
  Bhavcopy** (NSE's official EOD settlement report) via `jugaad-data`'s
  `NSEDailyReports` client. Trade-off: only the current/previous trading
  day is available (not arbitrary history), and it's settlement price
  rather than live bid/ask -- fine for a snapshot surface, but Nifty's
  smile will always retain a bit more residual noise than SPY's, since
  there's no live spread to filter on, only open interest.
- **Risk-free rate.** US uses `^IRX` (13-week T-bill) as a live proxy.
  India has no equivalently convenient free live quote, so `run_nifty.py`
  uses a fixed rate proxy (override with `--rate`).
- **Dividend yield.** Assumed 0 for US single names/ETFs, 1% for Nifty
  (rough index estimate). Both adjustable.

## Possible extensions

- Term-structure-of-ATM-vol chart
- Cross-market comparison of fitted SVI curves (SPY vs Nifty skew/curvature)
- Compare Yahoo's own quoted IV (`yahoo_iv` column) against the
  self-solved IV as a data-quality check
