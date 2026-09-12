# Options Implied Volatility Surface & Greeks (US + Nifty)

Dual-market implied volatility surface and Greeks calculator, built on
Black-Scholes-Merton, pulling live option chains for US equities/indices
(via yfinance) and NSE Nifty/Bank Nifty options from NSE's daily F&O
Bhavcopy (official EOD settlement report).

Companion to the [EVT tail-risk project](../evt-tail-risk) -- same
dual-market (S&P/Nifty) framing, this time looking at the market's own
forward-looking risk pricing (implied vol) rather than backward-looking
tail statistics.

## What it does

1. Pulls a US option chain (live, strikes x expiries) or a Nifty/Bank
   Nifty option chain (daily F&O Bhavcopy snapshot) for the requested
   symbol
2. Computes an option price to feed the IV solver: mid of bid/ask for
   US (falling back to last trade price), settlement price for Nifty
3. Solves for Black-Scholes implied volatility per strike/expiry via
   Brent's method
4. Computes Greeks (delta, gamma, vega, theta, rho) at the solved IV
5. Plots the full IV surface (3D) and volatility smile per expiry (2D)

## Project structure

```
vol-surface-project/
├── src/
│   ├── black_scholes.py   # BS pricer, IV solver, Greeks -- pure math, no I/O
│   ├── data_us.py         # yfinance option chain fetcher
│   ├── data_nifty.py      # NSE option chain fetcher
│   ├── surface.py         # chain -> IV/Greeks surface, plotting
│   ├── run_us.py          # CLI driver for US tickers
│   ├── run_nifty.py       # CLI driver for Nifty/Bank Nifty
│   ├── hedge_backtest.py  # delta-hedging simulation core
│   ├── run_hedge_backtest.py  # CLI driver for the hedging backtest
│   ├── svi_fit.py         # SVI parametric smile fit
│   └── run_svi_fit.py     # CLI driver for the SVI fit
├── output/                 # CSVs + PNGs land here
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

cd src
python run_us.py SPY --expiries 8
python run_nifty.py NIFTY --expiries 6
```

Each run writes a CSV of the full surface plus two plots (3D surface,
2D smile-by-expiry) to `output/`.

## Notes / known limitations

- **Liquidity filtering matters more than moneyness bounds for smile
  quality.** Deep ITM/OTM strikes have very low vega, so a few cents of
  ordinary bid-ask noise (or a stale settlement print) translates into a
  large, spurious swing in the solved IV -- this is what causes sharp
  sawtooth spikes in a smile plot, not real market information.
  An ABSOLUTE open-interest floor isn't enough on its own, though: SPY's
  genuinely liquid strikes sit in the thousands-to-tens-of-thousands,
  while NIFTY's run in the hundreds-of-thousands to millions -- a
  completely different scale. So the primary filter is `min_oi_percentile`
  (default 50th), which keeps only strikes at or above that percentile of
  open interest WITHIN THEIR OWN EXPIRY, auto-adapting to whatever scale
  the underlying trades at, without a hardcoded absolute number. Validated
  against real data on both markets: for a noisy SPY expiry, this cut a
  32-point zigzag down to near-monotonic; for a noisy NIFTY expiry, it cut
  32 sign-reversals in the IV curve down to 4, while still keeping 50+
  strikes for that expiry.
- **Even a liquid, tight-spread quote can still give a nonsense IV.**
  SPY options are American-style on a dividend-paying underlying; a deep
  ITM call near expiry can trade only a few cents above intrinsic value
  (real market behavior, not a bad quote), which our European
  Black-Scholes model doesn't capture correctly -- vega there is nearly
  zero, so matching that price forces the solver to an implausible IV.
  Found this in practice: a SPY 585 call, OI=809, a tight 2% spread,
  still solved to a 74% IV that had nothing to do with the rest of the
  smile. Fixed with a tighter default moneyness band (0.85-1.15, down
  from 0.7-1.3) plus a `max_iv` sanity cap (default 150%) as a backstop.
- NIFTY's residual smile noise won't fully disappear the way SPY's does,
  and that's an inherent data-source limitation, not something more
  filtering fixes: it's built from the daily settlement Bhavcopy, which
  has no live bid/ask spread to filter on the way the US live chain does
  -- open interest is the only quality signal available. Deep ITM/OTM
  strikes are filtered by moneyness as a first pass, on top of the
  liquidity filter and IV cap above.
- **Risk-free rate**: US uses `^IRX` (13-week T-bill) as a live proxy.
  India has no equivalently convenient free live quote, so `run_nifty.py`
  uses a fixed rate proxy (override with `--rate`) -- update it
  periodically or wire up a live source if you extend this.
- **Dividend yield**: assumed 0 for US single names/ETFs, 1% for Nifty
  (rough index dividend yield estimate). Both are adjustable.
- **Anti-bot / TLS fingerprinting (US side)**: Yahoo Finance blocks plain
  `requests`/`urllib3` traffic at the TLS-fingerprint level on some
  networks. `data_us.py` uses `curl_cffi` (`Session(impersonate="chrome")`)
  to present a real Chrome TLS fingerprint instead. It also avoids the
  heavyweight `fast_info`/`info` calls (which secretly pull a full year of
  price history) in favor of a cheap 5-day history call, and retries with
  exponential backoff on rate limits.
- **NSE data source (India side)**: the live `option-chain-indices`
  scraping route (what `data_nifty.py` originally used) turned out to be
  reliably blocked by NSE's anti-bot layer -- not just rate-limited, but
  served a generic 404 block page regardless of headers or TLS
  fingerprinting. Rather than keep fighting that, `data_nifty.py` now
  pulls the **daily F&O Bhavcopy** (NSE's official EOD settlement report)
  via `jugaad-data`'s `NSEDailyReports` client, which is a static report
  file and far less aggressively protected. Trade-off: NSE's daily-reports
  API only serves the current/previous trading day (not arbitrary
  history), and gives settlement price rather than live bid/ask -- fine
  for a snapshot surface, but if you want to backfill history later
  you'd need NSE's historical UDiFF archive instead.
- If NSE renames the F&O bhavcopy's `fileKey` again (they've done this
  before), `data_nifty.py` will raise an error listing every fileKey
  currently available for segment=FO -- add the correct one to
  `_FO_BHAVCOPY_FILEKEY_CANDIDATES`.

- **Delta-hedging backtest** (`hedge_backtest.py`, `run_hedge_backtest.py`):
  simulates writing (or buying) a BS-priced option and delta-hedging it
  daily over a REAL historical price path, using a volatility estimated
  from an earlier, non-overlapping lookback window (no look-ahead bias).
  Demonstrates the core result that a delta-hedged seller's P&L is driven
  by the gap between the vol they hedged with and what actually got
  realized. Validated against synthetic GBM paths with known volatility
  before being pointed at real data -- hedging with vol above/below the
  true realized vol produces P&L of the correct sign and roughly the
  expected magnitude.

  ```bash
  # Single window, most recent 30 trading days, with an equity-curve plot
  python run_hedge_backtest.py SPY --mode single --window-days 30

  # Rolling across ~2 years of history -- scatter of P&L vs vol forecast error
  python run_hedge_backtest.py SPY --mode rolling --window-days 20 --vol-lookback-days 20
  ```

- **SVI parametric smile fit** (`svi_fit.py`, `run_svi_fit.py`): fits the
  industry-standard SVI model (Gatheral 2004) to each expiry, giving a
  smooth curve you can evaluate at any strike (interpolation for free)
  and a compact 5-parameter description of the smile's shape.
  Validated against synthetic data with known parameters (recovered
  curve within ~1 vol point) before trusting it on real data. Two things
  worth knowing if you extend this:
    - SVI's individual parameters aren't uniquely identified in
      isolation -- several different (a,b,rho,m,sigma) combinations can
      produce nearly the same curve. Compare the FITTED CURVE (or
      derived quantities like ATM vol/skew) across markets, not the raw
      parameter values directly.
    - Fit quality consistently degrades for the shortest-dated expiry in
      both SPY and NIFTY data -- their sharper, more localized curvature
      near expiry is harder for a smooth 5-parameter model to capture.
      This showed up as ~2-4 vol points of RMSE on the nearest expiry
      vs ~0.1-1 point on everything further out, in both markets. Also
      caught and fixed a real bug during development: the initial guess
      for `b` needs to scale with the actual magnitude of total variance
      in the data (which itself scales with T and IV^2) -- a fixed
      constant initial guess caused the optimizer to collapse to a flat
      line for several real expiries before this was caught by actually
      plotting the fit against the market data, not just checking the
      RMSE number.

  ```bash
  python run_svi_fit.py us SPY --expiries 8
  python run_svi_fit.py nifty NIFTY --expiries 6
  ```

## Possible extensions

- Term-structure-of-ATM-vol chart
- Compare Yahoo's own quoted IV (`yahoo_iv` column) against the
  self-solved IV as a data-quality check
