"""
Fetch US option chains via yfinance.

yfinance gives you, per expiry, a chain with columns including:
    strike, lastPrice, bid, ask, impliedVolatility (Yahoo's own IV, we
    recompute ours from mid price so the two can be compared), openInterest,
    volume, inTheMoney

We use mid price = (bid + ask) / 2 when both are available and > 0,
falling back to lastPrice otherwise (illiquid strikes often have stale
lastPrice but a valid bid/ask).
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

try:
    from curl_cffi import requests as cf_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

import yfinance as yf


def _make_session():
    """
    Plain `requests`/urllib3 traffic gets blocked by Yahoo's anti-bot layer
    at the TLS-fingerprint level on some networks -- this shows up as
    YFRateLimitError even on the very first call. curl_cffi impersonates a
    real Chrome TLS fingerprint, which yfinance officially supports passing
    in as a session. Falls back to a plain requests.Session (old default
    behavior) if curl_cffi isn't installed.
    """
    if _HAS_CURL_CFFI:
        return cf_requests.Session(impersonate="chrome")
    return None  # yfinance builds its own default session


_SESSION = _make_session()


def _with_retry(fn, retries=4, base_delay=5):
    """
    Retry with exponential backoff on rate limiting / transient errors.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            delay = base_delay * (2 ** attempt)
            print(f"  Rate limited, retrying in {delay}s... "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(delay)
    raise last_err


def _ticker(symbol: str):
    if _SESSION is not None:
        return yf.Ticker(symbol, session=_SESSION)
    return yf.Ticker(symbol)


def get_spot(ticker: str) -> float:
    """
    Lightweight last-price fetch. Deliberately avoids `fast_info`/`info`,
    which internally pull a full 1y price history under the hood and are
    the main thing that trips yfinance's rate limiter -- a plain 5-day
    `history()` call is much cheaper.
    """
    def _fetch():
        hist = _ticker(ticker).history(period="5d")
        if hist.empty:
            raise ValueError(f"No price history returned for {ticker}")
        return float(hist["Close"].iloc[-1])

    return _with_retry(_fetch)


def get_risk_free_rate() -> float:
    """
    Rough proxy for US risk-free rate: 13-week T-bill yield (^IRX),
    quoted in percent, so divide by 100.
    """
    def _fetch():
        hist = _ticker("^IRX").history(period="5d")
        return float(hist["Close"].iloc[-1]) / 100.0

    try:
        return _with_retry(_fetch, retries=2)
    except Exception:
        return 0.05  # fallback if ^IRX fetch fails after retries


def get_price_history(ticker: str, period: str = "1y") -> pd.Series:
    """
    Daily close prices, indexed by date. Used for realized-vol estimation
    and delta-hedging backtests -- separate from get_spot() because those
    only need the single latest price, not the full series.
    """
    def _fetch():
        hist = _ticker(ticker).history(period=period)
        if hist.empty:
            raise ValueError(f"No price history returned for {ticker}")
        return hist["Close"]

    return _with_retry(_fetch)


def get_option_chain(ticker: str, max_expiries: int = 8) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with one row per (expiry, strike, option_type):
        expiry, T (years), strike, option_type, mid_price, bid, ask,
        volume, open_interest, yahoo_iv
    """
    t = _ticker(ticker)
    expiries = _with_retry(lambda: t.options)[:max_expiries]
    if not expiries:
        raise ValueError(f"No listed options found for {ticker}")

    today = datetime.now(timezone.utc).date()
    rows = []

    for exp in expiries:
        chain = _with_retry(lambda: t.option_chain(exp))
        time.sleep(1.5)  # spread requests out to avoid re-triggering the limiter
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        T = max((exp_date - today).days, 1) / 365.0

        for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
            for _, r in df.iterrows():
                bid, ask = r.get("bid", np.nan), r.get("ask", np.nan)
                if bid and ask and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                else:
                    mid = r.get("lastPrice", np.nan)

                rows.append({
                    "expiry": exp,
                    "T": T,
                    "strike": r["strike"],
                    "option_type": opt_type,
                    "mid_price": mid,
                    "bid": bid,
                    "ask": ask,
                    "volume": r.get("volume", np.nan),
                    "open_interest": r.get("openInterest", np.nan),
                    "yahoo_iv": r.get("impliedVolatility", np.nan),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    spot = get_spot("SPY")
    rfr = get_risk_free_rate()
    print(f"SPY spot: {spot:.2f}, risk-free rate proxy: {rfr:.4f}")
    chain = get_option_chain("SPY", max_expiries=3)
    print(chain.head(10))
    print(f"\nTotal rows: {len(chain)}")
