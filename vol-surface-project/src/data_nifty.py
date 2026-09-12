"""
Fetch NSE index options data from the daily F&O Bhavcopy (the official
end-of-day settlement report NSE publishes for every trading day).

Why this instead of the live option-chain-indices API:
    Live scraping of nseindia.com's option-chain JSON endpoint runs into
    NSE's anti-bot layer, which blocks plain HTTP clients (including
    curl_cffi with a Chrome TLS fingerprint) hard enough that it's not
    reliably scriptable right now. The Bhavcopy, by contrast, is a static
    daily report file NSE publishes for public/regulatory consumption --
    much less aggressively protected, and arguably a more appropriate
    data source for this project anyway: it's the same
    daily/EOD-snapshot approach used in the EVT tail-risk project, and it
    gives a full multi-strike, multi-expiry chain in one file rather than
    a single live quote.

Trade-off: NSE's public daily-reports API only serves the current and
previous trading day (it's not a historical archive), so this gives you
a snapshot, not an arbitrary historical date. That's fine for building
and demoing the vol surface; if you want to backfill history later,
NSE's historical F&O UDiFF archive (nsearchives.nseindia.com) covers
dates from Jul 8, 2024 onward and could be wired in as a separate
function.

We use `jugaad_data`'s NSEDailyReports client, which knows how to query
NSE's daily-reports API for the current fileKey/filename (NSE renames
files by date, so hardcoding a URL breaks daily) and download it.
"""

import io
import zipfile
import numpy as np
import pandas as pd
from jugaad_data.nse.archives import NSEDailyReports

# Candidate fileKeys for the F&O UDiFF bhavcopy -- NSE has used a couple
# of naming variants; we try each until one resolves.
_FO_BHAVCOPY_FILEKEY_CANDIDATES = [
    "F&O-UDIFF-BHAVCOPY",
    "FO-UDIFF-BHAVCOPY-CSV",
    "F&O-UDIFF-BHAVCOPY-CSV",
]

# Column name variants across NSE's UDiFF / legacy formats, normalized
# to upper-case-no-space for matching.
_COL_ALIASES = {
    "symbol": ["TCKRSYMB", "SYMBOL"],
    "instrument": ["FININSTRMTP", "INSTRUMENT"],
    "expiry": ["XPRYDT", "EXPIRY_DT", "EXPIRYDATE"],
    "strike": ["STRKPRIC", "STRIKE_PR", "STRIKEPRICE"],
    "option_type": ["OPTNTP", "OPTION_TYP", "OPTIONTYPE"],
    "close": ["CLSPRIC", "CLOSE"],
    "settle": ["STTLMPRIC", "SETTLE_PR", "SETTLEPRICE"],
    "open_interest": ["OPNINTRST", "OPEN_INT", "OPENINTEREST"],
    "underlying": ["UNDRLYGPRIC", "UNDERLYING_VALUE"],
}


def _find_col(columns_upper, key):
    for alias in _COL_ALIASES[key]:
        if alias in columns_upper:
            return columns_upper[alias]
    return None


def _fetch_fo_bhavcopy_df() -> pd.DataFrame:
    """
    Downloads today's (or the most recent trading day's) F&O bhavcopy and
    returns it as a raw DataFrame, column names untouched.
    """
    client = NSEDailyReports()
    last_err = None

    for file_key in _FO_BHAVCOPY_FILEKEY_CANDIDATES:
        try:
            content = client.download_file(file_key, segment="FO")
        except Exception as e:
            last_err = e
            continue

        # Content may be a raw CSV or a zipped CSV -- handle both.
        if content[:2] == b"PK":  # zip file magic bytes
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                fname = zf.namelist()[0]
                with zf.open(fname) as fp:
                    return pd.read_csv(fp)
        else:
            return pd.read_csv(io.BytesIO(content))

    # None of the candidate keys worked -- show what NSE actually offers
    # so the fileKey list above can be corrected.
    try:
        reports = client.get_daily_reports(segment="FO")
        available_keys = sorted({
            f.get("fileKey") for day in ("CurrentDay", "PreviousDay")
            for f in reports.get(day, [])
        })
    except Exception:
        available_keys = ["<could not fetch report list>"]

    raise RuntimeError(
        f"Could not find F&O bhavcopy under any of {_FO_BHAVCOPY_FILEKEY_CANDIDATES}. "
        f"Available fileKeys for segment=FO right now: {available_keys}. "
        f"Update _FO_BHAVCOPY_FILEKEY_CANDIDATES with the correct one. "
        f"(Underlying error: {last_err})"
    )


def get_option_chain(symbol: str = "NIFTY", max_expiries: int = 6) -> pd.DataFrame:
    """
    symbol: 'NIFTY' or 'BANKNIFTY'
    Returns a tidy DataFrame: expiry, T, strike, option_type, mid_price,
    open_interest, underlying_spot -- filtered to index options for `symbol`.

    Note: this is an EOD snapshot (today's or the last trading day's
    settlement), not live intraday bid/ask, so there's no `bid`/`ask`/
    `volume` -- `mid_price` here is the day's settlement price.
    """
    raw = _fetch_fo_bhavcopy_df()
    cols_upper = {c.strip().upper().replace(" ", ""): c for c in raw.columns}

    sym_col = _find_col(cols_upper, "symbol")
    instr_col = _find_col(cols_upper, "instrument")
    exp_col = _find_col(cols_upper, "expiry")
    strike_col = _find_col(cols_upper, "strike")
    opt_col = _find_col(cols_upper, "option_type")
    close_col = _find_col(cols_upper, "close")
    settle_col = _find_col(cols_upper, "settle")
    oi_col = _find_col(cols_upper, "open_interest")
    under_col = _find_col(cols_upper, "underlying")

    missing = [name for name, col in
               [("symbol", sym_col), ("expiry", exp_col), ("strike", strike_col),
                ("option_type", opt_col)]
               if col is None]
    if missing:
        raise RuntimeError(
            f"Bhavcopy is missing expected columns: {missing}. "
            f"Actual columns in the file: {list(raw.columns)}. "
            f"NSE may have changed the format -- update _COL_ALIASES."
        )

    df = raw[raw[sym_col].astype(str).str.strip().str.upper() == symbol.upper()].copy()
    if instr_col is not None:
        # NSE's UDiFF FinInstrmTp codes are abbreviated, not "OPTIDX"-style:
        #   IDO = Index Options, IDF = Index Futures,
        #   STO = Stock Options,  STF = Stock Futures
        # NIFTY/BANKNIFTY are indices, so we want IDO specifically.
        df = df[df[instr_col].astype(str).str.upper() == "IDO"]

    if df.empty:
        # Show what's actually in the file so the symbol/instrument
        # matching logic above can be corrected without another
        # round-trip -- exact-match filtering on scraped data is
        # notoriously fragile (padding, suffixes, casing all vary).
        raw_sym_upper = raw[sym_col].astype(str).str.strip().str.upper()
        near_matches = sorted(raw_sym_upper[raw_sym_upper.str.contains("NIFTY", na=False)].unique())[:20]
        instr_values = (sorted(raw[instr_col].astype(str).str.upper().unique())[:20]
                        if instr_col is not None else ["<no instrument column found>"])
        raise ValueError(
            f"No rows found for symbol={symbol} after filtering. Diagnostics:\n"
            f"  Symbol column used: {sym_col!r}\n"
            f"  Instrument column used: {instr_col!r}\n"
            f"  Symbol values containing 'NIFTY' in the raw file: {near_matches}\n"
            f"  All distinct instrument-type values in the raw file: {instr_values}\n"
            f"  Total raw rows before filtering: {len(raw)}\n"
            f"Compare these against the exact-match filters in get_option_chain() "
            f"and adjust as needed (e.g. exact symbol string, or instrument code "
            f"used for index options in this file)."
        )

    df["expiry"] = pd.to_datetime(df[exp_col], errors="coerce")
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    df["T"] = ((df["expiry"] - today).dt.days.clip(lower=1)) / 365.0

    df["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    df["option_type"] = df[opt_col].astype(str).str.upper().map({"CE": "call", "PE": "put"})

    # Prefer settlement price (more representative of fair value at market
    # close) over raw close, falling back to whichever is present.
    if settle_col is not None:
        df["mid_price"] = pd.to_numeric(df[settle_col], errors="coerce")
    elif close_col is not None:
        df["mid_price"] = pd.to_numeric(df[close_col], errors="coerce")
    else:
        raise RuntimeError("Bhavcopy has neither a settlement nor close price column.")

    df["open_interest"] = pd.to_numeric(df[oi_col], errors="coerce") if oi_col else np.nan
    df["underlying_spot"] = pd.to_numeric(df[under_col], errors="coerce") if under_col else np.nan

    keep_expiries = sorted(df["expiry"].dropna().unique())[:max_expiries]
    df = df[df["expiry"].isin(keep_expiries)]

    return df[["expiry", "T", "strike", "option_type", "mid_price",
               "open_interest", "underlying_spot"]].dropna(
        subset=["strike", "option_type", "mid_price"]).reset_index(drop=True)


def get_spot(symbol: str = "NIFTY") -> float:
    chain = get_option_chain(symbol, max_expiries=1)
    spot = chain["underlying_spot"].dropna()
    if spot.empty:
        raise ValueError(
            f"Bhavcopy didn't include an underlying price column for {symbol}. "
            f"Fetch the index level separately (e.g. from yfinance's ^NSEI)."
        )
    return float(spot.iloc[0])


if __name__ == "__main__":
    chain = get_option_chain("NIFTY", max_expiries=2)
    print(chain.head(10))
    print(f"\nTotal rows: {len(chain)}")
