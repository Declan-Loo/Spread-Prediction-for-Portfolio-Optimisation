"""
S&P 500 index data via yfinance.

Use this for benchmark series when Refinitiv S&P 500 is not available or
to avoid LSEG entitlement. Fetches close prices for the requested date range.
"""

from datetime import datetime, timedelta

import pandas as pd

# Yahoo Finance ticker for S&P 500 index
SP500_YF_TICKER = "^GSPC"


def get_sp500_prices(start: str, end: str) -> pd.Series:
    """
    Fetch S&P 500 daily close prices for the given date range using yfinance.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD), inclusive.

    Returns
    -------
    pd.Series
        Close prices with DatetimeIndex. Empty Series if download fails or
        no data in range.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.Series(dtype=float)

    # yfinance end is exclusive; use day after to include end date
    if isinstance(end, str):
        end_dt = datetime.fromisoformat(end)
    else:
        end_dt = datetime.combine(end, datetime.min.time()) if hasattr(end, "year") else end
    end_exclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = str(start) if hasattr(start, "strftime") else start

    try:
        obj = yf.Ticker(SP500_YF_TICKER)
        hist = obj.history(start=start_str, end=end_exclusive, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)

    if hist is None or hist.empty:
        return pd.Series(dtype=float)

    # Prefer Close; fallback to Adj Close
    if "Close" in hist.columns:
        close = hist["Close"].copy()
    elif "Adj Close" in hist.columns:
        close = hist["Adj Close"].copy()
    else:
        return pd.Series(dtype=float)

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().dropna()
    close.name = "SP500"
    return close
