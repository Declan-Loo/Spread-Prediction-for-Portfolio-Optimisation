import lseg.data as ld
import atexit
import hashlib
import json
import warnings
from collections.abc import Sequence
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

load_dotenv()

# --- Paths ---
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Interval normalisation ---
_INTERVAL_MAP = {
    "daily":   "1D",
    "1d":      "1D",
    "weekly":  "1W",
    "1w":      "1W",
    "monthly": "1M",
    "1m":      "1M",
    "hourly":  "1H",
    "1h":      "1H",
    "minute":  "1MIN",
    "1min":    "1MIN",
}

def _normalise_interval(interval: str) -> str:
    return _INTERVAL_MAP.get(interval.lower(), interval)

# --- Session Management ---
_session_open = False

def open_session():
    global _session_open
    if not _session_open:
        ld.open_session()
        _session_open = True
        print("LSEG session opened.")

def close_session():
    global _session_open
    if _session_open:
        ld.close_session()
        _session_open = False
        print("LSEG session closed.")

atexit.register(close_session)

# --- Cache helpers ---
# Cache key is date-independent: (tickers, interval, fields).
# One CSV file grows over time as new date ranges are requested.

def _cache_path(tickers: list, interval: str, fields: list) -> Path:
    key = {
        "tickers":  sorted(tickers),
        "interval": interval,
        "fields":   sorted(fields),
    }
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    tickers_str = "_".join(sorted(tickers)).replace(".", "-").replace("/", "-")[:40]
    fname = f"{tickers_str}__{interval}__{digest}.csv"
    return RAW_DATA_DIR / fname

def _save_cache(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=True)

def _load_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

_FIELD_RENAME = {
    "OPEN":      "Open",
    "HIGH":      "High",
    "LOW":       "Low",
    "TRDPRC_1":  "Close",
    "ACVOL_UNS": "Volume",
    "MKT_CAP":   "Market_Cap",
}

# --- Column normalisation (applied to every fresh fetch) ---

def _normalise_columns(df: pd.DataFrame, instruments: list, fields: list) -> pd.DataFrame:
    """Flatten MultiIndex columns and rename LSEG field codes to friendly names."""
    num_tickers = len(instruments)
    num_fields = len(fields)

    if isinstance(df.columns, pd.MultiIndex):
        level_0_vals = set(df.columns.get_level_values(0))
        if level_0_vals & set(instruments):
            ticker_level, field_level = 0, 1
        else:
            ticker_level, field_level = 1, 0

        if num_tickers > 1 and num_fields == 1:
            df.columns = df.columns.get_level_values(ticker_level)
        elif num_tickers == 1 and num_fields > 1:
            df.columns = df.columns.get_level_values(field_level)
            df.rename(columns=_FIELD_RENAME, inplace=True)
        else:
            if ticker_level != 0:
                df.columns = df.columns.swaplevel(0, 1)
            df.columns = df.columns.set_levels(
                [df.columns.levels[1].map(
                    lambda f: _FIELD_RENAME.get(f, f)
                )],
                level=1,
            )
    else:
        df.rename(columns=_FIELD_RENAME, inplace=True)

    return df


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Index cleanup, type coercion, ffill, and missing-data warnings."""
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.dropna(how="all", inplace=True)
    df.sort_index(inplace=True)

    for col in df.columns if not isinstance(df.columns, pd.MultiIndex) else df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.ffill(limit=5)

    if not isinstance(df.columns, pd.MultiIndex):
        missing_pct = df.isna().mean()
        bad_cols = missing_pct[missing_pct > 0.10].index.tolist()
        if bad_cols:
            warnings.warn(
                f"Dropping columns with >10% missing data: {bad_cols}"
            )
            df.drop(columns=bad_cols, inplace=True)

    return df


def _fetch_from_lseg(instruments, fields, interval, start_dt, end_dt):
    """Call LSEG API with error handling; returns a raw DataFrame."""
    open_session()

    try:
        df = ld.get_history(
            universe=instruments,
            fields=fields,
            interval=interval,
            start=start_dt,
            end=end_dt,
        )
    except Exception as exc:
        raise RuntimeError(
            f"LSEG get_history() failed for {instruments}: {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"LSEG returned no data for {instruments} "
            f"({start_dt.date()} to {end_dt.date()})"
        )

    df = _normalise_columns(df, instruments, fields)
    df = _clean_frame(df)
    return df


# --- Core fetch function ---
def get_price_timeseries(tickers, start=None, end=None, interval="1d", fields=None, use_cache=True):
    # Resolve dates
    if end is None:
        end_dt = datetime.today()
    else:
        end_dt = end if isinstance(end, datetime) else datetime.fromisoformat(end)

    if start is None:
        start_dt = end_dt - timedelta(days=365)
    else:
        start_dt = start if isinstance(start, datetime) else datetime.fromisoformat(start)

    # Normalise tickers
    if isinstance(tickers, str):
        instruments = [tickers]
    elif isinstance(tickers, Sequence):
        instruments = list(tickers)
    else:
        raise TypeError("tickers must be a string or a sequence of strings")

    # Normalise interval
    interval = _normalise_interval(interval)

    # Default fields
    if fields is None:
        fields = ["OPEN", "HIGH", "LOW", "TRDPRC_1", "ACVOL_UNS"]

    # --- Smart incremental cache ---
    path = _cache_path(instruments, interval, fields)
    cached_df = None

    if use_cache and path.exists():
        cached_df = _load_cache(path)
        cached_start = cached_df.index.min()
        cached_end = cached_df.index.max()

        # Determine which date ranges are missing
        need_before = start_dt < cached_start
        need_after = end_dt > cached_end

        if not need_before and not need_after:
            # Full cache hit — entire requested range is covered
            print(f"[cache hit] {path.name}")
            return cached_df.loc[start_dt:end_dt]

        # Fetch only the gap(s)
        fetched_parts = []

        if need_before:
            gap_end = cached_start - timedelta(days=1)
            # Roll back to nearest preceding business day
            gap_end = pd.Timestamp(np.busday_offset(gap_end.date(), 0, roll='preceding'))
            if start_dt <= gap_end:
                print(f"[cache partial] Fetching {start_dt.date()} → {gap_end.date()}")
                try:
                    fetched_parts.append(
                        _fetch_from_lseg(instruments, fields, interval, start_dt, gap_end)
                    )
                except ValueError as e:
                    print(f"[cache partial] Skipping (no trading data): {e}")

        if need_after:
            gap_start = cached_end + timedelta(days=1)
            if gap_start <= end_dt:
                print(f"[cache partial] Fetching {gap_start.date()} → {end_dt.date()}")
                try:
                    fetched_parts.append(
                        _fetch_from_lseg(instruments, fields, interval, gap_start, end_dt)
                    )
                except ValueError as e:
                    print(f"[cache partial] Skipping (no trading data): {e}")

        if fetched_parts:
            # Merge new data with cached data
            combined = pd.concat([cached_df] + fetched_parts)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)

            _save_cache(combined, path)
            print(f"[cache updated] {path.name}")
            return combined.loc[start_dt:end_dt]

        # Edge case: gaps calculated but nothing new fetched (e.g. weekends)
        return cached_df.loc[start_dt:end_dt]

    # --- No cache exists — full fetch ---
    df = _fetch_from_lseg(instruments, fields, interval, start_dt, end_dt)

    if use_cache:
        _save_cache(df, path)
        print(f"[cache write] {path.name}")

    return df

# --- Convenience wrappers ---
def get_close_prices(tickers, start=None, end=None, interval="1d", use_cache=True):
    """For spread analysis, cointegration, and MPT — close prices only."""
    return get_price_timeseries(
        tickers, start=start, end=end, interval=interval,
        fields=["TRDPRC_1"], use_cache=use_cache
    )

def get_ohlcv(tickers, start=None, end=None, interval="1d", use_cache=True):
    """For dashboard candlestick charts and backtesting."""
    return get_price_timeseries(
        tickers, start=start, end=end, interval=interval,
        fields=["OPEN", "HIGH", "LOW", "TRDPRC_1", "ACVOL_UNS"], use_cache=use_cache
    )

def get_market_cap(tickers, start=None, end=None, interval="1d", use_cache=True):
    """Return historical market capitalisation series for one or more tickers.

    Single ticker  → Series named 'Market_Cap'.
    Multiple tickers → DataFrame with one column per ticker.

    The underlying LSEG field is MKT_CAP (local-currency market cap).
    """
    df = get_price_timeseries(
        tickers, start=start, end=end, interval=interval,
        fields=["TR.CompanyMarketCap"], use_cache=use_cache,
    )

    return df 

def get_risk_free_rate(
    start=None,
    end=None,
    interval="1d",
    use_cache=True,
    annualise=False,
):
    """
    Fetch 3M US T-bill yield from LSEG and return as a time series.

    Returns a pd.Series indexed by Date:
    - If annualise=False (default): decimal annual yield, e.g. 0.025 = 2.5%.
    - If annualise=True: constant average annual yield over the sample.
    """
    # 3M US T-bill rate (check RIC in Workspace if needed)
    tickers = ["US3MT=RR"]
    fields = ["A_YLD_1"]  # yield in percent

    df = get_price_timeseries(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        fields=fields,
        use_cache=use_cache,
    )

    # Single ticker, single field → Series
    yld_pct = df.iloc[:, 0]              # e.g. 4.25 = 4.25%
    yld_annual = yld_pct / 100.0         # 0.0425

    if annualise:
        return float(yld_annual.mean())  # e.g. 0.019 ≈ 1.9% p.a.

    yld_annual.name = "rf_annual"
    return yld_annual


def get_risk_free_daily(start=None, end=None, use_cache=True):
    """
    Daily risk-free rate (decimal per day) from 3M T-bill yield.
    """
    rf_annual = get_risk_free_rate(
        start=start,
        end=end,
        interval="1d",
        use_cache=use_cache,
        annualise=False,
    )

    rf_daily = rf_annual / 252.0
    rf_daily.name = "rf_daily"
    return rf_daily
