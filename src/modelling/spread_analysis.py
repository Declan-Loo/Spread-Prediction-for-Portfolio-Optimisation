"""
Spread characterisation for pairs trading.

Computes spread series, rolling z-scores, half-life of mean reversion,
and the Hurst exponent to assess mean-reversion quality.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """
    Spread = y - hedge_ratio * x - intercept.
    """
    spread = y - hedge_ratio * x - intercept
    spread.name = "spread"
    return spread


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score: (spread - rolling_mean) / rolling_std.

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int
        Rolling window size (trading days).

    Returns
    -------
    pd.Series of z-scores (NaN for the first `window-1` observations).
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std(ddof=1)
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    return zscore


def compute_half_life(spread: pd.Series) -> float:
    """
    Half-life of mean reversion via an AR(1) / Ornstein-Uhlenbeck model.

    Fits  Δs_t = λ · s_{t-1} + ε  via OLS.
    Half-life = -ln(2) / λ.

    Returns np.inf if the spread does not appear mean-reverting (λ >= 0).
    """
    spread_clean = spread.dropna()
    lag = spread_clean.shift(1)
    delta = spread_clean.diff()

    # Drop the first NaN row
    lag = lag.iloc[1:]
    delta = delta.iloc[1:]

    X = sm.add_constant(lag)
    model = sm.OLS(delta, X).fit()
    lam = model.params.iloc[1]

    if lam >= 0:
        return np.inf

    return float(-np.log(2) / lam)


def compute_hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Rescaled-range (R/S) Hurst exponent.

    H < 0.5 → mean-reverting
    H ≈ 0.5 → random walk
    H > 0.5 → trending

    Parameters
    ----------
    series : pd.Series
        Time series (e.g. spread).
    max_lag : int
        Maximum lag to consider.

    Returns
    -------
    float: Estimated Hurst exponent.
    """
    series_clean = series.dropna().values
    n = len(series_clean)
    max_lag = min(max_lag, n // 2)

    lags = range(2, max_lag + 1)
    tau = []

    for lag in lags:
        # Use variance of differences at each lag
        tau.append(np.std(np.subtract(series_clean[lag:], series_clean[:-lag])))

    # Fit log(tau) = H * log(lag) + c
    log_lags = np.log(list(lags))
    log_tau  = np.log(tau)
    poly = np.polyfit(log_lags, log_tau, 1)
    return float(poly[0])

def compute_rolling_half_life(spread: pd.Series, window: int = 252, cap: float = 365.0) -> pd.Series:
    """
    Rolling half-life of mean reversion using a sliding window.

    At each time step t, fits the OU model on spread[t-window:t]
    and returns the half-life estimate.

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int
        Rolling window size in trading days (default: 252 = 1 year).

    Returns
    -------
    pd.Series of rolling half-life values (NaN for first `window` observations).
    """
    half_lives = [np.nan] * window

    for w in range(window, len(spread)):
        hl = compute_half_life(spread.iloc[w - window : w])
        half_lives.append(hl)

    return pd.Series(half_lives, index=spread.index, name="rolling_half_life")

def compute_rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score of the spread using a sliding window.

    At each time step t, computes the z-score using the rolling mean
    and standard deviation over spread[t-window:t].

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int
        Rolling window size in trading days (default: 60).

    Returns
    -------
    pd.Series of rolling z-scores (NaN for first `window` observations).
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std  = spread.rolling(window=window).std(ddof=1)
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "rolling_zscore"
    return zscore

def spread_summary(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
    window: int = 60,
) -> dict:
    """
    Aggregate spread statistics.

    Returns dict with: half_life, hurst, current_zscore, spread_mean,
    spread_std, adf_stat, adf_pvalue, is_stationary.
    """
    spread = compute_spread(y, x, hedge_ratio, intercept)
    zscore = compute_zscore(spread, window=window)

    half_life = compute_half_life(spread)
    hurst = compute_hurst_exponent(spread)

    # ADF on the spread
    adf_result = adfuller(spread.dropna(), autolag="AIC")

    return {
        "half_life": half_life,
        "hurst": hurst,
        "current_zscore": float(zscore.iloc[-1]) if not np.isnan(zscore.iloc[-1]) else np.nan,
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std(ddof=1)),
        "adf_stat": adf_result[0],
        "adf_pvalue": adf_result[1],
        "is_stationary": adf_result[1] < 0.05,
    }
