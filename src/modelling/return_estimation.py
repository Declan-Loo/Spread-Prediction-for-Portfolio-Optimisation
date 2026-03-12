"""
Return estimation for spread-based and traditional MPT portfolios.

Provides expected-return estimators (historical mean, EWMA, OU-implied)
and covariance estimators (sample, Ledoit-Wolf shrinkage) for use in
Markowitz mean-variance optimisation across spread and asset return streams.
"""

import numpy as np
import pandas as pd

from .spread_analysis import compute_spread, compute_half_life


# ---------------------------------------------------------------------------
# Spread return computation
# ---------------------------------------------------------------------------


def compute_spread_returns(
    y_prices: pd.Series,
    x_prices: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """
    Daily return of a dollar-neutral spread position: r_y - β · r_x.

    Consistent with the PnL definition used in the backtesting engine
    and benchmark modules.

    Parameters
    ----------
    y_prices, x_prices : pd.Series
        Close prices of the dependent (y) and independent (x) legs.
    hedge_ratio : float
        Cointegration hedge ratio (β).

    Returns
    -------
    pd.Series
        Daily spread returns.
    """
    y_ret = y_prices.pct_change()
    x_ret = x_prices.pct_change()
    spread_ret = y_ret - hedge_ratio * x_ret
    spread_ret.name = "spread_return"
    return spread_ret


def build_spread_return_matrix(
    prices_df: pd.DataFrame,
    coint_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a DataFrame of daily spread returns for all cointegrated pairs.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Close prices; columns = ticker RICs.
    coint_pairs : pd.DataFrame
        Cointegrated pairs with columns [y, x, hedge_ratio].

    Returns
    -------
    pd.DataFrame
        Each column is the spread return for one pair, named ``y_vs_x``.
    """
    spread_rets = {}
    for _, row in coint_pairs.iterrows():
        label = f"{row['y']}_vs_{row['x']}"
        spread_rets[label] = compute_spread_returns(
            prices_df[row["y"]],
            prices_df[row["x"]],
            row["hedge_ratio"],
        )
    return pd.DataFrame(spread_rets).dropna(how="all")


# ---------------------------------------------------------------------------
# Expected return estimators
# ---------------------------------------------------------------------------


def historical_mean_return(
    returns: pd.DataFrame,
    window: int | None = None,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Historical mean return estimator.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return series (columns = assets or spreads).
    window : int | None
        If provided, use the last ``window`` observations only.
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Expected (annualised) return per column.
    """
    sample = returns.tail(window) if window is not None and window > 0 else returns
    mu = sample.mean()
    if annualise:
        return mu * periods_per_year
    return mu


def ewma_return(
    returns: pd.DataFrame,
    span: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Exponentially weighted moving average return estimator.

    Gives more weight to recent observations, adapting faster to
    regime changes than the equal-weighted historical mean.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return series.
    span : int
        EWMA span parameter (decay half-life ≈ span · ln(2)).
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int

    Returns
    -------
    pd.Series
        EWMA-estimated expected return per column.
    """
    mu = returns.ewm(span=span).mean().iloc[-1]
    if annualise:
        return mu * periods_per_year
    return mu


def ou_expected_return(
    spread: pd.Series,
    half_life: float,
    window: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> float:
    """
    Ornstein-Uhlenbeck implied expected return for a mean-reverting spread.

    Uses the current deviation from the rolling mean and the estimated
    speed of mean reversion (half-life) to predict the expected daily
    change in the spread level.

    Model:  dS = κ(θ − S) dt + σ dW
    Expected one-step change:  E[ΔS] = (θ − S_t)(1 − exp(−κ))
    where κ = ln(2) / half_life.

    The return is normalised by the absolute spread level: E[ΔS] / |S_t|.

    Parameters
    ----------
    spread : pd.Series
        Spread level time series (y − β·x − α).
    half_life : float
        Estimated half-life of mean reversion (trading days).
    window : int
        Rolling window for long-run mean (θ) estimate.
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int

    Returns
    -------
    float
        OU-implied expected return for the spread.
    """
    if half_life <= 0 or np.isinf(half_life):
        return 0.0

    kappa = np.log(2) / half_life
    theta = spread.rolling(window=window).mean().iloc[-1]
    s_t = spread.iloc[-1]

    if np.isnan(theta) or np.isnan(s_t) or s_t == 0:
        return 0.0

    expected_delta = (theta - s_t) * (1 - np.exp(-kappa))
    daily_return = expected_delta / abs(s_t)

    if annualise:
        return float(daily_return * periods_per_year)
    return float(daily_return)


def build_ou_expected_returns(
    prices_df: pd.DataFrame,
    coint_pairs: pd.DataFrame,
    window: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    OU-implied expected returns for all cointegrated pairs.

    For each pair, computes the spread, estimates its half-life, and
    derives the OU-implied return from the current deviation.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Close prices; columns = ticker RICs.
    coint_pairs : pd.DataFrame
        Cointegrated pairs with columns [y, x, hedge_ratio, intercept].
    window : int
        Rolling window for long-run mean estimate.
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int

    Returns
    -------
    pd.Series
        OU-implied expected return per pair, indexed by ``y_vs_x``.
    """
    results = {}
    for _, row in coint_pairs.iterrows():
        label = f"{row['y']}_vs_{row['x']}"
        spread = compute_spread(
            prices_df[row["y"]],
            prices_df[row["x"]],
            row["hedge_ratio"],
            row.get("intercept", 0.0),
        )
        hl = compute_half_life(spread)
        results[label] = ou_expected_return(
            spread, hl,
            window=window,
            annualise=annualise,
            periods_per_year=periods_per_year,
        )

    return pd.Series(results, name="ou_expected_return")


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------


def sample_covariance(
    returns: pd.DataFrame,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Sample covariance matrix of daily returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return series.
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int

    Returns
    -------
    pd.DataFrame
        Covariance matrix (columns and index = asset / spread names).
    """
    cov = returns.cov()
    if annualise:
        cov = cov * periods_per_year
    return cov


def shrinkage_covariance(
    returns: pd.DataFrame,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Shrinks the sample covariance toward a scaled identity matrix,
    reducing estimation error when the number of assets is large
    relative to the sample size.  Implements the analytical solution
    from Ledoit & Wolf (2004).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return series.
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int

    Returns
    -------
    pd.DataFrame
        Shrinkage covariance matrix.
    """
    clean = returns.dropna()
    X = clean.values.copy().astype(float)
    T, p = X.shape

    # Demean
    X -= X.mean(axis=0)

    # Sample covariance (1/T normalisation for the LW formula)
    S = (X.T @ X) / T

    # Shrinkage target: scaled identity  F = (tr(S)/p) · I
    mu = np.trace(S) / p
    F = mu * np.eye(p)

    # Squared Frobenius distance between sample and target
    delta = np.sum((S - F) ** 2)

    # Estimate of β (numerator of shrinkage intensity)
    X2 = X ** 2
    phi = np.sum(X2.T @ X2) / T - np.sum(S ** 2)
    beta = phi / T

    # Optimal shrinkage intensity  α* = β / δ, clipped to [0, 1]
    if delta > 0:
        alpha = max(0.0, min(1.0, beta / delta))
    else:
        alpha = 1.0

    cov = alpha * F + (1 - alpha) * S

    result = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    if annualise:
        result *= periods_per_year
    return result


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


def spread_vs_traditional_estimates(
    prices_df: pd.DataFrame,
    coint_pairs: pd.DataFrame,
    method: str = "historical",
    window: int | None = None,
    span: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> dict:
    """
    Side-by-side return and covariance estimates for spread-based and
    traditional (asset-level) Markowitz MPT.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Close prices; columns = ticker RICs.
    coint_pairs : pd.DataFrame
        Cointegrated pairs with columns [y, x, hedge_ratio, intercept].
    method : str
        Return estimator: ``"historical"``, ``"ewma"``, or ``"ou"``
        (OU-implied, spread side only; asset side falls back to historical).
    window : int | None
        Lookback for historical mean (ignored when ``method="ewma"``).
    span : int
        EWMA span (ignored when ``method="historical"``).
    annualise : bool
    periods_per_year : int

    Returns
    -------
    dict with keys
        spread_returns  : pd.DataFrame – daily spread returns
        spread_mu       : pd.Series    – expected returns (spreads)
        spread_cov      : pd.DataFrame – covariance of spread returns
        asset_returns   : pd.DataFrame – daily asset returns
        asset_mu        : pd.Series    – expected returns (assets)
        asset_cov       : pd.DataFrame – covariance of asset returns
    """
    # --- Spread-based ---
    spread_rets = build_spread_return_matrix(prices_df, coint_pairs)

    # --- Traditional (asset-level) ---
    tickers = list(dict.fromkeys(
        coint_pairs["y"].tolist() + coint_pairs["x"].tolist()
    ))
    asset_rets = prices_df[tickers].pct_change().dropna(how="all")

    # --- Expected returns ---
    if method == "ewma":
        spread_mu = ewma_return(
            spread_rets, span=span,
            annualise=annualise, periods_per_year=periods_per_year,
        )
        asset_mu = ewma_return(
            asset_rets, span=span,
            annualise=annualise, periods_per_year=periods_per_year,
        )
    elif method == "ou":
        spread_mu = build_ou_expected_returns(
            prices_df, coint_pairs,
            window=window or 60,
            annualise=annualise, periods_per_year=periods_per_year,
        )
        asset_mu = historical_mean_return(
            asset_rets, window=window,
            annualise=annualise, periods_per_year=periods_per_year,
        )
    else:  # "historical"
        spread_mu = historical_mean_return(
            spread_rets, window=window,
            annualise=annualise, periods_per_year=periods_per_year,
        )
        asset_mu = historical_mean_return(
            asset_rets, window=window,
            annualise=annualise, periods_per_year=periods_per_year,
        )

    # --- Covariance ---
    spread_cov = sample_covariance(
        spread_rets.dropna(), annualise=annualise,
        periods_per_year=periods_per_year,
    )
    asset_cov = sample_covariance(
        asset_rets.dropna(), annualise=annualise,
        periods_per_year=periods_per_year,
    )

    return {
        "spread_returns": spread_rets,
        "spread_mu": spread_mu,
        "spread_cov": spread_cov,
        "asset_returns": asset_rets,
        "asset_mu": asset_mu,
        "asset_cov": asset_cov,
    }
