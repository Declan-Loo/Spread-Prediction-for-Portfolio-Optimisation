"""
Return estimation for spread-based and traditional MPT portfolios.

Provides expected-return estimators (historical mean, EWMA, OU-implied)
and covariance estimators (sample, Ledoit-Wolf shrinkage) for use in
Markowitz mean-variance optimisation across spread and asset return streams.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

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
    Daily log-return of a spread position: log(y_t/y_{t-1}) − β·log(x_t/x_{t-1}).

    Using log-returns (rather than arithmetic pct_change) ensures consistency
    with the log-price spread used in cointegration estimation and the OU model:
    if S_t = log(y_t) − β·log(x_t) − α, then ΔS_t = r_y^log − β·r_x^log exactly,
    so E[ΔS] from the OU model is directly in the same units as these returns.

    Parameters
    ----------
    y_prices, x_prices : pd.Series
        Close prices of the dependent (y) and independent (x) legs.
    hedge_ratio : float
        Cointegration hedge ratio (β).

    Returns
    -------
    pd.Series
        Daily log-return spread.
    """
    y_ret = np.log(y_prices).diff()
    x_ret = np.log(x_prices).diff()
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
    sample = sample.dropna()
    mu = sample.mean()
    if annualise:
        return mu * periods_per_year
    return mu


def ewma_mean_return(
    returns: pd.DataFrame,
    span: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Exponentially weighted moving average return estimator.

    Gives more weight to recent observations, adapting faster to
    regime changes than the equal-weighted historical mean.

    Note: ewm().mean() produces a full time series; .iloc[-1] takes the
    most recent row, which is the current EWMA estimate. This is
    intentional — it is a smoothed estimate, not a forward expectation.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return series.
    span : int
        EWMA span parameter (decay half-life ≈ (span − 1) / 2 days).
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int

    Returns
    -------
    pd.Series
        EWMA-estimated expected return per column.
    """
    mu = returns.dropna().ewm(span=span).mean().iloc[-1]
    if annualise:
        return mu * periods_per_year
    return mu


def ou_implied_spread_return(
    spread: pd.Series,
    half_life: float,
    window: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
    normalisation: str = "direct",
) -> float:
    """
    Ornstein-Uhlenbeck implied expected return for a mean-reverting spread.

    Uses the current deviation from the rolling mean and the estimated
    speed of mean reversion (half-life) to predict the expected daily
    change in the spread level.

    Model:  dS = κ(θ − S) dt + σ dW
    Expected one-step change:  E[ΔS] = (θ − S_t)(1 − exp(−κ))
    where κ = ln(2) / half_life.

    When called via ``build_ou_implied_returns``, the spread is built from
    log prices, so ΔS_t = Δlog(y) − β·Δlog(x) exactly.  E[ΔS] is therefore
    a log-return in the same units as ``compute_spread_returns``, and no
    further normalisation is needed.

    Normalisation options (``"direct"`` is the correct default):
    - ``"direct"``: daily_return = E[ΔS]  — true log-return units ✓
    - ``"std"``:    daily_return = E[ΔS] / σ_rolling  — dimensionless
      signal strength, not a return; annualised values >> 100%.
    - ``"level"``:  daily_return = E[ΔS] / |S_t|  — diverges at zero-
      crossings. Retained for reference only.

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
    normalisation : {"direct", "std", "level"}
        See above.

    Returns
    -------
    float
        OU-implied expected return for the spread.
    """
    if half_life <= 0 or np.isinf(half_life):
        return 0.0

    kappa = np.log(2) / half_life
    rolling = spread.rolling(window=window)
    theta = rolling.mean().iloc[-1]
    s_t   = spread.iloc[-1]

    if np.isnan(theta) or np.isnan(s_t):
        return 0.0

    expected_delta = (theta - s_t) * (1 - np.exp(-kappa))

    if normalisation == "level":
        if s_t == 0:
            return 0.0
        daily_return = expected_delta / abs(s_t)
    elif normalisation == "std":
        sigma = rolling.std().iloc[-1]
        if np.isnan(sigma) or sigma <= 0:
            return 0.0
        daily_return = expected_delta / sigma
    else:  # "direct"
        daily_return = expected_delta

    if annualise:
        return float(daily_return * periods_per_year)
    return float(daily_return)


def build_ou_implied_returns(
    prices_df: pd.DataFrame,
    coint_pairs: pd.DataFrame,
    window: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
    normalisation: str = "direct",
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
        Rolling window for long-run mean and volatility estimates.
    annualise : bool
        If True, scale by ``periods_per_year``.
    periods_per_year : int
    normalisation : {"std", "level"}
        See ``ou_implied_spread_return`` for details.

    Returns
    -------
    pd.Series
        OU-implied expected return per pair, indexed by ``y_vs_x``.
    """
    # Log-transform so the spread is in log-price space, matching
    # compute_spread_returns (which uses log-returns).  E[ΔS] is then a
    # log-return directly comparable to the spread return series.
    log_prices = np.log(prices_df)

    results = {}
    for _, row in coint_pairs.iterrows():
        label = f"{row['y']}_vs_{row['x']}"
        spread = compute_spread(
            log_prices[row["y"]],
            log_prices[row["x"]],
            row["hedge_ratio"],
            row.get("intercept", 0.0),
        )
        hl = compute_half_life(spread)
        results[label] = ou_implied_spread_return(
            spread, hl,
            window=window,
            annualise=annualise,
            periods_per_year=periods_per_year,
            normalisation=normalisation,
        )

    return pd.Series(results, name="ou_implied_return")


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
    cov = returns.dropna().cov()
    if annualise:
        cov = cov * periods_per_year
    return cov


def shrinkage_covariance(
    returns: pd.DataFrame,
    annualise: bool = True,
    periods_per_year: int = 252,
    estimator: str = "lw",
) -> pd.DataFrame:
    """
    Ledoit-Wolf analytical shrinkage covariance estimator.

    Shrinks the sample covariance toward a scaled identity matrix using
    the closed-form Ledoit & Wolf (2004) optimal shrinkage intensity.
    Chosen over the sample covariance to reduce estimation error and ensure
    positive definiteness; chosen over OAS because the low-dimensional
    regime here (n=3 spreads or n=6 assets, T≈1500 days, p/n ≈ 0.002–0.004)
    already yields a well-conditioned sample covariance, so the two
    estimators converge and LW has stronger theoretical guarantees.

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
    model = LedoitWolf()
    model.fit(clean.values.astype(float))
    cov = pd.DataFrame(
        model.covariance_,
        index=returns.columns,
        columns=returns.columns,
    )
    if annualise:
        cov = cov * periods_per_year
    return cov


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


def spread_vs_asset_estimates(
    prices_df: pd.DataFrame,
    coint_pairs: pd.DataFrame,
    method: str = "historical",
    window: int | None = None,
    span: int = 60,
    annualise: bool = True,
    periods_per_year: int = 252,
    cov_estimator: str = "sample",
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
    cov_estimator : {"sample", "lw"}
        Covariance estimator to use for both spread and asset covariances.
        ``"sample"`` uses the raw sample covariance; ``"lw"`` applies
        Ledoit-Wolf shrinkage (see ``shrinkage_covariance``).

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
    # Use log-returns for consistency with spread returns (which are log-return differentials).
    tickers = list(dict.fromkeys(
        coint_pairs["y"].tolist() + coint_pairs["x"].tolist()
    ))
    asset_rets = np.log(prices_df[tickers]).diff().dropna(how="all")

    # Drop NaNs once so mu and cov are estimated on the same clean sample.
    spread_rets_clean = spread_rets.dropna()
    asset_rets_clean  = asset_rets.dropna()

    # --- Expected returns ---
    if method == "ewma":
        spread_mu = ewma_mean_return(
            spread_rets_clean, span=span,
            annualise=annualise, periods_per_year=periods_per_year,
        )
        asset_mu = ewma_mean_return(
            asset_rets_clean, span=span,
            annualise=annualise, periods_per_year=periods_per_year,
        )
    elif method == "ou":
        spread_mu = build_ou_implied_returns(
            prices_df, coint_pairs,
            window=window or 60,
            annualise=annualise, periods_per_year=periods_per_year,
        )
        asset_mu = historical_mean_return(
            asset_rets_clean, window=window,
            annualise=annualise, periods_per_year=periods_per_year,
        )
    else:  # "historical"
        spread_mu = historical_mean_return(
            spread_rets_clean, window=window,
            annualise=annualise, periods_per_year=periods_per_year,
        )
        asset_mu = historical_mean_return(
            asset_rets_clean, window=window,
            annualise=annualise, periods_per_year=periods_per_year,
        )

    # --- Covariance ---
    if cov_estimator == "lw":
        cov_fn = lambda r: shrinkage_covariance(
            r, annualise=annualise,
            periods_per_year=periods_per_year,
        )
    else:
        cov_fn = lambda r: sample_covariance(
            r, annualise=annualise,
            periods_per_year=periods_per_year,
        )

    spread_cov = cov_fn(spread_rets_clean)
    asset_cov  = cov_fn(asset_rets_clean)

    return {
        "spread_returns": spread_rets,
        "spread_mu": spread_mu,
        "spread_cov": spread_cov,
        "asset_returns": asset_rets,
        "asset_mu": asset_mu,
        "asset_cov": asset_cov,
    }
