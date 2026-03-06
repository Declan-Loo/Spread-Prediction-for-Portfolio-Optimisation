"""
Modern Portfolio Theory optimiser for pairs trading portfolios.

Provides OLS hedge-ratio estimation (static and rolling) and Markowitz
mean-variance optimisation across multiple pair-spread return streams.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """
    Static OLS hedge ratio: y = α + β·x + ε.

    Returns
    -------
    (hedge_ratio β, intercept α)
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_clean, x_clean = aligned.iloc[:, 0], aligned.iloc[:, 1]

    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, X).fit()
    intercept, hedge_ratio = model.params

    return float(hedge_ratio), float(intercept)


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Rolling-window OLS hedge ratio over time.

    Returns
    -------
    pd.DataFrame with columns ['hedge_ratio', 'intercept'].
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(aligned)

    ratios = []
    for end in range(window, n + 1):
        chunk = aligned.iloc[end - window : end]
        X = sm.add_constant(chunk["x"])
        model = sm.OLS(chunk["y"], X).fit()
        intercept, hr = model.params
        ratios.append(
            {
                "date": aligned.index[end - 1],
                "hedge_ratio": hr,
                "intercept": intercept,
            }
        )

    return pd.DataFrame(ratios).set_index("date")


# ---------------------------------------------------------------------------
# Markowitz mean-variance utilities
# ---------------------------------------------------------------------------

def _portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Annualised return and volatility for a given weight vector."""
    port_return = np.dot(weights, mean_returns) * periods_per_year
    port_vol = np.sqrt(
        np.dot(weights, np.dot(cov_matrix, weights)) * periods_per_year
    )
    return port_return, port_vol


def minimum_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Minimum-variance portfolio weights (long-only constraint).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset / spread (columns).

    Returns
    -------
    np.ndarray of portfolio weights summing to 1.
    """
    n = returns.shape[1]
    cov = returns.cov().values

    def objective(w):
        return np.dot(w, np.dot(cov, w))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result.x


def max_sharpe_weights(
    returns: pd.DataFrame,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Maximum Sharpe ratio (tangency) portfolio weights (long-only).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset / spread (columns).
    rf_annual : float
        Annualised risk-free rate.
    periods_per_year : int
        Trading days per year.

    Returns
    -------
    np.ndarray of portfolio weights summing to 1.
    """
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    def neg_sharpe(w):
        port_ret, port_vol = _portfolio_stats(w, mean_ret, cov, periods_per_year)
        if port_vol == 0:
            return 1e6
        return -(port_ret - rf_annual) / port_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result.x


def mean_variance_weights(
    returns: pd.DataFrame,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> dict:
    """
    Convenience wrapper returning both max-Sharpe and min-variance portfolios.

    Returns
    -------
    dict with keys:
        max_sharpe_weights, min_var_weights,
        max_sharpe_return, max_sharpe_vol,
        min_var_return, min_var_vol
    """
    mean_ret = returns.mean().values
    cov = returns.cov().values

    w_sharpe = max_sharpe_weights(returns, rf_annual, periods_per_year)
    w_minvar = minimum_variance_weights(returns)

    sr_ret, sr_vol = _portfolio_stats(w_sharpe, mean_ret, cov, periods_per_year)
    mv_ret, mv_vol = _portfolio_stats(w_minvar, mean_ret, cov, periods_per_year)

    return {
        "max_sharpe_weights": w_sharpe,
        "min_var_weights": w_minvar,
        "max_sharpe_return": sr_ret,
        "max_sharpe_vol": sr_vol,
        "min_var_return": mv_ret,
        "min_var_vol": mv_vol,
    }


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compute the efficient frontier: for a range of target returns, find the
    minimum-volatility portfolio.

    Returns
    -------
    pd.DataFrame with columns ['return', 'volatility', 'sharpe'].
    """
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    # Anchor on the min-var and max-return portfolios
    w_min = minimum_variance_weights(returns)
    ret_min, _ = _portfolio_stats(w_min, mean_ret, cov, periods_per_year)
    ret_max = float(np.max(mean_ret)) * periods_per_year

    target_returns = np.linspace(ret_min, ret_max, n_points)
    records = []

    for target in target_returns:
        def objective(w):
            return np.dot(w, np.dot(cov, w))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, t=target: np.dot(w, mean_ret) * periods_per_year - t,
            },
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            port_ret, port_vol = _portfolio_stats(
                result.x, mean_ret, cov, periods_per_year
            )
            sharpe = (port_ret - rf_annual) / port_vol if port_vol > 0 else np.nan
            records.append(
                {"return": port_ret, "volatility": port_vol, "sharpe": sharpe}
            )

    return pd.DataFrame(records)


def estimate_expected_returns(
    returns: pd.DataFrame,
    window: int | None = None,
    annualise: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Estimate expected returns for each column in a return DataFrame.

    This function is used for spread-based return estimation in the MPT
    optimiser, providing a simple alternative to naive full-sample means.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return series (e.g. spread returns for each pair).
    window : int | None
        Optional lookback window (in periods). If provided, use the last
        `window` observations for the mean; otherwise use the full sample.
    annualise : bool
        If True, scale the expected daily mean by periods_per_year.
    periods_per_year : int
        Number of periods per year (252 for trading days).

    Returns
    -------
    pd.Series
        Estimated expected (annualised) return per column.
    """
    if window is not None and window > 0:
        sample = returns.tail(window)
    else:
        sample = returns

    mean_daily = sample.mean()

    if annualise:
        return mean_daily * periods_per_year
    return mean_daily
