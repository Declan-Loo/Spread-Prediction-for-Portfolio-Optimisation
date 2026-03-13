"""
Modern Portfolio Theory optimiser for pairs trading portfolios.

Provides OLS hedge-ratio estimation (static and rolling) and Markowitz
mean-variance optimisation across multiple pair-spread return streams.

All optimisation functions accept optional ``expected_returns`` and
``cov_matrix`` parameters so that custom return estimates (e.g. OU-implied
spread returns from ``return_estimation``) can be plugged in.  When omitted,
the historical sample mean and covariance are used (traditional MPT).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Hedge-ratio estimation
# ---------------------------------------------------------------------------


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


def _resolve_inputs(
    returns: pd.DataFrame,
    expected_returns: pd.Series | np.ndarray | None = None,
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve expected-return vector and covariance matrix.

    If ``expected_returns`` or ``cov_matrix`` are not provided, fall back
    to the sample mean and sample covariance of ``returns``.
    """
    if expected_returns is not None:
        mu = np.asarray(expected_returns, dtype=float)
    else:
        mu = returns.mean().values

    if cov_matrix is not None:
        cov = np.asarray(cov_matrix, dtype=float)
    else:
        cov = returns.cov().values

    return mu, cov


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


def minimum_variance_weights(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    """
    Minimum-variance portfolio weights (long-only constraint).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset / spread (columns).
    cov_matrix : pd.DataFrame | np.ndarray | None
        Custom covariance matrix (e.g. shrinkage estimator from
        ``return_estimation.shrinkage_covariance``).  If None, uses
        the sample covariance of ``returns``.

    Returns
    -------
    np.ndarray of portfolio weights summing to 1.
    """
    n = returns.shape[1]
    _, cov = _resolve_inputs(returns, cov_matrix=cov_matrix)

    def objective(w):
        return np.dot(w, np.dot(cov, w))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}   # ← add this
    )
    return result.x

def max_sharpe_weights(
    returns: pd.DataFrame,
    expected_returns: pd.Series | np.ndarray | None = None,
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Maximum Sharpe ratio (tangency) portfolio weights (long-only).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset / spread (columns).
    expected_returns : pd.Series | np.ndarray | None
        Custom *daily* expected-return vector (e.g. OU-implied estimates
        from ``return_estimation``).  If None, uses ``returns.mean()``.
    cov_matrix : pd.DataFrame | np.ndarray | None
        Custom covariance matrix.  If None, uses ``returns.cov()``.
    rf_annual : float
        Annualised risk-free rate.
    periods_per_year : int
        Trading days per year.

    Returns
    -------
    np.ndarray of portfolio weights summing to 1.
    """
    n = returns.shape[1]
    mean_ret, cov = _resolve_inputs(returns, expected_returns, cov_matrix)

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
    expected_returns: pd.Series | np.ndarray | None = None,
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> dict:
    """
    Convenience wrapper returning both max-Sharpe and min-variance portfolios.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset / spread.
    expected_returns : pd.Series | np.ndarray | None
        Custom daily expected-return vector.  Passed through to
        ``max_sharpe_weights`` (min-variance does not use it).
    cov_matrix : pd.DataFrame | np.ndarray | None
        Custom covariance matrix.  Passed to both optimisers.
    rf_annual : float
    periods_per_year : int

    Returns
    -------
    dict with keys:
        max_sharpe_weights, min_var_weights,
        max_sharpe_return, max_sharpe_vol,
        min_var_return, min_var_vol
    """
    mean_ret, cov = _resolve_inputs(returns, expected_returns, cov_matrix)

    w_sharpe = max_sharpe_weights(
        returns, expected_returns, cov_matrix, rf_annual, periods_per_year,
    )
    w_minvar = minimum_variance_weights(returns, cov_matrix)

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
    expected_returns: pd.Series | np.ndarray | None = None,
    cov_matrix: pd.DataFrame | np.ndarray | None = None,
    n_points: int = 50,
    rf_annual: float = 0.02,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compute the efficient frontier: for a range of target returns, find the
    minimum-volatility portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset / spread.
    expected_returns : pd.Series | np.ndarray | None
        Custom daily expected-return vector.
    cov_matrix : pd.DataFrame | np.ndarray | None
        Custom covariance matrix.
    n_points : int
    rf_annual : float
    periods_per_year : int

    Returns
    -------
    pd.DataFrame with columns ['return', 'volatility', 'sharpe'].
    """
    n = returns.shape[1]
    mean_ret, cov = _resolve_inputs(returns, expected_returns, cov_matrix)

    # Anchor on the min-var and max-return portfolios
    w_min = minimum_variance_weights(returns, cov_matrix)
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
