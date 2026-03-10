"""
Benchmark return series and metrics for backtest comparison.

Provides: risk-free rate, buy-and-hold (single pair or multi-asset),
equal-weight all pairs, and optional market (e.g. S&P 500).
Used to compare spread-based strategy vs historical MPT vs benchmarks
(Sharpe uses risk-free; volatility reduction and charts use B&H, equal-weight, market).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import (
    compute_ex_post_sharpe_ratio,
    compute_max_drawdown,
    compute_volatility_reduction,
)


# Default risk-free rate (e.g. 3-month T-bill), annualised - obtained from averaging the risk-free rate in training period
DEFAULT_RF_ANNUAL = 0.02
PERIODS_PER_YEAR = 252


def risk_free_returns(
    index: pd.DatetimeIndex,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Constant daily risk-free rate over the given index."""
    daily_rf = rf_annual / periods_per_year
    return pd.Series(daily_rf, index=index, name="risk_free")


def buy_and_hold_returns(
    prices_df: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Buy-and-hold portfolio returns from price DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame
        One column per asset; index = dates.
    weights : np.ndarray | None
        Weights per column (must sum to 1). If None, equal weight.

    Returns
    -------
    pd.Series
        Daily portfolio returns.
    """
    ret = prices_df.pct_change().dropna(how="all")
    if weights is None:
        weights = np.ones(ret.shape[1]) / ret.shape[1]
    weights = np.asarray(weights)
    if weights.size != ret.shape[1]:
        raise ValueError("weights length must match number of columns")
    port_ret = ret @ weights
    port_ret.name = "buy_and_hold"
    return port_ret


def equal_weight_pairs_returns(
    prices_df: pd.DataFrame,
    coint_pairs: pd.DataFrame,
) -> pd.Series:
    """
    Equal-weight portfolio of spread returns for each cointegrated pair.

    Each pair contributes (y_ret - hedge_ratio * x_ret); total return
    is the mean across pairs (equal weight per pair).
    """
    pair_rets = []
    for _, row in coint_pairs.iterrows():
        y_ret = prices_df[row["y"]].pct_change()
        x_ret = prices_df[row["x"]].pct_change()
        pair_rets.append(y_ret - row["hedge_ratio"] * x_ret)
    combined = pd.concat(pair_rets, axis=1).mean(axis=1)
    combined.name = "equal_weight_pairs"
    return combined.dropna()


def market_returns(market_prices: pd.Series) -> pd.Series:
    """
    Daily returns from a single market index price series (e.g. S&P 500).

    Parameters
    ----------
    market_prices : pd.Series
        Close prices of the market index.

    Returns
    -------
    pd.Series
        Daily returns, name "market".
    """
    ret = market_prices.pct_change().dropna()
    ret.name = "market"
    return ret


def compute_benchmark_metrics(
    returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    periods_per_year: int = PERIODS_PER_YEAR,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """
    Compute standard metrics for a return series (strategy or benchmark).

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    rf_annual : float
        Risk-free rate for Sharpe.
    periods_per_year : int
    benchmark_returns : pd.Series | None
        If provided, volatility reduction is computed vs this benchmark.

    Returns
    -------
    dict
        sharpe_ratio, max_drawdown, total_return, annualised_volatility,
        and optionally volatility_reduction (if benchmark_returns given).
    """
    ret = returns.dropna()
    if len(ret) < 2:
        return {
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "total_return": 0.0,
            "annualised_volatility": np.nan,
            "volatility_reduction": np.nan,
        }
    sharpe = compute_ex_post_sharpe_ratio(ret, rf_annual=rf_annual, periods_per_year=periods_per_year)
    max_dd = compute_max_drawdown(ret)
    total_ret = float((1 + ret).prod() - 1)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(periods_per_year))
    out = {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "annualised_volatility": ann_vol,
    }
    if benchmark_returns is not None:
        try:
            common = ret.index.intersection(benchmark_returns.dropna().index)
            if len(common) >= 2:
                out["volatility_reduction"] = compute_volatility_reduction(
                    ret.reindex(common).dropna(),
                    benchmark_returns.reindex(common).dropna(),
                    periods_per_year=periods_per_year,
                )
            else:
                out["volatility_reduction"] = np.nan
        except Exception:
            out["volatility_reduction"] = np.nan
    else:
        out["volatility_reduction"] = np.nan
    return out


def build_all_benchmarks(
    test_prices: pd.DataFrame,
    coint_pairs: pd.DataFrame,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    market_prices: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """
    Build benchmark daily return series for the test period.

    Parameters
    ----------
    test_prices : pd.DataFrame
        Close prices over test period; columns = tickers.
    coint_pairs : pd.DataFrame
        Cointegrated pairs (columns y, x, hedge_ratio, intercept).
    rf_annual : float
    market_prices : pd.Series | None
        If provided (e.g. S&P 500), included as "sp500" in the result.

    Returns
    -------
    dict[str, pd.Series]
        Keys: "risk_free", "buy_hold_pair", "equal_weight_pairs", optionally "sp500".
        "buy_hold_pair" is equal-weight of the first cointegrated pair's two legs
        (for single-pair backtest comparison). For multi-pair, consider using
        equal_weight_pairs as the main B&H benchmark.
    """
    index = test_prices.index
    out = {}
    out["risk_free"] = risk_free_returns(index, rf_annual=rf_annual)
    if not coint_pairs.empty:
        row = coint_pairs.iloc[0]
        two_legs = test_prices[[row["y"], row["x"]]]
        out["buy_hold_pair"] = buy_and_hold_returns(two_legs, weights=None)
        out["equal_weight_pairs"] = equal_weight_pairs_returns(test_prices, coint_pairs)
    if market_prices is not None:
        ret = market_returns(market_prices)
        ret = ret.reindex(index).dropna(how="all")
        if not ret.empty:
            out["sp500"] = ret
    return out


def historical_mpt_returns(
    train_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Out-of-sample returns of the historical mean-variance (max-Sharpe) portfolio.

    Weights are estimated on train_prices (historical mean and covariance of
    asset returns); the same weights are applied to test_prices. This is the
    traditional MPT approach for comparison with the spread-based strategy.

    Parameters
    ----------
    train_prices : pd.DataFrame
        Training period prices; columns = asset tickers (e.g. y, x for one pair).
    test_prices : pd.DataFrame
        Test period prices; same columns as train_prices.
    rf_annual : float
    periods_per_year : int

    Returns
    -------
    pd.Series
        Daily portfolio returns over the test period (aligned to test_prices index).
    """
    train_ret = train_prices.pct_change().dropna()
    if train_ret.shape[1] < 2 or len(train_ret) < 2:
        test_ret = test_prices.pct_change()
        return (test_ret.iloc[:, 0] if test_ret.shape[1] == 1 else test_ret.mean(axis=1)).fillna(0)
    mean_ret = train_ret.mean().values
    cov = train_ret.cov().values
    n = len(mean_ret)

    def neg_sharpe(w):
        port_ret = np.dot(w, mean_ret) * periods_per_year
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)) * periods_per_year)
        if port_vol == 0:
            return 1e6
        return -(port_ret - rf_annual) / port_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = res.x

    test_ret = test_prices[train_prices.columns].pct_change()
    port_ret = (test_ret * weights).sum(axis=1)
    port_ret.name = "historical_mpt"
    return port_ret.dropna()
