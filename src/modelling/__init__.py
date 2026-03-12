from .config import (
    TICKERS,
    CANDIDATE_PAIRS,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
    INTERVAL,
    TICKER_NAMES,
)

from .cointegration import (
    adf_test,
    engle_granger_test,
    screen_pairs,
)

from .spread_analysis import (
    compute_spread,
    compute_zscore,
    compute_half_life,
    compute_rolling_half_life,
    compute_hurst_exponent,
    compute_rolling_zscore,
    spread_summary,
)

from .return_estimation import (
    compute_spread_returns,
    build_spread_return_matrix,
    historical_mean_return,
    ewma_return,
    ou_expected_return,
    build_ou_expected_returns,
    sample_covariance,
    shrinkage_covariance,
    spread_vs_traditional_estimates,
)

from .optimiser import (
    ols_hedge_ratio,
    rolling_hedge_ratio,
    mean_variance_weights,
    minimum_variance_weights,
    max_sharpe_weights,
    efficient_frontier,
)

__all__ = [
    # Config
    "TICKERS",
    "CANDIDATE_PAIRS",
    "TRAIN_START",
    "TRAIN_END",
    "TEST_START",
    "TEST_END",
    "INTERVAL",
    "TICKER_NAMES",
    # Cointegration
    "adf_test",
    "engle_granger_test",
    "screen_pairs",
    # Spread analysis
    "compute_spread",
    "compute_zscore",
    "compute_half_life",
    "compute_rolling_half_life",
    "compute_hurst_exponent",
    "compute_rolling_zscore",
    "spread_summary",
    # Return estimation
    "compute_spread_returns",
    "build_spread_return_matrix",
    "historical_mean_return",
    "ewma_return",
    "ou_expected_return",
    "build_ou_expected_returns",
    "sample_covariance",
    "shrinkage_covariance",
    "spread_vs_traditional_estimates",
    # Optimiser
    "ols_hedge_ratio",
    "rolling_hedge_ratio",
    "mean_variance_weights",
    "minimum_variance_weights",
    "max_sharpe_weights",
    "efficient_frontier",
]
