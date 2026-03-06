from .engine import BacktestConfig, BacktestResult, PairsBacktestEngine
from .metrics import (
    compute_ex_post_sharpe_ratio,
    compute_max_drawdown,
    compute_volatility_reduction,
)
from .benchmarks import (
    risk_free_returns,
    buy_and_hold_returns,
    equal_weight_pairs_returns,
    market_returns,
    compute_benchmark_metrics,
    build_all_benchmarks,
    historical_mpt_returns,
)

__all__ = [
    # Engine
    "PairsBacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # Metrics
    "compute_ex_post_sharpe_ratio",
    "compute_max_drawdown",
    "compute_volatility_reduction",
    # Benchmarks
    "risk_free_returns",
    "buy_and_hold_returns",
    "equal_weight_pairs_returns",
    "market_returns",
    "compute_benchmark_metrics",
    "build_all_benchmarks",
    "historical_mpt_returns",
]
