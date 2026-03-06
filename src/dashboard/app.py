"""
Pairs-Trading Portfolio Optimiser — Streamlit Dashboard

  Input:  portfolio (pair selection + date range + capital).
  Output: cointegration, spread & z-scores, backtest metrics with benchmarks
          (buy-and-hold, risk-free, S&P 500, equal-weight all pairs),
          and comparison: Spread-based strategy vs Historical mean-variance MPT.

Launch:  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.refinitiv_client import get_close_prices
from src.data.yfinance_sp500 import get_sp500_prices
from src.modelling.cointegration import screen_pairs, engle_granger_test
from src.modelling.spread_analysis import (
    compute_spread,
    compute_zscore,
    spread_summary,
)
from src.modelling.optimiser import (
    mean_variance_weights,
    efficient_frontier,
    rolling_hedge_ratio,
    build_pair_returns,
)
from src.modelling.config import (
    CANDIDATE_PAIRS,
    TICKER_NAMES,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
)
from src.backtesting.engine import PairsBacktestEngine, BacktestConfig
from src.backtesting.metrics import compute_volatility_reduction
from src.backtesting.benchmarks import (
    build_all_benchmarks,
    compute_benchmark_metrics,
    historical_mpt_returns,
)
from src.dashboard.components import (
    plot_spread_with_bands,
    plot_cointegration_results,
    plot_cumulative_returns,
    plot_cumulative_returns_multi,
    plot_efficient_frontier,
    plot_zscore_heatmap,
    plot_position_timeline,
    format_metrics_table,
)

st.set_page_config(
    page_title="Pairs-Trading Portfolio Optimiser",
    layout="wide",
)
st.title("Pairs-Trading Portfolio Optimiser")
st.caption("Input: portfolio (pairs + dates). Output: backtest metrics vs benchmarks and Historical MPT.")

# ---------------------------------------------------------------------------
# Sidebar — Portfolio input
# ---------------------------------------------------------------------------
st.sidebar.header("Portfolio")
pair_options = [f"{y} / {x}" for y, x in CANDIDATE_PAIRS]
selected_pair_labels = st.sidebar.multiselect(
    "Pair selection",
    options=pair_options,
    default=pair_options,
    help="Pairs to include in cointegration and backtest.",
)
portfolio_pairs = []
for label in selected_pair_labels:
    parts = label.split(" / ", 1)
    if len(parts) == 2:
        portfolio_pairs.append((parts[0].strip(), parts[1].strip()))

st.sidebar.subheader("Date range")
train_start = st.sidebar.date_input("Train start", datetime.fromisoformat(TRAIN_START))
train_end = st.sidebar.date_input("Train end", datetime.fromisoformat(TRAIN_END))
test_start = st.sidebar.date_input("Test start", datetime.fromisoformat(TEST_START))
test_end = st.sidebar.date_input("Test end", datetime.fromisoformat(TEST_END))

st.sidebar.subheader("Backtest settings")
entry_z = st.sidebar.slider("Entry Z", 1.0, 4.0, 2.0, 0.1)
exit_z = st.sidebar.slider("Exit Z", 0.0, 2.0, 0.0, 0.1)
stop_z = st.sidebar.slider("Stop-loss Z", 2.5, 6.0, 4.0, 0.25)
lookback = st.sidebar.slider("Lookback window", 20, 120, 60, 5)
tx_cost = st.sidebar.number_input("Transaction cost (bps)", min_value=0.0, value=10.0, step=1.0)
initial_capital = st.sidebar.number_input("Initial capital (£)", min_value=1_000, value=100_000, step=5_000)
rf_annual = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")

include_sp500 = st.sidebar.checkbox("Include S&P 500 benchmark", value=True, help="Fetch S&P 500 via yfinance for the test date range.")

if not portfolio_pairs:
    st.warning("Select at least one pair in the sidebar.")
    st.stop()

@st.cache_data(show_spinner="Fetching price data from LSEG…")
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return get_close_prices(list(tickers), start=start, end=end)

unique_tickers = sorted({t for pair in portfolio_pairs for t in pair})
train_prices = load_prices(tuple(unique_tickers), str(train_start), str(train_end))
test_prices = load_prices(tuple(unique_tickers), str(test_start), str(test_end))

# Optional S&P 500 (yfinance; uses the same test date range)
sp500_prices = None
if include_sp500:
    sp500_prices = get_sp500_prices(str(test_start), str(test_end))
    if sp500_prices.empty:
        sp500_prices = None

screening_df = screen_pairs(train_prices, portfolio_pairs)
coint_pairs = screening_df[screening_df["is_cointegrated"]]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_coint, tab_spread, tab_bt, tab_compare, tab_opt = st.tabs([
    "Cointegration",
    "Spread & z-scores",
    "Backtest metrics",
    "Strategy vs MPT vs Benchmarks",
    "Efficient frontier",
])

with tab_coint:
    st.header("Cointegration (training period)")
    st.dataframe(
        screening_df.style.format({
            "hedge_ratio": "{:.4f}",
            "intercept": "{:.4f}",
            "adf_stat": "{:.4f}",
            "p_value": "{:.6f}",
        }),
        use_container_width=True,
    )
    st.plotly_chart(plot_cointegration_results(screening_df), use_container_width=True)
    st.info(f"{len(coint_pairs)} of {len(portfolio_pairs)} pairs are cointegrated at the 5% level.")

with tab_spread:
    st.header("Spread & z-scores")
    all_pair_labels = [f"{y} / {x}" for y, x in portfolio_pairs]
    selected_label = st.selectbox("Select pair", all_pair_labels)
    idx = all_pair_labels.index(selected_label)
    sel_y, sel_x = portfolio_pairs[idx]
    eg = engle_granger_test(train_prices[sel_y], train_prices[sel_x])
    hr_static, intercept_static = eg["hedge_ratio"], eg["intercept"]
    use_rolling = st.checkbox("Use rolling hedge ratio", value=False)
    if use_rolling:
        roll_df = rolling_hedge_ratio(train_prices[sel_y], train_prices[sel_x], window=lookback)
        hr_latest = roll_df["hedge_ratio"].iloc[-1]
        int_latest = roll_df["intercept"].iloc[-1]
        st.caption(f"Rolling β = {hr_latest:.4f}, α = {int_latest:.4f}  |  Static β = {hr_static:.4f}, α = {intercept_static:.4f}")
        import plotly.graph_objects as go
        hr_fig = go.Figure()
        hr_fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df["hedge_ratio"], name="Rolling β", line=dict(color="#636EFA")))
        hr_fig.add_hline(y=hr_static, line=dict(color="red", dash="dash"), annotation_text=f"Static β = {hr_static:.4f}")
        hr_fig.update_layout(yaxis_title="Hedge ratio (β)", height=280, margin=dict(t=20, b=20))
        st.plotly_chart(hr_fig, use_container_width=True)
        hr, intercept = hr_latest, int_latest
    else:
        hr, intercept = hr_static, intercept_static
    spread = compute_spread(train_prices[sel_y], train_prices[sel_x], hr, intercept)
    zscore = compute_zscore(spread, window=lookback)
    st.plotly_chart(
        plot_spread_with_bands(spread, zscore, entry_z=entry_z, exit_z=exit_z),
        use_container_width=True,
    )
    summary = spread_summary(train_prices[sel_y], train_prices[sel_x], hr, intercept, window=lookback)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Half-life (days)", f"{summary['half_life']:.1f}")
    c2.metric("Hurst exponent", f"{summary['hurst']:.3f}")
    c3.metric("ADF p-value", f"{summary['adf_pvalue']:.4f}")
    c4.metric("Current z-score", f"{summary['current_zscore']:.2f}")

with tab_bt:
    st.header("Backtest metrics (test period)")
    if coint_pairs.empty:
        st.warning("No cointegrated pairs — nothing to backtest.")
    else:
        config = BacktestConfig(
            entry_z=entry_z, exit_z=exit_z, stop_loss_z=stop_z,
            lookback_window=lookback, transaction_cost_bps=tx_cost,
            initial_capital=float(initial_capital),
        )
        engine = PairsBacktestEngine(config)
        bt_pair_labels = [f"{r['y']} / {r['x']}" for _, r in coint_pairs.iterrows()]
        bt_selected = st.selectbox("Select cointegrated pair", bt_pair_labels, key="bt_pair")
        bt_idx = bt_pair_labels.index(bt_selected)
        bt_row = coint_pairs.iloc[bt_idx]
        y_test = test_prices[bt_row["y"]]
        x_test = test_prices[bt_row["x"]]
        result = engine.run(y_test, x_test, hedge_ratio=bt_row["hedge_ratio"], intercept=bt_row["intercept"])

        st.subheader("Strategy performance")
        st.dataframe(format_metrics_table(result.metrics), use_container_width=True, hide_index=True)

        # Benchmarks (risk-free, B&H pair, equal-weight pairs, optional S&P 500)
        benchmarks = build_all_benchmarks(
            test_prices, coint_pairs, rf_annual=rf_annual, market_prices=sp500_prices,
        )
        b_cols = [k for k in ["buy_hold_pair", "risk_free", "equal_weight_pairs", "sp500"] if k in benchmarks]

        st.subheader("Benchmark metrics (Sharpe uses risk-free rate)")
        bench_metrics = []
        for name in b_cols:
            ret = benchmarks[name]
            m = compute_benchmark_metrics(ret, rf_annual=rf_annual, benchmark_returns=benchmarks.get("buy_hold_pair"))
            m["name"] = name
            bench_metrics.append(m)
        bench_df = pd.DataFrame(bench_metrics).set_index("name")
        bench_df = bench_df[["sharpe_ratio", "max_drawdown", "total_return", "annualised_volatility", "volatility_reduction"]]
        bench_df.columns = ["Sharpe", "Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]
        st.dataframe(bench_df.style.format("{:.2f}", subset=["Sharpe"]).format("{:.2%}", subset=["Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]), use_container_width=True)

        st.subheader("Cumulative returns: strategy vs benchmarks")
        series_for_chart = {"Strategy (spread-based)": result.daily_returns}
        label_map = {"risk_free": "Risk-free", "buy_hold_pair": "B&H (pair)", "equal_weight_pairs": "Equal-weight pairs", "sp500": "S&P 500"}
        for k in b_cols:
            series_for_chart[label_map.get(k, k)] = benchmarks[k]
        st.plotly_chart(plot_cumulative_returns_multi(series_for_chart), use_container_width=True)

        st.subheader("Position timeline")
        st.plotly_chart(plot_position_timeline(result.positions["position"]), use_container_width=True)
        st.subheader("Trade log")
        if not result.trades.empty:
            st.dataframe(result.trades, use_container_width=True)
        else:
            st.info("No trades in the test period.")

with tab_compare:
    st.header("Spread-based strategy vs Historical mean-variance MPT")
    st.caption("Research comparison: same pair, same period. MPT weights from training data; spread strategy is z-score mean reversion.")
    if coint_pairs.empty:
        st.warning("No cointegrated pairs.")
    else:
        bt_pair_labels_c = [f"{r['y']} / {r['x']}" for _, r in coint_pairs.iterrows()]
        compare_pair = st.selectbox("Select pair for comparison", bt_pair_labels_c, key="compare_pair")
        c_idx = bt_pair_labels_c.index(compare_pair)
        c_row = coint_pairs.iloc[c_idx]
        y_t = test_prices[c_row["y"]]
        x_t = test_prices[c_row["x"]]
        train_two = train_prices[[c_row["y"], c_row["x"]]].copy()
        test_two = test_prices[[c_row["y"], c_row["x"]]].copy()

        # Spread-based backtest
        config = BacktestConfig(
            entry_z=entry_z, exit_z=exit_z, stop_loss_z=stop_z,
            lookback_window=lookback, transaction_cost_bps=tx_cost,
            initial_capital=float(initial_capital),
        )
        engine = PairsBacktestEngine(config)
        spread_result = engine.run(y_t, x_t, hedge_ratio=c_row["hedge_ratio"], intercept=c_row["intercept"])
        strategy_ret = spread_result.daily_returns

        # Historical MPT
        mpt_ret = historical_mpt_returns(train_two, test_two, rf_annual=rf_annual)

        # Benchmarks
        benchmarks_c = build_all_benchmarks(test_prices, coint_pairs, rf_annual=rf_annual, market_prices=sp500_prices)

        # Metrics table
        def row_metrics(name: str, ret: pd.Series, bench_bh: pd.Series | None = None) -> dict:
            m = compute_benchmark_metrics(ret, rf_annual=rf_annual, benchmark_returns=bench_bh)
            return {"Strategy": name, **m}

        bh_ret = benchmarks_c.get("buy_hold_pair")
        rows = [
            row_metrics("Spread-based (z-score)", strategy_ret, bh_ret),
            row_metrics("Historical MPT (max-Sharpe)", mpt_ret, bh_ret),
        ]
        for bname, bret in benchmarks_c.items():
            label = {"risk_free": "Risk-free", "buy_hold_pair": "B&H (pair)", "equal_weight_pairs": "Equal-weight pairs", "sp500": "S&P 500"}.get(bname, bname)
            rows.append(row_metrics(label, bret, bh_ret))
        compare_df = pd.DataFrame(rows)
        compare_df = compare_df.set_index("Strategy")
        compare_df = compare_df[["sharpe_ratio", "max_drawdown", "total_return", "annualised_volatility", "volatility_reduction"]]
        compare_df.columns = ["Sharpe", "Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]
        st.dataframe(
            compare_df.style.format("{:.2f}", subset=["Sharpe"]).format("{:.2%}", subset=["Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]),
            use_container_width=True,
        )

        st.subheader("Cumulative returns comparison")
        chart_series = {
            "Spread-based": strategy_ret,
            "Historical MPT": mpt_ret,
        }
        for bname, bret in benchmarks_c.items():
            label = {"risk_free": "Risk-free", "buy_hold_pair": "B&H (pair)", "equal_weight_pairs": "Equal-weight pairs", "sp500": "S&P 500"}.get(bname, bname)
            chart_series[label] = bret
        st.plotly_chart(plot_cumulative_returns_multi(chart_series), use_container_width=True)

with tab_opt:
    st.header("Efficient frontier")
    if coint_pairs.empty:
        st.warning("No cointegrated pairs — cannot optimise.")
    else:
        returns_df = build_pair_returns(test_prices, coint_pairs)
        if returns_df.shape[1] < 2:
            st.info("Need at least 2 cointegrated pairs for the frontier.")
            st.dataframe(returns_df.describe(), use_container_width=True)
        else:
            mv = mean_variance_weights(returns_df, rf_annual=rf_annual)
            frontier_df = efficient_frontier(returns_df, rf_annual=rf_annual)
            max_sharpe_pt = (mv["max_sharpe_vol"], mv["max_sharpe_return"])
            min_var_pt = (mv["min_var_vol"], mv["min_var_return"])
            st.plotly_chart(
                plot_efficient_frontier(frontier_df, max_sharpe_pt, min_var_pt),
                use_container_width=True,
            )
            bench_ret = returns_df.mean(axis=1)
            port_ret_sharpe = (returns_df * mv["max_sharpe_weights"]).sum(axis=1)
            port_ret_minvar = (returns_df * mv["min_var_weights"]).sum(axis=1)
            mask = port_ret_sharpe.notna() & bench_ret.notna()
            vol_red_sharpe = compute_volatility_reduction(port_ret_sharpe[mask], bench_ret[mask])
            vol_red_minvar = compute_volatility_reduction(port_ret_minvar[mask], bench_ret[mask])
            st.subheader("Portfolio-level volatility reduction")
            col1, col2 = st.columns(2)
            col1.metric("Max-Sharpe vs equal-weight", f"{vol_red_sharpe:.1%}")
            col2.metric("Min-variance vs equal-weight", f"{vol_red_minvar:.1%}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Max-Sharpe weights")
                w_df = pd.DataFrame({"Pair": returns_df.columns, "Weight": mv["max_sharpe_weights"]})
                w_df["Weight"] = w_df["Weight"].map("{:.2%}".format)
                st.dataframe(w_df, hide_index=True)
            with col_b:
                st.subheader("Min-variance weights")
                w_df = pd.DataFrame({"Pair": returns_df.columns, "Weight": mv["min_var_weights"]})
                w_df["Weight"] = w_df["Weight"].map("{:.2%}".format)
                st.dataframe(w_df, hide_index=True)
