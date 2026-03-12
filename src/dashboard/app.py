"""
Pairs-Trading Portfolio Optimiser — Streamlit Dashboard

  Input:  portfolio (pair selection + date range + capital).
  Output: cointegration, spread & z-scores, estimated expected returns
          (spread-based vs traditional), backtest metrics with benchmarks
          (buy-and-hold, risk-free, S&P 500, equal-weight all pairs),
          and portfolio optimisation comparison.

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
    compute_half_life,
    compute_hurst_exponent,
    spread_summary,
)
from src.modelling.optimiser import (
    mean_variance_weights,
    efficient_frontier,
    rolling_hedge_ratio,
)
from src.modelling.return_estimation import (
    build_spread_return_matrix,
    historical_mean_return,
    ewma_return,
    build_ou_expected_returns,
    sample_covariance,
    shrinkage_covariance,
)
from src.modelling.config import (
    CANDIDATE_PAIRS,
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
    plot_cumulative_returns_multi,
    plot_efficient_frontier,
    plot_position_timeline,
    format_metrics_table,
    plot_return_estimates_comparison,
    plot_rolling_return_estimate,
    plot_drawdown,
    plot_returns_distribution,
    plot_rolling_sharpe,
)

st.set_page_config(
    page_title="Pairs-Trading Portfolio Optimiser",
    layout="wide",
)
st.title("Pairs-Trading Portfolio Optimiser")
st.caption(
    "Estimating expected returns by predicting the spread between "
    "cointegrated asset pairs to optimise a portfolio."
)

# ---------------------------------------------------------------------------
# Sidebar — Portfolio input
# ---------------------------------------------------------------------------
st.sidebar.header("Portfolio")

pair_input_mode = st.sidebar.radio(
    "Pair input mode",
    ["Preset pairs", "Custom tickers"],
    horizontal=True,
    help="Use preset candidate pairs or enter your own LSEG ticker symbols.",
)

if pair_input_mode == "Preset pairs":
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
else:
    st.sidebar.caption(
        "Enter LSEG Workspace ticker symbols (RICs). "
        "Define pairs as comma-separated lines: `TICKER_Y, TICKER_X`"
    )
    custom_input = st.sidebar.text_area(
        "Custom pairs (one per line)",
        placeholder="AAPL.O, MSFT.O\nGOOGL.O, META.O\nXOM.N, CVX.N",
        height=120,
    )
    portfolio_pairs = []
    for line in custom_input.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [t.strip() for t in line.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            portfolio_pairs.append((parts[0], parts[1]))
    if portfolio_pairs:
        st.sidebar.success(f"{len(portfolio_pairs)} pair(s) defined.")
    elif custom_input.strip():
        st.sidebar.error("Invalid format. Use: TICKER_Y, TICKER_X (one pair per line).")

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
initial_capital = st.sidebar.number_input("Initial capital", min_value=1_000, value=100_000, step=5_000)
rf_annual = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")

include_sp500 = st.sidebar.checkbox("Include S&P 500 benchmark", value=True, help="Fetch S&P 500 via yfinance for the test date range.")

if not portfolio_pairs:
    st.warning("Select at least one pair in the sidebar.")
    st.stop()

@st.cache_data(show_spinner="Fetching price data from LSEG...")
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return get_close_prices(list(tickers), start=start, end=end)

unique_tickers = sorted({t for pair in portfolio_pairs for t in pair})
train_prices = load_prices(tuple(unique_tickers), str(train_start), str(train_end))
test_prices = load_prices(tuple(unique_tickers), str(test_start), str(test_end))

# Optional S&P 500 (yfinance)
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
tab_coint, tab_spread, tab_returns, tab_bt, tab_compare, tab_opt = st.tabs([
    "Cointegration",
    "Spread & Z-Scores",
    "Return Estimation",
    "Backtest Results",
    "Strategy vs Benchmarks",
    "Portfolio Optimisation",
])

# ===== TAB 1: Cointegration =====
with tab_coint:
    st.header("Cointegration screening (training period)")
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

# ===== TAB 2: Spread & Z-Scores =====
with tab_spread:
    st.header("Spread & Z-Scores")
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
        st.caption(f"Rolling B = {hr_latest:.4f}, a = {int_latest:.4f}  |  Static B = {hr_static:.4f}, a = {intercept_static:.4f}")
        import plotly.graph_objects as go
        hr_fig = go.Figure()
        hr_fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df["hedge_ratio"], name="Rolling B", line=dict(color="#636EFA")))
        hr_fig.add_hline(y=hr_static, line=dict(color="red", dash="dash"), annotation_text=f"Static B = {hr_static:.4f}")
        hr_fig.update_layout(yaxis_title="Hedge ratio (B)", height=280, margin=dict(t=20, b=20))
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

# ===== TAB 3: Return Estimation (core FYP output) =====
with tab_returns:
    st.header("Estimated Expected Returns")
    st.caption(
        "Comparing return estimates derived from spread predictions (OU model) "
        "against traditional historical mean approaches — the core research question."
    )

    if coint_pairs.empty:
        st.warning("No cointegrated pairs found — cannot estimate returns.")
    else:
        # Build spread returns for training period
        train_spread_rets = build_spread_return_matrix(train_prices, coint_pairs)
        train_asset_rets = train_prices[
            list(dict.fromkeys(coint_pairs["y"].tolist() + coint_pairs["x"].tolist()))
        ].pct_change().dropna(how="all")

        # ------------------------------------------------------------------
        # 1. Spread-based return estimates (all three methods)
        # ------------------------------------------------------------------
        st.subheader("1. Spread-based return estimates (training period)")

        ou_mu = build_ou_expected_returns(
            train_prices, coint_pairs, window=lookback, annualise=True,
        )
        hist_mu = historical_mean_return(train_spread_rets, annualise=True)
        ewma_mu = ewma_return(train_spread_rets, span=lookback, annualise=True)

        # Friendly pair labels
        pair_labels = [f"{r['y']} / {r['x']}" for _, r in coint_pairs.iterrows()]
        pair_keys = [f"{r['y']}_vs_{r['x']}" for _, r in coint_pairs.iterrows()]

        estimates_df = pd.DataFrame({
            "OU-implied (spread prediction)": ou_mu.reindex(pair_keys).values,
            "Historical mean": hist_mu.reindex(pair_keys).values,
            "EWMA": ewma_mu.reindex(pair_keys).values,
        }, index=pair_labels)

        st.plotly_chart(
            plot_return_estimates_comparison(estimates_df),
            use_container_width=True,
        )
        st.dataframe(
            estimates_df.style.format("{:.2%}"),
            use_container_width=True,
        )

        # ------------------------------------------------------------------
        # 2. Detailed per-pair statistics
        # ------------------------------------------------------------------
        st.subheader("2. Per-pair spread statistics")

        detail_rows = []
        for _, row in coint_pairs.iterrows():
            label = f"{row['y']} / {row['x']}"
            key = f"{row['y']}_vs_{row['x']}"
            sp = compute_spread(
                train_prices[row["y"]], train_prices[row["x"]],
                row["hedge_ratio"], row.get("intercept", 0.0),
            )
            hl = compute_half_life(sp)
            hurst = compute_hurst_exponent(sp)
            zs = compute_zscore(sp, window=lookback)
            detail_rows.append({
                "Pair": label,
                "Hedge ratio": row["hedge_ratio"],
                "Half-life (days)": hl,
                "Hurst exponent": hurst,
                "Current z-score": float(zs.iloc[-1]) if not np.isnan(zs.iloc[-1]) else np.nan,
                "OU E[r] (ann.)": ou_mu.get(key, 0.0),
                "Hist. E[r] (ann.)": hist_mu.get(key, 0.0),
            })
        detail_df = pd.DataFrame(detail_rows).set_index("Pair")
        st.dataframe(
            detail_df.style
            .format("{:.4f}", subset=["Hedge ratio"])
            .format("{:.1f}", subset=["Half-life (days)"])
            .format("{:.3f}", subset=["Hurst exponent", "Current z-score"])
            .format("{:.2%}", subset=["OU E[r] (ann.)", "Hist. E[r] (ann.)"]),
            use_container_width=True,
        )

        st.caption(
            "Half-life < 30 days and Hurst < 0.5 indicate strong mean reversion. "
            "The OU model exploits this to predict expected returns from the current "
            "spread deviation."
        )

        # ------------------------------------------------------------------
        # 3. Spread-based vs Traditional asset-level estimates
        # ------------------------------------------------------------------
        st.subheader("3. Spread-based vs Traditional (asset-level) estimates")
        st.caption(
            "Traditional MPT uses historical mean returns of individual assets. "
            "Spread-based MPT uses OU-implied returns from cointegrated pair spreads."
        )

        asset_hist_mu = historical_mean_return(train_asset_rets, annualise=True)

        col_sp, col_tr = st.columns(2)
        with col_sp:
            st.markdown("**Spread-based estimates (OU-implied)**")
            sp_df = pd.DataFrame({
                "Pair": pair_labels,
                "E[r] (ann.)": ou_mu.reindex(pair_keys).values,
            })
            st.dataframe(
                sp_df.style.format("{:.2%}", subset=["E[r] (ann.)"]),
                hide_index=True, use_container_width=True,
            )
        with col_tr:
            st.markdown("**Traditional estimates (historical mean)**")
            tr_df = pd.DataFrame({
                "Asset": asset_hist_mu.index,
                "E[r] (ann.)": asset_hist_mu.values,
            })
            st.dataframe(
                tr_df.style.format("{:.2%}", subset=["E[r] (ann.)"]),
                hide_index=True, use_container_width=True,
            )

        # ------------------------------------------------------------------
        # 4. Rolling stability of return estimates
        # ------------------------------------------------------------------
        st.subheader("4. Rolling return estimate stability")
        st.caption(
            "Shows how noisy the return estimate is over time. "
            "More stable estimates lead to more reliable portfolio allocations."
        )

        pair_sel_ret = st.selectbox(
            "Select pair to examine", pair_labels, key="ret_stability_pair",
        )
        sel_key = pair_keys[pair_labels.index(pair_sel_ret)]

        if sel_key in train_spread_rets.columns:
            st.plotly_chart(
                plot_rolling_return_estimate(
                    train_spread_rets[sel_key], window=lookback,
                    pair_label=pair_sel_ret,
                ),
                use_container_width=True,
            )

        # ------------------------------------------------------------------
        # 5. Covariance comparison
        # ------------------------------------------------------------------
        st.subheader("5. Covariance matrix (spread returns)")

        cov_method = st.radio(
            "Covariance estimator",
            ["Sample", "Ledoit-Wolf shrinkage"],
            horizontal=True, key="cov_method",
        )
        if cov_method == "Ledoit-Wolf shrinkage":
            cov_mat = shrinkage_covariance(train_spread_rets.dropna(), annualise=True)
        else:
            cov_mat = sample_covariance(train_spread_rets.dropna(), annualise=True)

        cov_display = cov_mat.copy()
        cov_display.index = pair_labels
        cov_display.columns = pair_labels
        st.dataframe(cov_display.style.format("{:.6f}"), use_container_width=True)

# ===== TAB 4: Backtest Results (enhanced) =====
with tab_bt:
    st.header("Backtest results (test period)")
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

        # Metrics
        st.subheader("Strategy performance")
        st.dataframe(format_metrics_table(result.metrics), use_container_width=True, hide_index=True)

        # Benchmarks
        benchmarks = build_all_benchmarks(
            test_prices, coint_pairs, rf_annual=rf_annual, market_prices=sp500_prices,
        )
        b_cols = [k for k in ["buy_hold_pair", "risk_free", "equal_weight_pairs", "sp500"] if k in benchmarks]

        st.subheader("Benchmark comparison")
        bench_metrics = []
        for name in b_cols:
            ret = benchmarks[name]
            m = compute_benchmark_metrics(ret, rf_annual=rf_annual, benchmark_returns=benchmarks.get("buy_hold_pair"))
            m["name"] = name
            bench_metrics.append(m)
        bench_df = pd.DataFrame(bench_metrics).set_index("name")
        bench_df = bench_df[["sharpe_ratio", "max_drawdown", "total_return", "annualised_volatility", "volatility_reduction"]]
        bench_df.columns = ["Sharpe", "Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]
        st.dataframe(
            bench_df.style.format("{:.2f}", subset=["Sharpe"])
            .format("{:.2%}", subset=["Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]),
            use_container_width=True,
        )

        # Cumulative returns
        st.subheader("Cumulative returns")
        series_for_chart = {"Strategy (spread-based)": result.daily_returns}
        label_map = {"risk_free": "Risk-free", "buy_hold_pair": "B&H (pair)", "equal_weight_pairs": "Equal-weight pairs", "sp500": "S&P 500"}
        for k in b_cols:
            series_for_chart[label_map.get(k, k)] = benchmarks[k]
        st.plotly_chart(plot_cumulative_returns_multi(series_for_chart), use_container_width=True)

        # Drawdown
        st.subheader("Drawdown")
        st.plotly_chart(plot_drawdown(result.daily_returns), use_container_width=True)

        # Rolling Sharpe
        st.subheader("Rolling Sharpe ratio")
        st.plotly_chart(
            plot_rolling_sharpe(result.daily_returns, window=lookback, rf_annual=rf_annual),
            use_container_width=True,
        )

        # Returns distribution
        st.subheader("Daily returns distribution")
        dist_series = {"Strategy": result.daily_returns}
        if "buy_hold_pair" in benchmarks:
            dist_series["B&H (pair)"] = benchmarks["buy_hold_pair"]
        st.plotly_chart(plot_returns_distribution(dist_series), use_container_width=True)

        # Position timeline + trade log
        st.subheader("Position timeline")
        st.plotly_chart(plot_position_timeline(result.positions["position"]), use_container_width=True)
        st.subheader("Trade log")
        if not result.trades.empty:
            st.dataframe(result.trades, use_container_width=True)
        else:
            st.info("No trades in the test period.")

# ===== TAB 5: Strategy vs Benchmarks =====
with tab_compare:
    st.header("Spread-based strategy vs Historical MPT vs Benchmarks")
    st.caption(
        "Side-by-side comparison: spread-based z-score strategy vs traditional "
        "Markowitz (max-Sharpe from training data) vs passive benchmarks."
    )
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
        compare_df = pd.DataFrame(rows).set_index("Strategy")
        compare_df = compare_df[["sharpe_ratio", "max_drawdown", "total_return", "annualised_volatility", "volatility_reduction"]]
        compare_df.columns = ["Sharpe", "Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]
        st.dataframe(
            compare_df.style.format("{:.2f}", subset=["Sharpe"])
            .format("{:.2%}", subset=["Max DD", "Total return", "Ann. vol", "Vol reduction vs B&H"]),
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

        # Drawdown comparison
        st.subheader("Drawdown comparison")
        dd_col1, dd_col2 = st.columns(2)
        with dd_col1:
            st.markdown("**Spread-based strategy**")
            st.plotly_chart(plot_drawdown(strategy_ret), use_container_width=True)
        with dd_col2:
            st.markdown("**Historical MPT**")
            st.plotly_chart(plot_drawdown(mpt_ret), use_container_width=True)

# ===== TAB 6: Portfolio Optimisation =====
with tab_opt:
    st.header("Portfolio Optimisation: Spread-Based vs Traditional MPT")
    st.caption(
        "Compares portfolio weights derived from spread-based return estimates "
        "(OU-implied) against traditional historical mean returns."
    )
    if coint_pairs.empty:
        st.warning("No cointegrated pairs — cannot optimise.")
    else:
        # Build spread return matrix for all cointegrated pairs
        spread_returns_df = build_spread_return_matrix(test_prices, coint_pairs)

        if spread_returns_df.shape[1] < 2:
            st.info("Need at least 2 cointegrated pairs for the frontier.")
            st.dataframe(spread_returns_df.describe(), use_container_width=True)
        else:
            # --- Return estimation method selector ---
            ret_method = st.radio(
                "Return estimation method (for spread-based MPT)",
                ["OU-implied (spread prediction)", "Historical mean"],
                horizontal=True,
            )

            if ret_method == "OU-implied (spread prediction)":
                spread_mu = build_ou_expected_returns(
                    test_prices, coint_pairs, window=lookback, annualise=False,
                )
                spread_mu = spread_mu.reindex(spread_returns_df.columns).fillna(0.0)
                custom_mu = spread_mu.values
            else:
                custom_mu = None

            # --- Spread-based MPT ---
            st.subheader("Spread-based efficient frontier")
            mv_spread = mean_variance_weights(
                spread_returns_df, expected_returns=custom_mu, rf_annual=rf_annual,
            )
            frontier_spread = efficient_frontier(
                spread_returns_df, expected_returns=custom_mu, rf_annual=rf_annual,
            )
            max_sharpe_pt = (mv_spread["max_sharpe_vol"], mv_spread["max_sharpe_return"])
            min_var_pt = (mv_spread["min_var_vol"], mv_spread["min_var_return"])
            st.plotly_chart(
                plot_efficient_frontier(frontier_spread, max_sharpe_pt, min_var_pt),
                use_container_width=True,
            )

            # Volatility reduction
            bench_ret = spread_returns_df.mean(axis=1)
            port_ret_sharpe = (spread_returns_df * mv_spread["max_sharpe_weights"]).sum(axis=1)
            port_ret_minvar = (spread_returns_df * mv_spread["min_var_weights"]).sum(axis=1)
            mask = port_ret_sharpe.notna() & bench_ret.notna()
            vol_red_sharpe = compute_volatility_reduction(port_ret_sharpe[mask], bench_ret[mask])
            vol_red_minvar = compute_volatility_reduction(port_ret_minvar[mask], bench_ret[mask])
            st.subheader("Portfolio-level volatility reduction")
            col1, col2 = st.columns(2)
            col1.metric("Max-Sharpe vs equal-weight", f"{vol_red_sharpe:.1%}")
            col2.metric("Min-variance vs equal-weight", f"{vol_red_minvar:.1%}")

            # Weights comparison
            st.subheader("Weights: spread-based vs traditional MPT")
            mv_traditional = mean_variance_weights(
                spread_returns_df, expected_returns=None, rf_annual=rf_annual,
            )

            opt_pair_labels = [c.replace("_vs_", " / ") for c in spread_returns_df.columns]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Spread-based Max-Sharpe**")
                w_df = pd.DataFrame({"Pair": opt_pair_labels, "Weight": mv_spread["max_sharpe_weights"]})
                w_df["Weight"] = w_df["Weight"].map("{:.2%}".format)
                st.dataframe(w_df, hide_index=True)
            with col_b:
                st.markdown("**Traditional Max-Sharpe (historical mean)**")
                w_df = pd.DataFrame({"Pair": opt_pair_labels, "Weight": mv_traditional["max_sharpe_weights"]})
                w_df["Weight"] = w_df["Weight"].map("{:.2%}".format)
                st.dataframe(w_df, hide_index=True)

            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown("**Spread-based Min-Variance**")
                w_df = pd.DataFrame({"Pair": opt_pair_labels, "Weight": mv_spread["min_var_weights"]})
                w_df["Weight"] = w_df["Weight"].map("{:.2%}".format)
                st.dataframe(w_df, hide_index=True)
            with col_d:
                st.markdown("**Traditional Min-Variance**")
                w_df = pd.DataFrame({"Pair": opt_pair_labels, "Weight": mv_traditional["min_var_weights"]})
                w_df["Weight"] = w_df["Weight"].map("{:.2%}".format)
                st.dataframe(w_df, hide_index=True)

            # Performance summary
            st.subheader("Performance summary")
            def _sharpe(ret, vol):
                return f"{(ret - rf_annual) / vol:.2f}" if vol > 0 else "N/A"
            summary_rows = [
                {"Portfolio": "Spread-based Max-Sharpe",   "Ann. Return": f"{mv_spread['max_sharpe_return']:.2%}",    "Ann. Vol": f"{mv_spread['max_sharpe_vol']:.2%}",    "Sharpe": _sharpe(mv_spread["max_sharpe_return"], mv_spread["max_sharpe_vol"])},
                {"Portfolio": "Traditional Max-Sharpe",    "Ann. Return": f"{mv_traditional['max_sharpe_return']:.2%}", "Ann. Vol": f"{mv_traditional['max_sharpe_vol']:.2%}", "Sharpe": _sharpe(mv_traditional["max_sharpe_return"], mv_traditional["max_sharpe_vol"])},
                {"Portfolio": "Spread-based Min-Variance", "Ann. Return": f"{mv_spread['min_var_return']:.2%}",       "Ann. Vol": f"{mv_spread['min_var_vol']:.2%}",       "Sharpe": _sharpe(mv_spread["min_var_return"], mv_spread["min_var_vol"])},
                {"Portfolio": "Traditional Min-Variance",  "Ann. Return": f"{mv_traditional['min_var_return']:.2%}",    "Ann. Vol": f"{mv_traditional['min_var_vol']:.2%}",    "Sharpe": _sharpe(mv_traditional["min_var_return"], mv_traditional["min_var_vol"])},
            ]
            st.dataframe(pd.DataFrame(summary_rows).set_index("Portfolio"), use_container_width=True)
