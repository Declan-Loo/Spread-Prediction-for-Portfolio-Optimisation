"""
Plotly chart-builder functions for the pairs-trading dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_spread_with_bands(
    spread: pd.Series,
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> go.Figure:
    """
    2-panel subplot:
      Top — spread with rolling mean ± 2σ bands.
      Bottom — z-score with entry/exit threshold lines.
    """
    window = 60
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std(ddof=1)
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Spread with ±2σ Bands", "Z-Score"),
        row_heights=[0.55, 0.45],
    )

    # --- Top panel: Spread ---
    fig.add_trace(
        go.Scatter(x=spread.index, y=spread, name="Spread",
                   line=dict(color="#636EFA")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=rolling_mean.index, y=rolling_mean, name="Mean",
                   line=dict(color="grey", dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=upper.index, y=upper, name="+2σ",
                   line=dict(color="rgba(255,0,0,0.3)"), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=lower.index, y=lower, name="-2σ",
                   line=dict(color="rgba(255,0,0,0.3)"), showlegend=False,
                   fill="tonexty", fillcolor="rgba(255,0,0,0.05)"),
        row=1, col=1,
    )

    # --- Bottom panel: Z-score ---
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore, name="Z-Score",
                   line=dict(color="#EF553B")),
        row=2, col=1,
    )
    for val, label, color in [
        (entry_z, f"+Entry ({entry_z})", "red"),
        (-entry_z, f"-Entry ({-entry_z})", "red"),
        (exit_z, f"Exit ({exit_z})", "green"),
    ]:
        fig.add_hline(
            y=val, row=2, col=1,
            line=dict(color=color, dash="dash", width=1),
            annotation_text=label, annotation_position="right",
        )

    fig.update_layout(height=600, margin=dict(t=40, b=20))
    return fig


def plot_cointegration_results(screening_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of p-values per pair.
    Green bars for significant (< 0.05), red for non-significant.
    """
    df = screening_df.copy().sort_values("p_value", ascending=True)
    df["pair"] = df["y"] + " / " + df["x"]
    colours = [
        "#2CA02C" if p < 0.05 else "#D62728" for p in df["p_value"]
    ]

    fig = go.Figure(go.Bar(
        x=df["p_value"],
        y=df["pair"],
        orientation="h",
        marker_color=colours,
        text=df["p_value"].round(4),
        textposition="outside",
    ))
    fig.add_vline(
        x=0.05,
        line=dict(color="black", dash="dash", width=1.5),
        annotation_text="α = 0.05",
    )
    fig.update_layout(
        xaxis_title="p-value",
        yaxis_title="Pair",
        height=max(300, len(df) * 45),
        margin=dict(l=120),
    )
    return fig


def plot_cumulative_returns_multi(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative returns",
) -> go.Figure:
    """
    Plot multiple daily-return series as normalised cumulative returns (start=1).
    """
    fig = go.Figure()
    colours = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
    for i, (name, ret) in enumerate(returns_dict.items()):
        r = ret.dropna()
        if r.empty:
            continue
        cum = (1 + r).cumprod()
        cum = cum / cum.iloc[0]
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=name, line=dict(color=colours[i % len(colours)]),
        ))
    fig.update_layout(
        title=title,
        yaxis_title="Cumulative return (normalised)",
        height=400,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_cumulative_returns(
    strategy: pd.Series,
    y_bh: pd.Series,
    x_bh: pd.Series,
) -> go.Figure:
    """
    3 normalised cumulative-return lines:
    strategy vs buy-hold of each leg.
    """
    fig = go.Figure()
    for series, name, colour in [
        (strategy, "Strategy", "#636EFA"),
        (y_bh, "Y Buy & Hold", "#EF553B"),
        (x_bh, "X Buy & Hold", "#00CC96"),
    ]:
        normalised = series / series.iloc[0]
        fig.add_trace(go.Scatter(
            x=normalised.index, y=normalised,
            name=name, line=dict(color=colour),
        ))
    fig.update_layout(
        yaxis_title="Cumulative Return (normalised)",
        height=400,
        margin=dict(t=30, b=20),
    )
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    max_sharpe_pt: tuple[float, float] | None = None,
    min_var_pt: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Scatter plot of the efficient frontier coloured by Sharpe ratio.
    Star markers for max-Sharpe and min-variance portfolios.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"],
        y=frontier_df["return"],
        mode="markers",
        marker=dict(
            color=frontier_df["sharpe"],
            colorscale="Viridis",
            colorbar=dict(title="Sharpe"),
            size=6,
        ),
        name="Frontier",
    ))
    if max_sharpe_pt:
        fig.add_trace(go.Scatter(
            x=[max_sharpe_pt[0]], y=[max_sharpe_pt[1]],
            mode="markers",
            marker=dict(symbol="star", size=16, color="gold",
                        line=dict(width=1, color="black")),
            name="Max Sharpe",
        ))
    if min_var_pt:
        fig.add_trace(go.Scatter(
            x=[min_var_pt[0]], y=[min_var_pt[1]],
            mode="markers",
            marker=dict(symbol="star", size=16, color="cyan",
                        line=dict(width=1, color="black")),
            name="Min Variance",
        ))
    fig.update_layout(
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        height=450,
        margin=dict(t=30, b=20),
    )
    return fig


def plot_zscore_heatmap(zscore_dict: dict[str, pd.Series]) -> go.Figure:
    """
    Heatmap of z-scores across all pairs over time.

    Parameters
    ----------
    zscore_dict : dict mapping pair label → z-score Series.
    """
    df = pd.DataFrame(zscore_dict)
    fig = go.Figure(go.Heatmap(
        z=df.T.values,
        x=df.index,
        y=df.columns,
        colorscale="RdYlGn_r",
        zmid=0,
        colorbar=dict(title="Z-Score"),
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Pair",
        height=max(300, len(df.columns) * 50),
        margin=dict(l=120, t=30, b=20),
    )
    return fig


def plot_position_timeline(positions: pd.Series) -> go.Figure:
    """
    Step chart showing +1 / 0 / −1 positions over time.
    """
    fig = go.Figure(go.Scatter(
        x=positions.index,
        y=positions.values,
        mode="lines",
        line=dict(shape="hv", color="#AB63FA", width=2),
        name="Position",
    ))
    fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1))
    fig.update_layout(
        yaxis=dict(
            title="Position",
            tickvals=[-1, 0, 1],
            ticktext=["Short (−1)", "Flat (0)", "Long (+1)"],
            range=[-1.3, 1.3],
        ),
        height=300,
        margin=dict(t=20, b=20),
    )
    return fig


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """
    Format a backtest metrics dict into a styled DataFrame for display.
    """
    fmt = {
        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', float('nan')):.2f}",
        "Total Return": f"{metrics.get('total_return', 0):.2%}",
        "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
        "Ann. Volatility": f"{metrics.get('annualised_volatility', 0):.2%}",
        "Volatility Reduction": f"{metrics.get('volatility_reduction', float('nan')):.1%}" if pd.notna(metrics.get('volatility_reduction')) else "N/A",
        "Num Trades": str(metrics.get("num_trades", 0)),
        "Win Rate": f"{metrics.get('win_rate', float('nan')):.1%}",
        "Profit Factor": f"{metrics.get('profit_factor', float('nan')):.2f}",
    }
    return pd.DataFrame(
        list(fmt.items()), columns=["Metric", "Value"]
    )


# ---------------------------------------------------------------------------
# Return estimation charts
# ---------------------------------------------------------------------------


def plot_return_estimates_comparison(
    estimates_df: pd.DataFrame,
) -> go.Figure:
    """
    Grouped bar chart comparing return estimates across methods.

    Parameters
    ----------
    estimates_df : pd.DataFrame
        Index = pair labels, columns = estimation methods
        (e.g. "OU-implied", "Historical mean", "EWMA").
        Values are annualised expected returns.
    """
    fig = go.Figure()
    colours = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for i, col in enumerate(estimates_df.columns):
        fig.add_trace(go.Bar(
            x=estimates_df.index,
            y=estimates_df[col],
            name=col,
            marker_color=colours[i % len(colours)],
            text=estimates_df[col].map(lambda v: f"{v:.2%}"),
            textposition="outside",
        ))
    fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1))
    fig.update_layout(
        barmode="group",
        yaxis_title="Annualised expected return",
        yaxis_tickformat=".1%",
        height=400,
        margin=dict(t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_rolling_return_estimate(
    returns: pd.Series,
    window: int = 60,
    pair_label: str = "",
) -> go.Figure:
    """
    Rolling annualised mean return over time — shows estimation stability.
    """
    rolling_mu = returns.rolling(window).mean() * 252
    ewma_mu = returns.ewm(span=window).mean() * 252

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_mu.index, y=rolling_mu.values,
        name=f"Rolling {window}d mean",
        line=dict(color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=ewma_mu.index, y=ewma_mu.values,
        name=f"EWMA (span={window})",
        line=dict(color="#EF553B", dash="dash"),
    ))
    fig.add_hline(
        y=float(returns.mean() * 252),
        line=dict(color="grey", dash="dot", width=1),
        annotation_text="Full-sample mean",
        annotation_position="right",
    )
    fig.update_layout(
        title=f"Rolling return estimate: {pair_label}" if pair_label else None,
        yaxis_title="Annualised return estimate",
        yaxis_tickformat=".1%",
        height=350,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Backtest charts
# ---------------------------------------------------------------------------


def plot_drawdown(daily_returns: pd.Series) -> go.Figure:
    """
    Underwater / drawdown chart from a daily return series.
    """
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy",
        fillcolor="rgba(239,85,59,0.2)",
        line=dict(color="#EF553B", width=1),
        name="Drawdown",
    ))
    fig.update_layout(
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        height=300,
        margin=dict(t=20, b=20),
    )
    return fig


def plot_returns_distribution(
    returns_dict: dict[str, pd.Series],
) -> go.Figure:
    """
    Overlaid histograms of daily return distributions.
    """
    fig = go.Figure()
    colours = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, (name, ret) in enumerate(returns_dict.items()):
        r = ret.dropna()
        if r.empty:
            continue
        fig.add_trace(go.Histogram(
            x=r.values,
            name=name,
            marker_color=colours[i % len(colours)],
            opacity=0.6,
            nbinsx=80,
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Daily return",
        xaxis_tickformat=".2%",
        yaxis_title="Frequency",
        height=350,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 60,
    rf_annual: float = 0.02,
) -> go.Figure:
    """
    Rolling Sharpe ratio over time.
    """
    rf_daily = rf_annual / 252
    excess = daily_returns - rf_daily
    rolling_mean = excess.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std(ddof=1)
    rolling_sr = (rolling_mean / rolling_std) * np.sqrt(252)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sr.index, y=rolling_sr.values,
        line=dict(color="#636EFA"),
        name=f"Rolling {window}d Sharpe",
    ))
    fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1))
    fig.update_layout(
        yaxis_title="Sharpe ratio (annualised)",
        height=300,
        margin=dict(t=20, b=20),
    )
    return fig
