import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.db import get_connection, create_schema
from src.signals import (
    load_returns_wide, compute_momentum,
    cross_sectional_zscore, compute_ic_series
)

st.set_page_config(page_title="Equity Signal Dashboard", layout="wide")


@st.cache_resource
def get_con():
    """Cache DB connection — only opens once per Streamlit session."""
    con = get_connection()
    create_schema(con)
    return con


con = get_con()

# Sidebar navigation
view = st.sidebar.radio(
    "View",
    ["Signal Dashboard", "Portfolio", "Risk & Factors", "Model Comparison"]
)

# ── VIEW 1: Signal Dashboard ──────────────────────────────────────────────────
if view == "Signal Dashboard":
    st.title("Signal Dashboard")
    st.caption("4 signals · Spearman IC analysis")

    from src.signals import compute_ic_series

    signal_choice = st.sidebar.selectbox(
        "Signal",
        ["momentum", "mean_rev", "lgbm", "lstm"]
    )

    with st.spinner("Loading signal and computing IC..."):
        ret_wide = load_returns_wide(con)

        df = con.execute("""
            SELECT date, ticker, zscore
            FROM signals
            WHERE signal_name = ?
            ORDER BY date
        """, [signal_choice]).df()

        if df.empty:
            st.warning(f"No data for signal '{signal_choice}'. Run scripts/train_ml_signals.py first.")
            st.stop()

        sig_wide = df.pivot(index="date", columns="ticker", values="zscore")
        sig_wide.index = pd.to_datetime(sig_wide.index)

        ic_series = compute_ic_series(sig_wide, ret_wide, horizon=21)
        ic_series["rolling_ic"] = ic_series["ic"].rolling(63).mean()

    col1, col2, col3 = st.columns(3)
    mean_ic = ic_series["ic"].mean()
    icir = ic_series["ic"].mean() / ic_series["ic"].std()
    col1.metric("Mean IC", f"{mean_ic:.4f}")
    col2.metric("ICIR", f"{icir:.2f}")
    col3.metric("Observations", f"{int(ic_series['ic'].notna().sum())}")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ic_series.index, y=ic_series["ic"],
                              mode="lines", name="Daily IC",
                              line=dict(color="#cbd5e1", width=0.8)))
    fig1.add_trace(go.Scatter(x=ic_series.index, y=ic_series["rolling_ic"],
                              mode="lines", name="63-day rolling IC",
                              line=dict(color="#3b82f6", width=2)))
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig1.update_layout(title=f"Rolling IC — {signal_choice}", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(ic_series.dropna(), x="ic", nbins=40,
                       title="IC Distribution",
                       color_discrete_sequence=["#6366f1"])
    fig2.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Signal Benchmark — All 4 Signals")
    benchmark_rows = []
    for name in ["momentum", "mean_rev", "lgbm", "lstm"]:
        df_s = con.execute("""
            SELECT date, ticker, zscore FROM signals
            WHERE signal_name = ? ORDER BY date
        """, [name]).df()
        if df_s.empty:
            continue
        s_wide = df_s.pivot(index="date", columns="ticker", values="zscore")
        s_wide.index = pd.to_datetime(s_wide.index)
        ic = compute_ic_series(s_wide, ret_wide, horizon=21)["ic"]
        benchmark_rows.append({
            "Signal": name,
            "Mean IC": ic.mean(),
            "ICIR": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "Type": "Classical" if name in ("momentum", "mean_rev") else "ML",
        })
    bench_df = pd.DataFrame(benchmark_rows)
    st.dataframe(bench_df.style.format({"Mean IC": "{:.4f}", "ICIR": "{:.2f}"}))

elif view == "Portfolio":
    st.title("Portfolio")
    from src.backtest import (walk_forward_backtest,
                              compute_performance_metrics)
    from src.signals import (load_returns_wide, compute_momentum,
                             compute_mean_reversion, cross_sectional_zscore,
                             composite_signal)

    with st.spinner("Running walk-forward backtest..."):
        ret_wide = load_returns_wide(con)
        mom_raw = compute_momentum(ret_wide)
        rev_raw = compute_mean_reversion(ret_wide)
        mom_z = cross_sectional_zscore(mom_raw)
        rev_z = cross_sectional_zscore(rev_raw)
        comp_z = composite_signal({"momentum": mom_z, "mean_rev": rev_z})

        backtest = walk_forward_backtest(comp_z, ret_wide)
        metrics = compute_performance_metrics(backtest["portfolio_return"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    col2.metric("Sortino", f"{metrics['sortino']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")

    cum_ret = (1 + backtest["portfolio_return"]).cumprod()
    fig = px.line(cum_ret, title="Cumulative Return — MV Optimised Portfolio")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret - peak) / peak
    fig2 = px.area(drawdown, title="Drawdown", color_discrete_sequence=["#ef4444"])
    st.plotly_chart(fig2, use_container_width=True)

elif view == "Risk & Factors":
    st.title("Risk & Factors")
    from src.backtest import walk_forward_backtest
    from src.signals import (load_returns_wide, compute_momentum,
                             compute_mean_reversion, cross_sectional_zscore,
                             composite_signal)
    from src.risk import (historical_var, expected_shortfall,
                          load_ff3_factors, ff3_regression)

    with st.spinner("Running backtest + risk analysis..."):
        ret_wide = load_returns_wide(con)
        mom_z = cross_sectional_zscore(compute_momentum(ret_wide))
        rev_z = cross_sectional_zscore(compute_mean_reversion(ret_wide))
        comp_z = composite_signal({"momentum": mom_z, "mean_rev": rev_z})
        backtest = walk_forward_backtest(comp_z, ret_wide)
        port_rets = backtest["portfolio_return"]

        var_99 = historical_var(port_rets, 0.99)
        es_99 = expected_shortfall(port_rets, 0.99)

    st.subheader("Tail Risk (daily returns)")
    col1, col2 = st.columns(2)
    col1.metric("Historical VaR 99%", f"{var_99:.2%}")
    col2.metric("Expected Shortfall 99%", f"{es_99:.2%}")

    st.subheader("Fama-French 3-Factor Exposure")
    ff3 = load_ff3_factors()
    if ff3 is not None:
        port_rets.index = pd.to_datetime(port_rets.index)
        result = ff3_regression(port_rets, ff3)
        col1, col2, col3 = st.columns(3)
        col1.metric("Alpha (annualised)", f"{result['alpha_annualised']:.2%}")
        col2.metric("β Market", f"{result['beta_mkt']:.2f}")
        col3.metric("R²", f"{result['r_squared']:.2f}")
        col4, col5 = st.columns(2)
        col4.metric("β SMB (size)", f"{result['beta_smb']:.2f}")
        col5.metric("β HML (value)", f"{result['beta_hml']:.2f}")
    else:
        st.warning("Could not download Fama-French factors. Check internet connection.")

elif view == "Model Comparison":
    st.title("Model Comparison")
    st.info("Coming in V4 — LightGBM vs LSTM vs GNN covariance vs FinBERT.")