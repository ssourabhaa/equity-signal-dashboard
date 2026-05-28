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
    st.caption("Momentum 12-1 signal · Spearman IC analysis")

    with st.spinner("Loading returns and computing signal..."):
        ret_wide = load_returns_wide(con)
        mom_raw = compute_momentum(ret_wide)
        mom_z = cross_sectional_zscore(mom_raw)
        mom_rank = mom_z.rank(axis=1, pct=True)

        # Compute IC series (21-day forward return horizon)
        ic_series = compute_ic_series(mom_z, ret_wide, horizon=21)
        ic_series["rolling_ic"] = ic_series["ic"].rolling(63).mean()  # 3-month rolling

    # Metric cards
    col1, col2, col3 = st.columns(3)
    mean_ic = ic_series["ic"].mean()
    icir = ic_series["ic"].mean() / ic_series["ic"].std()
    col1.metric("Mean IC", f"{mean_ic:.4f}")
    col2.metric("ICIR", f"{icir:.2f}")
    col3.metric("Signal observations", f"{int(ic_series['ic'].notna().sum())}")

    # Rolling IC chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ic_series.index, y=ic_series["ic"],
        mode="lines", name="Daily IC",
        line=dict(color="#cbd5e1", width=0.8)
    ))
    fig1.add_trace(go.Scatter(
        x=ic_series.index, y=ic_series["rolling_ic"],
        mode="lines", name="63-day rolling IC",
        line=dict(color="#3b82f6", width=2)
    ))
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig1.update_layout(title="Rolling IC — Momentum 12-1", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # IC histogram
    fig2 = px.histogram(
        ic_series.dropna(), x="ic", nbins=40,
        title="IC Distribution",
        color_discrete_sequence=["#6366f1"]
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

    # Latest z-scores (bar chart)
    latest = mom_z.dropna(how="all").iloc[-1].sort_values(ascending=False)
    fig3 = px.bar(
        x=latest.index, y=latest.values,
        title="Latest Momentum Z-Score by Ticker",
        color=latest.values,
        color_continuous_scale="RdYlGn",
        labels={"x": "Ticker", "y": "Z-Score"}
    )
    st.plotly_chart(fig3, use_container_width=True)

elif view == "Portfolio":
    st.title("Portfolio")
    st.info("Coming in V2 — walk-forward backtest, cvxpy optimiser.")

elif view == "Risk & Factors":
    st.title("Risk & Factors")
    st.info("Coming in V2 — VaR, ES, Fama-French 3-factor.")

elif view == "Model Comparison":
    st.title("Model Comparison")
    st.info("Coming in V4 — LightGBM vs LSTM vs GNN covariance vs FinBERT.")