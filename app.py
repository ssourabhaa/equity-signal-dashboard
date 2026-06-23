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

def check_password():
    """Simple shared-password gate. Returns True if user is authenticated."""
    APP_PASSWORD = "12345"  

    if st.session_state.get("authenticated"):
        return True

    st.title("Equity Signal Dashboard")
    st.caption("Enter the password to continue.")
    pw = st.text_input("Password", type="password")

    if st.button("Sign in"):
        if pw == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

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
        ["momentum", "mean_rev", "lgbm", "lstm", "finbert_sentiment"]
    )

    universe_tickers = con.execute(
        "SELECT ticker FROM universe ORDER BY ticker"
    ).df()["ticker"].tolist()

    stock_choice = st.sidebar.selectbox(
        "Company / Stock",
        ["All stocks"] + universe_tickers,
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

        if stock_choice != "All stocks":
            st.subheader(f"{stock_choice} — {signal_choice} signal")
            if stock_choice in sig_wide.columns:
                stock_series = sig_wide[stock_choice].dropna()
                latest_row = sig_wide.dropna(how="all").iloc[-1]
                latest_val = stock_series.iloc[-1] if not stock_series.empty else float("nan")
                pct_rank = latest_row.rank(pct=True).get(stock_choice, float("nan")) * 100

                c1, c2 = st.columns(2)
                c1.metric("Latest z-score", f"{latest_val:.2f}")
                c2.metric("Latest rank (percentile)", f"{pct_rank:.0f}%")

                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(
                    x=stock_series.index, y=stock_series.values,
                    mode="lines", name=f"{stock_choice} z-score",
                    line=dict(color="#10b981", width=1.5)))
                fig_stock.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_stock.update_layout(
                    title=f"{stock_choice} — {signal_choice} z-score over time",
                    height=300)
                st.plotly_chart(fig_stock, use_container_width=True)
            else:
                st.info(f"No '{signal_choice}' signal data for {stock_choice}.")

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

        st.subheader("Signal Benchmark — All Signals")
        type_map = {
            "momentum": "Classical",
            "mean_rev": "Classical",
            "lgbm": "ML",
            "lstm": "ML",
            "finbert_sentiment": "NLP/SOTA",
        }
        benchmark_rows = []
        for name in ["momentum", "mean_rev", "lgbm", "lstm", "finbert_sentiment"]:
            df_s = con.execute("""
                SELECT date, ticker, zscore FROM signals
                WHERE signal_name = ? ORDER BY date
            """, [name]).df()
            if df_s.empty:
                continue
            s_wide = df_s.pivot(index="date", columns="ticker", values="zscore")
            s_wide.index = pd.to_datetime(s_wide.index)

            # IC needs multiple dates — FinBERT only has 1 day, so IC isn't defined
            if len(s_wide) > 30:
                ic = compute_ic_series(s_wide, ret_wide, horizon=21)["ic"]
                mean_ic = ic.mean()
                icir = ic.mean() / ic.std() if ic.std() > 0 else 0
            else:
                mean_ic = float("nan")
                icir = float("nan")

            benchmark_rows.append({
                "Signal": name,
                "Mean IC": mean_ic,
                "ICIR": icir,
                "Type": type_map.get(name, "?"),
                "Observations (days)": len(s_wide),
            })
        bench_df = pd.DataFrame(benchmark_rows)
        st.dataframe(bench_df.style.format({"Mean IC": "{:.4f}", "ICIR": "{:.2f}"}))
        st.caption("Note: FinBERT sentiment shows N/A because yfinance.news only returns ~2 weeks of recent headlines — there's no historical depth to compute a Spearman IC. See README for limitations.")

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
        # 4-signal composite — note FinBERT has limited history (yfinance ~2wk)
        comp_z = composite_signal(
            {"momentum": mom_z, "mean_rev": rev_z},
            weights={"momentum": 0.55, "mean_rev": 0.45},
        )

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
    st.caption("Sample covariance vs GNN-learned covariance")

    import os
    from src.backtest import compute_performance_metrics

    if not (os.path.exists("data/bt_sample.parquet") and
            os.path.exists("data/bt_gnn.parquet")):
        st.warning("Run `python -m scripts.compare_cov` first.")
        st.stop()

    bt_sample = pd.read_parquet("data/bt_sample.parquet")
    bt_gnn = pd.read_parquet("data/bt_gnn.parquet")

    m_sample = compute_performance_metrics(bt_sample["portfolio_return"])
    m_gnn = compute_performance_metrics(bt_gnn["portfolio_return"])

    st.subheader("Performance Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe",  f"{m_gnn['sharpe']:.3f}",
                f"{m_gnn['sharpe'] - m_sample['sharpe']:+.3f} vs sample")
    col2.metric("Sortino", f"{m_gnn['sortino']:.3f}",
                f"{m_gnn['sortino'] - m_sample['sortino']:+.3f} vs sample")
    col3.metric("Max Drawdown", f"{m_gnn['max_drawdown']:.2%}",
                f"{(m_gnn['max_drawdown'] - m_sample['max_drawdown'])*100:+.2f} ppts")

    # Side-by-side metrics table
    st.subheader("Full Comparison")
    table = pd.DataFrame({
        "Metric": ["Sharpe", "Sortino", "Max Drawdown"],
        "Sample Cov": [m_sample["sharpe"], m_sample["sortino"], f"{m_sample['max_drawdown']:.2%}"],
        "GNN Cov":    [m_gnn["sharpe"], m_gnn["sortino"], f"{m_gnn['max_drawdown']:.2%}"],
    })
    st.dataframe(table, hide_index=True)

    # Cumulative return overlay
    cum_sample = (1 + bt_sample["portfolio_return"]).cumprod()
    cum_gnn = (1 + bt_gnn["portfolio_return"]).cumprod()

    overlay = pd.DataFrame({"Sample Cov": cum_sample, "GNN Cov": cum_gnn})
    fig = px.line(overlay, title="Cumulative Return — Sample vs GNN Covariance")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown overlay
    peak_s, peak_g = cum_sample.expanding().max(), cum_gnn.expanding().max()
    dd = pd.DataFrame({
        "Sample Cov": (cum_sample - peak_s) / peak_s,
        "GNN Cov":    (cum_gnn - peak_g) / peak_g,
    })
    fig2 = px.line(dd, title="Drawdown — Sample vs GNN Covariance")
    st.plotly_chart(fig2, use_container_width=True)