import numpy as np
import pandas as pd
import cvxpy as cp
from src.db import get_connection


def mv_optimise(signal_scores, ret_window, gamma=1.0, custom_cov=None):
    """
    Mean-Variance optimisation using cvxpy.

    signal_scores: pd.Series, index=tickers, values=z-scores (expected return proxy)
    ret_window: pd.DataFrame, rows=dates, cols=tickers (historical returns for covariance)
    gamma: risk-aversion parameter (higher = more conservative)
    custom_cov: optional precomputed covariance matrix (used in V4 for GNN covariance)

    Returns: pd.Series of weights, index=tickers
    """
    tickers = signal_scores.dropna().index
    tickers = tickers.intersection(ret_window.columns)
    n = len(tickers)

    if n < 5:
        return pd.Series(1.0 / n, index=tickers)

    mu = signal_scores[tickers].values

    if custom_cov is not None:
        Sigma = custom_cov
    else:
        Sigma = ret_window[tickers].cov().values

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)

    objective = cp.Maximize(ret - gamma * risk)
    constraints = [
        cp.sum(w) == 1,
        w >= -0.05,
        w <= 0.10,
        cp.norm1(w) <= 1.6,
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
    except Exception:
        return pd.Series(1.0 / n, index=tickers)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return pd.Series(1.0 / n, index=tickers)

    return pd.Series(w.value, index=tickers)


def walk_forward_backtest(composite_z, ret_wide, cost_bps=5, lookback=252, gamma=1.0):
    """
    Monthly walk-forward backtest with transaction costs.
    """
    results = []
    prev_w = pd.Series(dtype=float)

    all_dates = composite_z.index
    rebal_dates = all_dates[lookback::21]

    for rebal_date in rebal_dates:
        signal = composite_z.loc[rebal_date].dropna()
        ret_window = ret_wide.loc[:rebal_date].tail(lookback)

        weights = mv_optimise(signal, ret_window, gamma=gamma)

        common = weights.index.intersection(prev_w.index)
        turnover = (weights[common] - prev_w[common]).abs().sum()
        cost = turnover * cost_bps / 10000

        idx = all_dates.get_loc(rebal_date)
        next_idx = min(idx + 21, len(all_dates) - 1)
        next_date = all_dates[next_idx]

        period_rets = ret_wide.loc[rebal_date:next_date, weights.index]
        period_portfolio = (period_rets * weights).sum(axis=1)

        for date, r in period_portfolio.items():
            results.append({"date": date, "portfolio_return": r - cost / 21})

        prev_w = weights

    df = pd.DataFrame(results).set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return df


def compute_performance_metrics(returns_series):
    """Sharpe, Sortino, max drawdown from a daily return series."""
    r = returns_series.dropna()
    ann = 252

    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0

    neg = r[r < 0]
    sortino = r.mean() / neg.std() * np.sqrt(ann) if neg.std() > 0 else 0

    cum = (1 + r).cumprod()
    peak = cum.expanding().max()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    return {"sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd}