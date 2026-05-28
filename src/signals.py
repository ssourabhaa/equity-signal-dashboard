import pandas as pd
import numpy as np
from scipy import stats
from src.db import get_connection


def load_returns_wide(con):
    """
    Pull log returns from DuckDB and pivot to wide format.
    Returns a DataFrame where:
    - rows = dates
    - columns = ticker symbols
    - values = log returns
    """
    df = con.execute("""
        SELECT date, ticker, log_return
        FROM returns
        WHERE log_return IS NOT NULL
        ORDER BY date
    """).df()
    ret_wide = df.pivot(index="date", columns="ticker", values="log_return")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    return ret_wide


def cross_sectional_zscore(df):
    """
    Standardise each row (date) across all tickers.
    Result: each day, signals have mean=0 and std=1 across stocks.
    This removes market-wide effects and makes signals comparable.
    """
    return df.apply(lambda row: stats.zscore(row, nan_policy="omit"), axis=1)


def compute_momentum(ret_wide):
    """
    12-1 month momentum:
    - 252 trading days = ~12 months
    - .shift(21) skips the most recent 21 days (~1 month) to avoid short-term reversal
    - We look at cumulative return from 252 days ago to 21 days ago
    """
    # Cumulative return over 252 days, lagged by 21 days
    price_approx = ret_wide.cumsum()              # approximate price in log space
    raw = price_approx - price_approx.shift(252)  # 12-month return
    raw = raw.shift(21)                            # skip last month
    return raw


def compute_ic_series(signal_df, returns_df, horizon=21):
    """
    Compute Spearman IC for each date:
    - signal_df: today's signal (rows=dates, cols=tickers)
    - returns_df: actual future returns (we shift backwards by horizon)
    - IC = Spearman correlation between signal and future returns

    Positive IC = signal predicts direction correctly.
    """
    forward_returns = returns_df.shift(-horizon)  # future returns
    ic_series = []

    for date in signal_df.index:
        if date not in forward_returns.index:
            continue
        sig = signal_df.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()
        common = sig.index.intersection(fwd.index)

        if len(common) < 5:
            continue

        ic, _ = stats.spearmanr(sig[common], fwd[common])
        ic_series.append({"date": date, "ic": ic})

    return pd.DataFrame(ic_series).set_index("date")


def write_signals_to_db(raw_df, z_df, rank_df, signal_name, con):
    """Write signal scores to DuckDB signals table."""
    dates = z_df.index
    tickers = z_df.columns

    for date in dates:
        for ticker in tickers:
            raw = raw_df.loc[date, ticker] if ticker in raw_df.columns else None
            z = z_df.loc[date, ticker] if ticker in z_df.columns else None
            rank = rank_df.loc[date, ticker] if ticker in rank_df.columns else None

            if pd.isna(z):
                continue

            con.execute("""
                INSERT OR REPLACE INTO signals
                (date, ticker, signal_name, raw_score, zscore, rank_pct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                date.date(), ticker, signal_name,
                float(raw) if raw else None,
                float(z),
                float(rank) if rank else None,
            ])