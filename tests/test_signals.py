"""Basic sanity tests for signals, schema, and FinBERT."""
import pandas as pd
import numpy as np
import pytest

from src.db import get_connection, create_schema
from src.signals import (compute_momentum, compute_ic_series,
                         cross_sectional_zscore)


@pytest.fixture
def sample_returns():
    """Synthetic returns: 500 days × 10 tickers, random walk."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=500)
    tickers = [f"T{i}" for i in range(10)]
    data = np.random.normal(0, 0.01, size=(500, 10))
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_schema_creation():
    """Schema creation should run without error and create all tables."""
    con = get_connection()
    create_schema(con)
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='main' ORDER BY table_name
    """).fetchall()
    table_names = {row[0] for row in tables}
    assert {"prices", "universe", "signals", "weights"}.issubset(table_names)
    con.close()


def test_momentum_shape(sample_returns):
    """Momentum signal should return same shape as input returns."""
    mom = compute_momentum(sample_returns)
    assert mom.shape == sample_returns.shape


def test_momentum_no_lookahead(sample_returns):
    """
    Momentum at date t should only use data from before t.
    Test: zero out future returns; momentum should still be the same up to date t.
    """
    mom_full = compute_momentum(sample_returns)
    truncated = sample_returns.copy()
    truncated.iloc[400:] = 0.0
    mom_truncated = compute_momentum(truncated)

    # Momentum value at date 300 must be identical in both cases
    pd.testing.assert_series_equal(
        mom_full.iloc[300], mom_truncated.iloc[300], check_names=False
    )


def test_zscore_properties(sample_returns):
    """Cross-sectional z-score should have ~0 mean and ~1 std per row."""
    z = cross_sectional_zscore(sample_returns)
    row = z.iloc[300].dropna()
    assert abs(row.mean()) < 0.01
    assert abs(row.std() - 1.0) < 0.1


def test_ic_series_horizon(sample_returns):
    """IC at date t should compare signal[t] with future returns [t to t+horizon]."""
    signal = compute_momentum(sample_returns)
    ic = compute_ic_series(signal, sample_returns, horizon=21)
    assert "ic" in ic.columns
    assert ic["ic"].abs().max() <= 1.0  # IC is a correlation


def test_finbert_score_range():
    """FinBERT sentiment scores should be in [-1, 1]."""
    from src.ml.finbert_signal import score_sentiment, load_finbert
    finbert = load_finbert()
    headlines = [
        {"title": "Apple beats earnings expectations with record revenue"},
        {"title": "Tesla shares plunge on disappointing delivery numbers"},
        {"title": "The quick brown fox jumps over the lazy dog"},
    ]
    score = score_sentiment(headlines, finbert)
    assert -1.0 <= score <= 1.0