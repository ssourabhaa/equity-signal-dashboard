"""
One-time training of ML signals. Writes z-scored predictions to DuckDB.
Run with: python -m scripts.train_ml_signals
"""
import pandas as pd
from src.db import get_connection, create_schema
from src.signals import (load_returns_wide, cross_sectional_zscore,
                         write_signals_to_db)
from src.ml.lgbm_signal import train_lgbm_signal
from src.ml.lstm_signal import train_lstm_signal


def train_and_store(signal_name, signal_wide, con):
    z_wide = cross_sectional_zscore(signal_wide)
    rank_wide = z_wide.rank(axis=1, pct=True)
    write_signals_to_db(signal_wide, z_wide, rank_wide, signal_name, con)
    print(f"  Wrote {signal_name} to signals table.")


if __name__ == "__main__":
    con = get_connection()
    create_schema(con)
    ret_wide = load_returns_wide(con)

    print("\n=== Training LightGBM ===")
    lgbm_raw = train_lgbm_signal(ret_wide)
    train_and_store("lgbm", lgbm_raw, con)

    print("\n=== Training LSTM ===")
    lstm_raw = train_lstm_signal(ret_wide, epochs=10, batch_size=128)
    train_and_store("lstm", lstm_raw, con)

    print("\nAlso writing classical signals for completeness...")
    from src.signals import compute_momentum, compute_mean_reversion
    mom_raw = compute_momentum(ret_wide)
    rev_raw = compute_mean_reversion(ret_wide)
    train_and_store("momentum", mom_raw, con)
    train_and_store("mean_rev", rev_raw, con)

    con.close()
    print("\nDone.")