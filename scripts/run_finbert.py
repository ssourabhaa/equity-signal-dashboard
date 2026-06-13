"""
Score sentiment for all 50 tickers and write to DuckDB.
Run with: python -m scripts.run_finbert
"""
from src.db import get_connection
from src.signals import cross_sectional_zscore
from src.ml.finbert_signal import (load_finbert, build_sentiment_signal,
                                    write_sentiment_to_db)
from src.ingest import UNIVERSE_V2


if __name__ == "__main__":
    tickers = [t for t, _ in UNIVERSE_V2]
    con = get_connection()

    finbert = load_finbert()

    print("Building sentiment signal for 50 tickers...")
    raw_wide = build_sentiment_signal(tickers, finbert)

    z_wide = cross_sectional_zscore(raw_wide)
    rank_wide = z_wide.rank(axis=1, pct=True)

    write_sentiment_to_db(raw_wide, z_wide, rank_wide, con)
    print("Done.")
    con.close()