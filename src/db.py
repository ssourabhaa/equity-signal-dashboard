import duckdb
from pathlib import Path

DB_PATH = Path("data/signals.duckdb")


def get_connection():
    """Return a DuckDB connection. Creates the file if it doesn't exist."""
    DB_PATH.parent.mkdir(exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def create_schema(con):
    """
    Create all tables. Safe to run multiple times — IF NOT EXISTS prevents errors.

    Tables:
    - prices: raw OHLCV data per stock per day
    - returns: VIEW that auto-computes log returns from prices
    - universe: the list of stocks we track
    - signals: every signal value for every stock for every day
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date            DATE        NOT NULL,
            ticker          VARCHAR     NOT NULL,
            open            DOUBLE,
            high            DOUBLE,
            low             DOUBLE,
            close           DOUBLE,
            adjusted_close  DOUBLE      NOT NULL,
            volume          BIGINT,
            ingested_at     TIMESTAMP   DEFAULT current_timestamp,
            PRIMARY KEY (date, ticker)
        )
    """)

    # VIEW: recomputes log returns every query — always in sync with prices
    # LN(today_price / yesterday_price) = log return
    # LAG() gets yesterday's price for the same ticker
    con.execute("""
        CREATE VIEW IF NOT EXISTS returns AS
        SELECT
            date,
            ticker,
            adjusted_close,
            LN(adjusted_close / LAG(adjusted_close)
                OVER (PARTITION BY ticker ORDER BY date)) AS log_return
        FROM prices
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS universe (
            ticker          VARCHAR     PRIMARY KEY,
            sector          VARCHAR,
            inclusion_flag  BOOLEAN     DEFAULT TRUE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            date            DATE        NOT NULL,
            ticker          VARCHAR     NOT NULL,
            signal_name     VARCHAR     NOT NULL,
            raw_score       DOUBLE,
            zscore          DOUBLE,
            rank_pct        DOUBLE,
            PRIMARY KEY (date, ticker, signal_name)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS weights (
            rebal_date DATE NOT NULL,
            ticker VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            weight DOUBLE,
            signal_score DOUBLE,
            prev_weight DOUBLE,
            PRIMARY KEY (rebal_date, ticker, run_id)
        )
    """)

    print("Schema created successfully.")


if __name__ == "__main__":
    con = get_connection()
    create_schema(con)
    con.close()