import yfinance as yf
import pandas as pd
from curl_cffi import requests as cffi_requests
from src.db import get_connection, create_schema

# 20-stock universe — 4 per sector
UNIVERSE_V1 = [
    ("AAPL", "Technology"), ("MSFT", "Technology"),
    ("NVDA", "Technology"), ("GOOGL", "Technology"),
    ("JPM", "Financials"), ("BAC", "Financials"),
    ("GS", "Financials"), ("MS", "Financials"),
    ("JNJ", "Healthcare"), ("UNH", "Healthcare"),
    ("PFE", "Healthcare"), ("ABBV", "Healthcare"),
    ("AMZN", "Cons.Discr."), ("HD", "Cons.Discr."),
    ("MCD", "Cons.Discr."), ("NKE", "Cons.Discr."),
    ("CAT", "Industrials"), ("HON", "Industrials"),
    ("UPS", "Industrials"), ("DE", "Industrials"),
]

UNIVERSE_V2 = UNIVERSE_V1 + [
    ("TSLA", "Technology"), ("AMD", "Technology"), ("INTC", "Technology"), ("ORCL", "Technology"),
    ("WFC", "Financials"), ("C", "Financials"), ("AXP", "Financials"), ("BLK", "Financials"),
    ("MRK", "Healthcare"), ("BMY", "Healthcare"), ("GILD", "Healthcare"), ("CVS", "Healthcare"),
    ("TGT", "Cons.Discr."), ("LOW", "Cons.Discr."), ("SBUX", "Cons.Discr."), ("GM", "Cons.Discr."),
    ("GE", "Industrials"), ("LMT", "Industrials"), ("RTX", "Industrials"), ("BA", "Industrials"),
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"), ("SLB", "Energy"),
    ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"), ("D", "Utilities"),
    ("AMT", "Real Estate"), ("PLD", "Real Estate"), ("EQIX", "Real Estate"), ("PSA", "Real Estate"),
]

START_DATE = "2014-01-01"


def seed_universe(con, universe):
    """Insert ticker list into universe table. Skip if already exists."""
    for ticker, sector in universe:
        con.execute("""
            INSERT OR IGNORE INTO universe (ticker, sector, inclusion_flag)
            VALUES (?, ?, TRUE)
        """, [ticker, sector])
    print(f"Universe seeded: {len(universe)} tickers")


def download_and_upsert(con, tickers, start=START_DATE):
    """
    Download price history from yfinance and upsert into prices table.
    'Upsert' = insert if new, replace if already exists (idempotent — safe to re-run)
    """
    ticker_list = [t for t, _ in tickers]
    print(f"Downloading {len(ticker_list)} tickers from {start}...")

    # auto_adjust=False gives us separate 'Adj Close' and 'Close' columns
    # curl_cffi session impersonates a browser — avoids Yahoo rate limits
    session = cffi_requests.Session(impersonate="chrome")
    raw = yf.download(
        ticker_list, start=start, auto_adjust=False,
        progress=False, session=session
    )

    # raw has MultiIndex columns: (field, ticker) e.g. ('Adj Close', 'AAPL')
    # Stack reshapes wide → long format
    df = raw.stack(level=1, future_stack=True).reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Rename to match our schema
    df = df.rename(columns={
        "level_1": "ticker",  # or "ticker" depending on yfinance version
        "adj_close": "adjusted_close",
        "date": "date",
    })

    # Handle yfinance column name variations
    if "level_1" in df.columns:
        df = df.rename(columns={"level_1": "ticker"})

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["adjusted_close"])

    # Upsert row by row (simple; fast enough for this size)
    inserted = 0
    for _, row in df.iterrows():
        con.execute("""
            INSERT OR REPLACE INTO prices
            (date, ticker, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["date"], row["ticker"],
            row.get("open"), row.get("high"),
            row.get("low"), row.get("close"),
            row["adjusted_close"], row.get("volume")
        ])
        inserted += 1

    print(f"Upserted {inserted} rows into prices.")


if __name__ == "__main__":
    con = get_connection()
    create_schema(con)
    seed_universe(con, UNIVERSE_V1)
    download_and_upsert(con, UNIVERSE_V1)

    # Quick sanity check
    count = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"Total rows in prices: {count}")
    con.close()