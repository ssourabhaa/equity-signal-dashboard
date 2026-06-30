"""Run the Gemini news sentiment signal end-to-end.

For each ticker in the configured universe:
  - Fetch news for the configured time window via Google News RSS
  - Batch-score headlines with Gemini (FinBERT-style: label + signed score)
  - Aggregate to one daily sentiment value per (ticker, date)
  - Cross-sectionally z-score across the universe on each date
  - Write to DuckDB `signals` table as `gemini_news`

Cached per-headline scores are kept so re-runs don't waste API calls.
"""
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.db import get_connection, create_schema
from src.news import fetch_news_for_ticker_chunked
from src.ml.gemini_sentiment import score_all_headlines


# ── Configuration (top-of-file knobs — easy to expand later) ─────────────
TICKERS = [
       "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA",  # Tech (existing)
       "META", "AMD", "ORCL",                              # Tech (new)
       "JPM", "BAC", "WFC", "GS", "BLK",                   # Finance
       "JNJ", "UNH", "PFE", "MRK",                         # Healthcare
       "BA", "CAT", "HON",                                 # Industrial
       "XOM", "CVX", "COP",                                # Energy
   ]
WEEKS_BACK = 5
SIGNAL_NAME = "gemini_news"
REFRESH = "--refresh" in sys.argv


# ── Storage layer ────────────────────────────────────────────────────────
def init_headline_cache(con):
    """Cache table for per-headline scores — avoids re-spending API calls on re-runs."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS gemini_news_headlines (
            ticker VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            published_at TIMESTAMP NOT NULL,
            title VARCHAR,
            source VARCHAR,
            label VARCHAR,
            score DOUBLE,
            signed_score DOUBLE,
            scored_at TIMESTAMP NOT NULL,
            PRIMARY KEY (ticker, url)
        )
    """)


def get_cached_urls(con, ticker: str) -> set:
    """Return URLs already scored for this ticker."""
    rows = con.execute(
        "SELECT url FROM gemini_news_headlines WHERE ticker = ?", [ticker]
    ).fetchall()
    return {row[0] for row in rows}


def upsert_headline(con, ticker, headline_dict, score_result):
    """Insert or replace a scored headline."""
    con.execute("""
        INSERT OR REPLACE INTO gemini_news_headlines
        (ticker, url, published_at, title, source, label, score, signed_score, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ticker,
        headline_dict["url"],
        headline_dict["published_at"],
        headline_dict["title"][:500],
        headline_dict["source"][:200],
        score_result["label"],
        score_result["score"],
        score_result["signed_score"],
        datetime.now(timezone.utc),
    ])


def load_all_scored(con, tickers: list[str]) -> pd.DataFrame:
    """Load every cached scored headline for the configured tickers."""
    placeholders = ",".join("?" * len(tickers))
    df = con.execute(f"""
        SELECT ticker, published_at, signed_score
        FROM gemini_news_headlines
        WHERE ticker IN ({placeholders})
        AND signed_score IS NOT NULL
    """, tickers).df()
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["date"] = df["published_at"].dt.tz_convert("UTC").dt.date
    return df


def aggregate_to_daily(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Mean signed score per (ticker, date). NaN naturally for missing days."""
    if scored_df.empty:
        return pd.DataFrame()
    daily = (
        scored_df.groupby(["date", "ticker"])["signed_score"]
        .mean()
        .reset_index()
        .rename(columns={"signed_score": "raw_score"})
    )
    return daily


def cross_sectional_zscore(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-date z-score across tickers. Returns long-format with raw + zscore."""
    if daily.empty:
        return daily
    wide = daily.pivot(index="date", columns="ticker", values="raw_score")
    z = wide.apply(lambda row: stats.zscore(row, nan_policy="omit"), axis=1)
    z_long = (
        z.reset_index().melt(id_vars="date", var_name="ticker", value_name="zscore")
    )
    merged = daily.merge(z_long, on=["date", "ticker"])
    # Percentile rank for completeness
    rank = wide.rank(axis=1, pct=True)
    rank_long = (
        rank.reset_index().melt(id_vars="date", var_name="ticker", value_name="rank_pct")
    )
    merged = merged.merge(rank_long, on=["date", "ticker"])
    return merged


def write_signal_to_db(con, daily_signal: pd.DataFrame, signal_name: str):
    """Write aggregated (ticker, date, score) rows into the `signals` table."""
    for _, row in daily_signal.iterrows():
        if pd.isna(row["zscore"]):
            continue
        con.execute("""
            INSERT OR REPLACE INTO signals
            (date, ticker, signal_name, raw_score, zscore, rank_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            row["date"],
            row["ticker"],
            signal_name,
            float(row["raw_score"]),
            float(row["zscore"]),
            float(row["rank_pct"]) if not pd.isna(row["rank_pct"]) else None,
        ])


# ── Main pipeline ────────────────────────────────────────────────────────
def main():
    con = get_connection()
    create_schema(con)
    init_headline_cache(con)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(weeks=WEEKS_BACK)

    print(f"Universe: {TICKERS}")
    print(f"Window: {start_date.date()} to {end_date.date()} ({WEEKS_BACK} weeks)")
    print(f"Refresh mode: {REFRESH}\n")

    api_calls_used = 0
    total_new_headlines = 0
    total_scored = 0

    pbar = tqdm(TICKERS, desc="Scoring tickers", unit="ticker")
    for ticker in pbar:
        try:
            # Fetch headlines from Google News RSS
            pbar.set_postfix_str(f"{ticker}: fetching news")
            all_headlines = fetch_news_for_ticker_chunked(
                ticker, start_date.replace(tzinfo=None), end_date.replace(tzinfo=None)
            )

            # Filter to ones not already scored (unless --refresh)
            cached_urls = set() if REFRESH else get_cached_urls(con, ticker)
            new_headlines = [h for h in all_headlines if h["url"] not in cached_urls]
            total_new_headlines += len(new_headlines)

            if not new_headlines:
                pbar.set_postfix_str(f"{ticker}: all cached")
                continue

            # Score with Gemini (batched internally)
            pbar.set_postfix_str(f"{ticker}: scoring {len(new_headlines)} via Gemini")
            titles = [h["title"] for h in new_headlines]
            results = score_all_headlines(titles, ticker)
            # Rough API call counter (batch size = 75)
            api_calls_used += -(-len(new_headlines) // 75)

            # Write each successfully scored headline to the cache
            for h, r in zip(new_headlines, results):
                if r is not None:
                    upsert_headline(con, ticker, h, r)
                    total_scored += 1

        except Exception as e:
            print(f"\n  {ticker} failed: {type(e).__name__}: {e}")

    print(f"\n{'='*60}\nFETCH + SCORE COMPLETE\n{'='*60}")
    print(f"New headlines fetched: {total_new_headlines}")
    print(f"Successfully scored:   {total_scored}")
    print(f"Approx API calls used: {api_calls_used}")

    # ── Aggregate, z-score, write to signals table ─────────────────────
    print("\nAggregating to daily signal and writing to `signals` table...")
    scored_df = load_all_scored(con, TICKERS)
    if scored_df.empty:
        print("  No scored headlines available — nothing to aggregate.")
        con.close()
        return

    daily = aggregate_to_daily(scored_df)
    daily_z = cross_sectional_zscore(daily)
    write_signal_to_db(con, daily_z, SIGNAL_NAME)

    print(f"  Written: {len(daily_z)} (ticker, date) rows as signal '{SIGNAL_NAME}'")
    print(f"  Date range: {daily_z['date'].min()} to {daily_z['date'].max()}")
    print(f"  Tickers with data: {daily_z['ticker'].nunique()}")

    # Per-ticker coverage summary
    print("\nPer-ticker coverage:")
    coverage = daily_z.groupby("ticker").agg(
        days=("date", "nunique"),
        avg_signed=("raw_score", "mean"),
    ).round(3)
    print(coverage.to_string())

    con.close()


if __name__ == "__main__":
    main()
