"""Test the Google News fetcher for one ticker over 2 months."""
from datetime import datetime, timedelta
from src.news import fetch_news_for_ticker_chunked

ticker = "AAPL"
end_date = datetime.utcnow()
start_date = end_date - timedelta(weeks=9)

print(f"Fetching {ticker} news from {start_date.date()} to {end_date.date()}...")
print(f"(Chunked into weekly windows to avoid Google's per-query cap.)\n")

headlines = fetch_news_for_ticker_chunked(ticker, start_date, end_date)

print(f"Total headlines fetched: {len(headlines)}\n")

if headlines:
    print("First 3 headlines:")
    for h in headlines[:3]:
        print(f"  [{h['published_at'].strftime('%Y-%m-%d')}] {h['source']}")
        print(f"  -> {h['title'][:100]}\n")

    print("Last 3 headlines:")
    for h in headlines[-3:]:
        print(f"  [{h['published_at'].strftime('%Y-%m-%d')}] {h['source']}")
        print(f"  -> {h['title'][:100]}\n")

    import pandas as pd
    dates = pd.Series([h["published_at"].date() for h in headlines])
    print(f"Date range covered: {dates.min()} to {dates.max()}")
    print(f"Unique days with coverage: {dates.nunique()}")
