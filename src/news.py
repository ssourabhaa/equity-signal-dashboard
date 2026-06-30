"""Google News RSS fetcher for historical news headlines per ticker."""
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import pandas as pd

RSS_SLEEP_SECONDS = 1.0
MAX_RSS_ENTRIES_PER_WINDOW = 100


def fetch_news_for_ticker(
    ticker: str,
    period_start: datetime,
    period_end: datetime,
) -> list[dict]:
    """Fetch news headlines for a ticker in a given date window."""
    after = period_start.strftime("%Y-%m-%d")
    before = period_end.strftime("%Y-%m-%d")

    q = f"{ticker} after:{after} before:{before}"
    url = (
        f"https://news.google.com/rss/search?"
        f"q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(url)
    headlines = []

    period_start_ts = pd.Timestamp(period_start, tz="UTC")
    period_end_ts = pd.Timestamp(period_end, tz="UTC") + pd.Timedelta(days=1)

    for entry in feed.entries[:MAX_RSS_ENTRIES_PER_WINDOW]:
        published = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if not published:
            continue

        published_ts = pd.to_datetime(published, utc=True, errors="coerce")
        if pd.isna(published_ts):
            continue

        if not (period_start_ts <= published_ts <= period_end_ts):
            continue

        source = ""
        if hasattr(entry, "source") and isinstance(entry.source, dict):
            source = entry.source.get("title", "")

        headlines.append({
            "title": getattr(entry, "title", ""),
            "snippet": getattr(entry, "summary", ""),
            "source": source or "Google News RSS",
            "url": getattr(entry, "link", ""),
            "published_at": published_ts.to_pydatetime(),
        })

    time.sleep(RSS_SLEEP_SECONDS)
    return headlines


def fetch_news_for_ticker_chunked(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    window_days: int = 7,
) -> list[dict]:
    """Fetch news in chunks (Google RSS caps results per query at ~100)."""
    headlines = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=window_days), end_date)
        chunk = fetch_news_for_ticker(ticker, current, chunk_end)
        headlines.extend(chunk)
        current = chunk_end

    seen_urls = set()
    deduped = []
    for h in headlines:
        if h["url"] and h["url"] not in seen_urls:
            seen_urls.add(h["url"])
            deduped.append(h)

    return deduped
