import pandas as pd
import numpy as np
import time
from datetime import datetime
import yfinance as yf
from transformers import pipeline
import torch


def load_finbert():
    """
    ProsusAI/finbert: BERT fine-tuned on financial news.
    Labels: positive, negative, neutral.
    """
    print("Loading FinBERT model...")
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=0 if torch.cuda.is_available() else -1,
        max_length=512,
        truncation=True,
    )


def fetch_news_for_ticker(ticker_symbol, max_headlines=20):
    """
    Fetch recent news headlines for a ticker via yfinance.
    Returns list of dicts: [{title, timestamp}, ...]
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news or []
        results = []
        for item in news[:max_headlines]:
            # yfinance news format varies — handle both old and new schemas
            content = item.get("content", item)
            title = content.get("title", "")
            ts = (content.get("pubDate")
                  or item.get("providerPublishTime")
                  or 0)

            if isinstance(ts, str):
                try:
                    ts = pd.Timestamp(ts).timestamp()
                except Exception:
                    ts = 0

            if title and ts:
                results.append({
                    "title": title,
                    "timestamp": datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else ts,
                })
        return results
    except Exception as e:
        print(f"  News fetch failed for {ticker_symbol}: {e}")
        return []


def score_sentiment(headlines, finbert):
    """
    Run FinBERT on a list of headlines.
    Returns a single sentiment score:
    - positive → +score
    - negative → -score
    - neutral  → 0
    """
    if not headlines:
        return 0.0

    texts = [h["title"] for h in headlines]
    results = finbert(texts)

    scores = []
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def build_sentiment_signal(tickers, finbert, lookback_days=30):
    """
    Build sentiment scores for all tickers.

    Note: yfinance.news only returns recent news (~2 weeks).
    For full historical backtesting you'd use EDGAR or a news archive.
    This gives a live/recent signal suitable for demonstration.

    Returns: pd.DataFrame, rows=date, cols=tickers, values=sentiment_score
    """
    records = []

    for i, ticker in enumerate(tickers):
        print(f"  Fetching news: {ticker} ({i+1}/{len(tickers)})")
        headlines = fetch_news_for_ticker(ticker)
        score = score_sentiment(headlines, finbert)

        today = datetime.now().date()
        records.append({
            "date": today,
            "ticker": ticker,
            "sentiment_raw": score,
        })

        time.sleep(0.5)  # polite rate limiting

    df = pd.DataFrame(records)
    signal_wide = df.pivot(index="date", columns="ticker", values="sentiment_raw")
    signal_wide.index = pd.to_datetime(signal_wide.index)
    return signal_wide


def write_sentiment_to_db(signal_wide, z_wide, rank_wide, con):
    """Write FinBERT sentiment signal to DuckDB signals table."""
    from src.signals import write_signals_to_db
    write_signals_to_db(signal_wide, z_wide, rank_wide, "finbert_sentiment", con)
    print("FinBERT sentiment signal written to DB.")