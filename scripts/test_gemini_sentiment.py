"""Quick test of Gemini sentiment scoring on a small batch of real AAPL headlines."""
from datetime import datetime, timedelta
from src.news import fetch_news_for_ticker_chunked
from src.ml.gemini_sentiment import score_headlines_batch

# Get ~10 recent AAPL headlines
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=7)

print("Fetching ~10 recent AAPL headlines...")
all_headlines = fetch_news_for_ticker_chunked("AAPL", start_date, end_date)
sample = all_headlines[:10]

print(f"Got {len(sample)} headlines. Sending to Gemini...\n")

titles = [h["title"] for h in sample]
results = score_headlines_batch(titles, "AAPL")

print("--- RESULTS ---")
print(f"{'#':<3} {'LABEL':<10} {'SCORE':<8} {'SIGNED':<8} HEADLINE")
print("-" * 100)
for i, (h, r) in enumerate(zip(sample, results)):
    title = h["title"][:70]
    if r is None:
        print(f"{i+1:<3} {'(parse failure)':<28} {title}")
    else:
        print(f"{i+1:<3} {r['label']:<10} {r['score']:<8.2f} {r['signed_score']:<+8.2f} {title}")

# Summary
successful = [r for r in results if r is not None]
if successful:
    avg_signed = sum(r["signed_score"] for r in successful) / len(successful)
    print(f"\nAvg signed sentiment: {avg_signed:+.3f}")
    print(f"Parse success: {len(successful)}/{len(results)}")
