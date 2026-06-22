"""
Build FinBERT sentiment signal from the FNSPID dataset (CC BY-NC 4.0 — non-commercial).
Filters to your universe, scores each headline, aggregates per (date, ticker), writes to DB.
"""
import pandas as pd
import time
from huggingface_hub import hf_hub_download
from src.db import get_connection, create_schema
from src.signals import cross_sectional_zscore, write_signals_to_db
from src.ml.finbert_signal import load_finbert

# ── 1. Universe from your DB ────────────────────────────────────────────────
con = get_connection()
create_schema(con)
universe = set(con.execute("SELECT ticker FROM universe").df()["ticker"].tolist())
print(f"Universe: {len(universe)} tickers — {sorted(universe)}")

# ── 2. Download the FNSPID news CSV (~22 GB; one-time, then cached) ─────────
print("Downloading FNSPID news CSV... (one-time, large file)")
csv_path = hf_hub_download(
    repo_id="Zihan1004/FNSPID",
    filename="Stock_news/nasdaq_exteral_data.csv",
    repo_type="dataset",
)
print(f"CSV at: {csv_path}")

# ── 3. Stream-read in chunks; keep only universe rows ───────────────────────
print("Filtering to universe...")
rows = []
chunk_iter = pd.read_csv(
    csv_path,
    usecols=["Date", "Article_title", "Stock_symbol"],
    chunksize=200_000,
    on_bad_lines="skip",
    low_memory=False,
)
for i, chunk in enumerate(chunk_iter):
    keep = chunk[chunk["Stock_symbol"].isin(universe)].copy()
    if len(keep):
        rows.append(keep)
    if i % 20 == 0:
        print(f"  chunk {i:>4} | kept rows so far: {sum(len(r) for r in rows):,}")

df = pd.concat(rows, ignore_index=True)
df["date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.date
df = df.dropna(subset=["date", "Article_title", "Stock_symbol"])
df = df[df["Article_title"].str.len() > 5]
df = df.rename(columns={"Article_title": "title", "Stock_symbol": "ticker"})
df = df[["date", "ticker", "title"]]
print(f"Filtered news: {len(df):,} headlines | "
      f"{df['ticker'].nunique()} tickers | "
      f"{df['date'].min()} → {df['date'].max()}")

# ── 4. Score with FinBERT (batched) ─────────────────────────────────────────
finbert = load_finbert()
titles = df["title"].tolist()
BATCH, signed = 64, []
t0 = time.time()
for i in range(0, len(titles), BATCH):
    for r in finbert(titles[i:i+BATCH]):
        label, score = r["label"].lower(), r["score"]
        signed.append(score if label == "positive"
                      else (-score if label == "negative" else 0.0))
    if i % (BATCH * 100) == 0:
        print(f"  scored {i + BATCH:,}/{len(titles):,} | {time.time()-t0:.0f}s elapsed")
df["signed"] = signed

# ── 5. Aggregate to (date × ticker) panel ───────────────────────────────────
daily = df.groupby(["date", "ticker"])["signed"].mean().reset_index()
signal_wide = daily.pivot(index="date", columns="ticker", values="signed")
signal_wide.index = pd.to_datetime(signal_wide.index)
print(f"Sentiment panel shape: {signal_wide.shape}  (dates × tickers)")

# ── 6. Z-score + write to DB ────────────────────────────────────────────────
z_wide = cross_sectional_zscore(signal_wide)
rank_wide = z_wide.rank(axis=1, pct=True)
write_signals_to_db(signal_wide, z_wide, rank_wide, "finbert_sentiment", con)
con.close()
print("Done. FinBERT now has a real historical time series.")