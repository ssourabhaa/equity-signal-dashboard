"""Gemini-based sentiment scorer for news headlines.

Batches headlines per Gemini call to maximize free-tier quota usage.
Returns FinBERT-style per-headline labels and signed scores.

Output schema matches FinBERT for direct comparison in the benchmark table:
  - label: "positive" | "negative" | "neutral"
  - score: confidence in [0.0, 1.0]
  - signed_score: signed version in [-1.0, 1.0]
       positive 0.8 -> +0.8
       negative 0.6 -> -0.6
       neutral  *   ->  0.0
"""
import os
import json
import time
import re
import warnings
from bs4 import XMLParsedAsHTMLWarning
from dotenv import load_dotenv
from google import genai

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
load_dotenv()

_api_key = os.getenv("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

_client = genai.Client(api_key=_api_key)
MODEL_NAME = "gemini-flash-lite-latest"

BATCH_SIZE = 75
INTER_CALL_SLEEP_SECONDS = 4


def _build_prompt(headlines: list[str], ticker: str) -> str:
    """Build the structured prompt for one batch."""
    numbered_headlines = "\n".join(
        f"{i+1}. {headline}" for i, headline in enumerate(headlines)
    )
    return f"""You are a financial sentiment classifier. For each news headline below \
about {ticker}, classify whether it expresses positive, negative, or neutral \
sentiment about the company's outlook, business, or stock price.

Guidelines:
- positive = favorable news (beats earnings, new product launch, analyst upgrade, expansion)
- negative = unfavorable news (lawsuits, misses, downgrades, executive departure, supply problems)
- neutral = no clear directional sentiment (general industry coverage, factual announcements)

For each headline, return a confidence score between 0.0 and 1.0 indicating \
how confident you are in the classification.

Output ONLY a JSON array of objects, one per headline, in the SAME ORDER as input:
[
  {{"id": 1, "label": "positive", "score": 0.85}},
  {{"id": 2, "label": "negative", "score": 0.72}},
  ...
]

No markdown, no commentary, no extra text.

Headlines about {ticker}:
{numbered_headlines}
"""


def _parse_response(raw_text: str, n_expected: int) -> list[dict | None]:
    """
    Parse Gemini's JSON response into a list of {label, score, signed_score}.
    Returns None at positions where parsing fails — caller decides what to do.
    """
    raw = raw_text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return [None] * n_expected
    except json.JSONDecodeError:
        return [None] * n_expected

    # Index items by id (1-based per prompt)
    by_id = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        label = item.get("label", "").lower()
        score = item.get("score")

        if not isinstance(item_id, int):
            continue
        if label not in ("positive", "negative", "neutral"):
            continue
        try:
            score = float(score)
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            continue

        # Convert to signed score (FinBERT-style)
        if label == "positive":
            signed = score
        elif label == "negative":
            signed = -score
        else:
            signed = 0.0

        by_id[item_id] = {
            "label": label,
            "score": score,
            "signed_score": signed,
        }

    # Return in original order, None for missing ids
    return [by_id.get(i + 1) for i in range(n_expected)]


def score_headlines_batch(headlines: list[str], ticker: str) -> list[dict | None]:
    """
    Score one batch of headlines for a single ticker.
    Returns list of {label, score, signed_score} dicts (or None on parse failure).
    """
    if not headlines:
        return []

    prompt = _build_prompt(headlines, ticker)
    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"temperature": 0.0},
    )
    return _parse_response(response.text, len(headlines))


def score_all_headlines(headlines: list[str], ticker: str) -> list[dict | None]:
    """
    Score an arbitrary-length list of headlines, batching automatically.
    Inserts a sleep between batches to respect rate limits.
    """
    results: list[dict | None] = []
    for i in range(0, len(headlines), BATCH_SIZE):
        chunk = headlines[i:i + BATCH_SIZE]
        chunk_results = score_headlines_batch(chunk, ticker)
        results.extend(chunk_results)
        # Sleep between batches if there are more to come
        if i + BATCH_SIZE < len(headlines):
            time.sleep(INTER_CALL_SLEEP_SECONDS)
    return results
