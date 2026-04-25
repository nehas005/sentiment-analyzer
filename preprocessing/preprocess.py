"""
preprocess.py — Clean and preprocess raw review text for sentiment analysis.

Usage:
    python preprocess.py \
        --input  ../data/raw_reviews.csv \
        --output ../data/cleaned_reviews.csv
"""

import argparse
import os
import re
import string
import warnings

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ── Download NLTK assets on first run ─────────────────────────────────────────
for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_reviews.csv")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_reviews.csv")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Extended emoji / unicode pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


# ──────────────────────────────────────────────────────────────────────────────
# Individual cleaning steps
# ──────────────────────────────────────────────────────────────────────────────

def remove_html(text: str) -> str:
    return HTML_TAG_PATTERN.sub(" ", text)


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub(" ", text)


def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t not in STOP_WORDS)


def lemmatize(text: str) -> str:
    tokens = text.split()
    return " ".join(LEMMATIZER.lemmatize(t) for t in tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE = [
    remove_html,
    remove_urls,
    remove_emojis,
    lowercase,
    remove_punctuation,
    remove_numbers,
    remove_stopwords,
    lemmatize,
    remove_extra_whitespace,
]


def clean_text(text: str) -> str:
    """Apply all cleaning steps in order."""
    if not isinstance(text, str):
        return ""
    for step in PIPELINE:
        text = step(text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment label from numeric rating
# ──────────────────────────────────────────────────────────────────────────────

def rating_to_sentiment(rating: float) -> str:
    """Convert a 1-5 star rating into a sentiment class."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "neutral"

    if r >= 4.0:
        return "positive"
    elif r <= 2.0:
        return "negative"
    else:
        return "neutral"


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame-level processing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: review, rating  (plus optional title, date, reviewer, source).
    Returns enriched DataFrame with cleaned text and sentiment labels.
    """
    required = {"review", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)

    # Combine title + review for richer text if title column exists
    if "title" in df.columns:
        df["full_text"] = (df["title"].fillna("") + " " + df["review"]).str.strip()
    else:
        df["full_text"] = df["review"]

    print("Cleaning text …")
    df["cleaned_text"] = df["full_text"].apply(clean_text)

    print("Assigning sentiment labels from ratings …")
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    # Map to numeric labels for ML
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)

    # Drop rows with empty cleaned text
    before = len(df)
    df = df[df["cleaned_text"].str.len() > 3].reset_index(drop=True)
    print(f"Dropped {before - len(df)} empty rows. Final: {len(df)} rows.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess review CSV")
    parser.add_argument("--input", default=RAW_PATH)
    parser.add_argument("--output", default=CLEAN_PATH)
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows.")

    df_clean = preprocess_dataframe(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    print(f"Saved cleaned data → {args.output}")
    print(df_clean["sentiment"].value_counts())


if __name__ == "__main__":
    main()
