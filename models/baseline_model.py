"""
baseline_model.py — Train and evaluate baseline ML sentiment classifiers.

Models: Logistic Regression, Naive Bayes, Linear SVM
Feature extraction: TF-IDF

Usage:
    python baseline_model.py --data ../data/cleaned_reviews.csv
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_reviews.csv")
MODEL_DIR = os.path.dirname(__file__)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline builders
# ──────────────────────────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced", random_state=42)),
    ])


def build_nb_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.5)),
    ])


def build_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)),
    ])


MODELS = {
    "logistic_regression": build_lr_pipeline,
    "naive_bayes": build_nb_pipeline,
    "svm": build_svm_pipeline,
}


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, model_name: str = "logistic_regression") -> dict:
    X = df["cleaned_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = MODELS[model_name]()
    print(f"\nTraining {model_name} …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "model": pipeline,
        "model_name": model_name,
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_all(df: pd.DataFrame) -> dict:
    results = {}
    for name in MODELS:
        results[name] = train_and_evaluate(df, name)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved → {path}")


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(text: str, pipeline: Pipeline) -> dict:
    """Predict sentiment for a single review text."""
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label = int(pipeline.predict([text])[0])
    proba = None

    # LinearSVC doesn't natively support predict_proba
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba_arr = pipeline.predict_proba([text])[0]
        proba = {label_map[i]: float(p) for i, p in enumerate(proba_arr)}

    return {"sentiment": label_map[label], "label": label, "probabilities": proba}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train baseline ML models")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--model", choices=list(MODELS.keys()), default="logistic_regression")
    parser.add_argument("--all", action="store_true", help="Train all models and pick best")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows | Sentiment distribution:\n{df['sentiment'].value_counts()}")

    if args.all:
        results = train_all(df)
        best_name = max(results, key=lambda k: results[k]["cv_mean"])
        best_pipeline = results[best_name]["model"]
        print(f"\nBest model: {best_name} (CV={results[best_name]['cv_mean']:.4f})")
    else:
        result = train_and_evaluate(df, args.model)
        best_pipeline = result["model"]
        best_name = args.model

    save_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    save_model(best_pipeline, save_path)


if __name__ == "__main__":
    main()
