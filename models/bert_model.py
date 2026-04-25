"""
bert_model.py — Fine-tune distilBERT for 3-class sentiment classification.

Usage:
    # Fine-tune:
    python bert_model.py --train --data ../data/cleaned_reviews.csv

    # Predict:
    python bert_model.py --predict "This product is amazing!"
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "distilbert-base-uncased"
BERT_DIR = os.path.join(os.path.dirname(__file__), "bert_model")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_reviews.csv")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
NUM_LABELS = 3


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class BERTSentimentTrainer:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        max_len: int = 256,
        batch_size: int = 16,
        epochs: int = 3,
        lr: float = 2e-5,
        save_dir: str = BERT_DIR,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

    def _build_loader(self, texts, labels, shuffle: bool = True) -> DataLoader:
        ds = ReviewDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, df: pd.DataFrame):
        X = df["cleaned_text"].tolist()
        y = df["label"].tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        train_loader = self._build_loader(X_train, y_train)
        val_loader = self._build_loader(X_val, y_val, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            val_acc = self._evaluate(val_loader)
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        self.save()

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].numpy()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()
                preds.extend(np.argmax(logits, axis=1))
                true_labels.extend(labels)

        print(classification_report(true_labels, preds, target_names=list(LABEL_MAP.values())))
        return accuracy_score(true_labels, preds)

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        print(f"BERT model saved → {self.save_dir}")

    def predict(self, text: str) -> dict:
        self.model.eval()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits.cpu().numpy()[0]
        probs = _softmax(logits)
        label = int(np.argmax(probs))

        return {
            "sentiment": LABEL_MAP[label],
            "label": label,
            "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Inference loader (for use in dashboard without full trainer)
# ──────────────────────────────────────────────────────────────────────────────

class BERTSentimentPredictor:
    """Lightweight inference-only wrapper. Loads a saved model."""

    def __init__(self, model_dir: str = BERT_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        probs = _softmax(logits.cpu().numpy()[0])
        label = int(np.argmax(probs))
        return {
            "sentiment": LABEL_MAP[label],
            "label": label,
            "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str, default="")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if args.train:
        df = pd.read_csv(args.data)
        trainer = BERTSentimentTrainer(epochs=args.epochs, batch_size=args.batch)
        trainer.train(df)

    if args.predict:
        predictor = BERTSentimentPredictor()
        result = predictor.predict(args.predict)
        print(result)


if __name__ == "__main__":
    main()
