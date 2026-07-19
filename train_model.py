"""
train_model.py
---------------
Trains the Fake News Detector ML model.

Pipeline:
 1. Load data/news_dataset.csv
 2. Clean text (lowercase, strip punctuation noise, remove extra whitespace)
 3. Feature extraction:
       a) TF-IDF over word unigrams + bigrams  (captures vocabulary/content)
       b) Hand-engineered stylistic features    (captures WRITING STYLE):
            - exclamation_count
            - question_count
            - caps_word_ratio        (ALL-CAPS words / total words)
            - sensational_word_count (clickbait lexicon hits)
            - avg_word_length
            - text_length
 4. Combine TF-IDF + engineered features via scipy sparse hstack
 5. Train a Logistic Regression classifier (fast, well-calibrated
    probabilities -> good for a confidence score in the UI)
 6. Evaluate with accuracy / precision / recall / F1 / confusion matrix
 7. Persist model, vectorizer, and feature scaler with pickle to model/

Run:  python train_model.py
"""

import os
import re
import csv
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "news_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

SENSATIONAL_WORDS = [
    "shocking", "secret", "banned", "won't believe", "wont believe", "hidden",
    "truth", "conspiracy", "insider", "leaked", "urgent", "breaking",
    "miracle", "exposed", "cover up", "coverup", "anonymous", "hate this",
    "you'll never", "youll never", "wake up", "share this", "before it's",
    "before its", "stunned", "terrifying", "evil", "rigged", "scandal",
    "whistleblower", "blackout", "silent", "doesn't want", "doesnt want",
]


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def engineered_features(text: str) -> list:
    words = text.split()
    n_words = max(len(words), 1)
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    lower = text.lower()
    sensational_hits = sum(lower.count(term) for term in SENSATIONAL_WORDS)
    avg_word_len = sum(len(w) for w in words) / n_words

    return [
        text.count("!"),
        text.count("?"),
        caps_words / n_words,
        sensational_hits,
        avg_word_len,
        len(text),
    ]


def load_dataset():
    texts, labels = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(clean_text(row["text"]))
            labels.append(int(row["label"]))
    return texts, labels


def main():
    print("Loading dataset...")
    texts, labels = load_dataset()
    print(f"  {len(texts)} samples loaded")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=6000,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    print("Computing engineered stylistic features...")
    train_eng = np.array([engineered_features(t) for t in X_train_text])
    test_eng = np.array([engineered_features(t) for t in X_test_text])

    scaler = StandardScaler()
    train_eng_scaled = scaler.fit_transform(train_eng)
    test_eng_scaled = scaler.transform(test_eng)

    X_train = hstack([X_train_tfidf, csr_matrix(train_eng_scaled)])
    X_test = hstack([X_test_tfidf, csr_matrix(test_eng_scaled)])

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix (rows=actual, cols=predicted) [FAKE, REAL]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    print("Saving model artifacts...")
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "n_train": len(y_train), "n_test": len(y_test),
    }
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nSaved model, vectorizer, scaler, and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
