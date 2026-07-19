"""
app.py
------
Flask backend for the Fake News Detector.

Endpoints
  GET  /                 -> serves the frontend (index.html)
  GET  /api/health       -> health check
  GET  /api/stats        -> model metrics computed at training time
  POST /api/predict      -> body: {"text": "..."}
                             returns verdict, confidence, and an
                             explainability breakdown (top contributing
                             words + the stylistic signals detected)
  POST /api/history/clear -> clears in-memory history (per server run)
  GET  /api/history       -> last N predictions made this session
"""

import os
import re
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
    metrics = pickle.load(f)

FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())
N_TFIDF_FEATURES = len(FEATURE_NAMES)

ENGINEERED_LABELS = [
    "Exclamation marks",
    "Question marks",
    "ALL-CAPS word ratio",
    "Sensational phrase hits",
    "Average word length",
    "Text length (chars)",
]

SENSATIONAL_WORDS = [
    "shocking", "secret", "banned", "won't believe", "wont believe", "hidden",
    "truth", "conspiracy", "insider", "leaked", "urgent", "breaking",
    "miracle", "exposed", "cover up", "coverup", "anonymous", "hate this",
    "you'll never", "youll never", "wake up", "share this", "before it's",
    "before its", "stunned", "terrifying", "evil", "rigged", "scandal",
    "whistleblower", "blackout", "silent", "doesn't want", "doesnt want",
]

# in-memory history for this server run (not persisted to disk)
HISTORY = []
MAX_HISTORY = 25


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


def top_contributing_terms(text: str, tfidf_vec, k=6):
    """Find which words in the input pushed the decision toward FAKE or REAL,
    using the model's learned coefficients for the TF-IDF portion only."""
    coefs = clf.coef_[0][:N_TFIDF_FEATURES]
    row = tfidf_vec.tocoo()
    contributions = []
    for idx, val in zip(row.col, row.data):
        weight = coefs[idx] * val
        if abs(weight) > 1e-6:
            contributions.append((FEATURE_NAMES[idx], weight))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:k]
    return [
        {"term": term, "direction": "REAL" if w > 0 else "FAKE", "weight": round(float(w), 4)}
        for term, w in top
    ]


def predict_text(raw_text: str):
    text = clean_text(raw_text)
    tfidf_vec = vectorizer.transform([text])
    eng = scaler.transform(np.array([engineered_features(text)]))
    X = hstack([tfidf_vec, csr_matrix(eng)])

    proba = clf.predict_proba(X)[0]  # [P(FAKE), P(REAL)]
    pred = int(np.argmax(proba))
    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(proba[pred])

    raw_eng_values = engineered_features(text)
    engineered_breakdown = [
        {"name": name, "value": round(val, 3) if isinstance(val, float) else val}
        for name, val in zip(ENGINEERED_LABELS, raw_eng_values)
    ]

    result = {
        "label": label,
        "confidence": round(confidence * 100, 1),
        "probabilities": {
            "FAKE": round(float(proba[0]) * 100, 1),
            "REAL": round(float(proba[1]) * 100, 1),
        },
        "top_terms": top_contributing_terms(text, tfidf_vec),
        "stylistic_signals": engineered_breakdown,
        "input_preview": text[:160] + ("..." if len(text) > 160 else ""),
    }
    return result


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def stats():
    return jsonify({
        "accuracy": round(metrics["accuracy"] * 100, 2),
        "precision": round(metrics["precision"] * 100, 2),
        "recall": round(metrics["recall"] * 100, 2),
        "f1": round(metrics["f1"] * 100, 2),
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "vocab_size": N_TFIDF_FEATURES,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "Please provide non-empty article text."}), 400
    if len(text.strip()) < 15:
        return jsonify({"error": "Please provide at least a full sentence (15+ characters) for a reliable result."}), 400

    result = predict_text(text)
    HISTORY.insert(0, {
        "preview": result["input_preview"],
        "label": result["label"],
        "confidence": result["confidence"],
    })
    del HISTORY[MAX_HISTORY:]
    return jsonify(result)


@app.route("/api/history")
def history():
    return jsonify(HISTORY)


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    HISTORY.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Fake News Detector backend running at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
