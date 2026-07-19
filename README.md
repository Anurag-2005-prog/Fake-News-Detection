# The Verification Desk — Fake News Detector

An end-to-end Machine Learning project that classifies news text as **REAL** or
**FAKE**, built with a scikit-learn ML pipeline, a Flask REST API backend, and
a custom-designed HTML/CSS/JS frontend ("newsroom verification desk" theme).

This project is designed to be presented as an **internship / academic project**:
it includes a full ML pipeline (data → features → model → evaluation), a working
API, an explainability layer (shows *why* the model made its decision), and a
polished UI — everything needed for a demo, a report, and a viva.

---

## 1. What it does

You paste a headline or article into the console. The backend:

1. Cleans the text
2. Converts it to a **TF-IDF** vector (word unigrams + bigrams)
3. Computes **hand-engineered stylistic features**: exclamation marks, question
   marks, ALL-CAPS word ratio, sensational-phrase-lexicon hits, average word
   length, and text length
4. Feeds both feature sets into a trained **Logistic Regression** classifier
5. Returns a verdict (REAL/FAKE), a confidence score, the top words that swayed
   the decision, and a breakdown of the stylistic signals detected

The frontend displays this as a rubber-stamp verdict, an animated confidence
bar, an "evidence" panel, and a running history of everything checked in the
session — plus a live data plate showing the model's accuracy/precision/recall/F1.

---

## 2. Project structure

```
fake-news-detector/
├── backend/
│   ├── app.py                # Flask API + serves the frontend
│   ├── train_model.py        # Trains & evaluates the ML model
│   ├── generate_dataset.py   # Builds the labeled training dataset
│   └── requirements.txt
├── data/
│   └── news_dataset.csv      # Generated REAL/FAKE labeled dataset
├── model/
│   ├── model.pkl             # Trained Logistic Regression classifier
│   ├── vectorizer.pkl        # Fitted TF-IDF vectorizer
│   ├── scaler.pkl            # Fitted StandardScaler for engineered features
│   └── metrics.pkl           # Accuracy/precision/recall/F1 from training
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

---

## 3. How to run it

### Requirements
- Python 3.9+
- pip

### Setup

```bash
cd fake-news-detector/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### (Already done for you, but to regenerate from scratch)

```bash
python3 generate_dataset.py     # builds data/news_dataset.csv
python3 train_model.py          # trains the model, prints metrics, saves to model/
```

### Run the app

```bash
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser. That's it — one server
serves both the API and the frontend, so there's no separate frontend build
step or CORS configuration needed.

---

## 4. API reference

| Method | Endpoint             | Description                                   |
|--------|-----------------------|------------------------------------------------|
| GET    | `/api/health`         | Health check                                   |
| GET    | `/api/stats`          | Model metrics (accuracy, precision, recall, F1)|
| POST   | `/api/predict`        | Body: `{"text": "..."}` → verdict + explanation|
| GET    | `/api/history`        | Last 25 predictions made this session          |
| POST   | `/api/history/clear`  | Clears session history                         |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SHOCKING: insiders reveal a secret cure banned by doctors!!!"}'
```

---

## 5. The dataset — important note for your report

Because this environment doesn't have internet access to download a public
corpus, `generate_dataset.py` **programmatically builds** a 1,200-row labeled
dataset using template + entity substitution across 10 topics (politics,
health, tech, science, business, sports, etc). It deliberately encodes the
same stylistic signals researchers use in real fake-news detection work:

- **REAL style**: attributed ("according to", "officials said"), specific
  (numbers, dates, named organizations), neutral tone
- **FAKE style**: sensational (ALL CAPS, "!!!"), unattributed ("anonymous
  insiders"), clickbait phrasing ("you won't believe", "SHOCKING")

This keeps the whole pipeline runnable offline and produces a model that
already generalizes to new, hand-written sentences (verified in testing —
see below). **For your final submission, it's strongly recommended you swap
in a real public dataset** for stronger credibility — the code needs almost
no changes:

- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) (William Yang Wang, 2017)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

To use a real dataset: replace `data/news_dataset.csv` with one that has
`text` and `label` columns (label = 1 for REAL, 0 for FAKE), then re-run
`python3 train_model.py`.

### Model performance
On the generated dataset (80/20 train/test split), the model reaches ~100%
test accuracy — expected, since the synthetic data has a clean stylistic
separation. When you swap in a real-world dataset, expect a more realistic
90–96% accuracy range depending on preprocessing choices, which is still a
strong, presentable result and more representative of real deployment
conditions. The pipeline, API, and frontend all work unchanged either way.

---

## 6. Why these ML choices (for your viva / report)

- **TF-IDF over word n-grams**: captures *what* is being said — vocabulary
  and phrasing patterns common to fake vs. real news.
- **Hand-engineered stylistic features**: captures *how* it's being said —
  independent of topic, so the model doesn't just memorize subject matter.
  This is a common technique in real fake-news-detection research, since
  topic-only models overfit to whatever subjects appear in training data.
- **Logistic Regression**: chosen over more complex models (SVM, random
  forest, neural nets) because it (a) trains fast on a laptop with no GPU,
  (b) gives well-calibrated probabilities — useful for the confidence score
  in the UI, and (c) is directly interpretable via its coefficients, which
  powers the "words that swayed the ruling" explainability feature.
- **StandardScaler on engineered features**: TF-IDF values and raw counts
  (like text length) live on very different scales; scaling prevents the
  larger-magnitude raw features from dominating the model.

### Possible extensions (good for bonus marks / future work section)
- Swap Logistic Regression for a fine-tuned transformer (e.g. DistilBERT) for
  higher accuracy on real-world data at the cost of speed/interpretability.
- Add a source-credibility check (cross-reference claims against a
  fact-checking API).
- Persist history to a real database (SQLite/PostgreSQL) instead of in-memory.
- Add user accounts and a "flagged articles" dashboard.
- Deploy with Gunicorn + Nginx (the built-in Flask server is dev-only).

---

## 7. Tech stack summary

| Layer      | Technology                                   |
|------------|-----------------------------------------------|
| ML         | scikit-learn (TF-IDF, LogisticRegression, StandardScaler) |
| Backend    | Flask (Python)                                |
| Frontend   | HTML5, CSS3, vanilla JavaScript (no build step)|
| Data       | pandas/csv, numpy, scipy sparse matrices      |

---

## 8. Disclaimer

This is an educational/demo project. Model verdicts are statistical
predictions based on writing style and should not be treated as fact-checking.
Always verify important claims against primary sources.
