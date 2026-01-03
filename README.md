# AI Internship Challenge – Text Classification Task

A small end‑to‑end text classification project that loads a public dataset, performs preprocessing and EDA, trains a few simple models, evaluates them, and exposes a prediction script and an optional API.

## ✅ Dataset

**20 Newsgroups (subset of 4 categories):**

- `comp.sys.mac.hardware`
- `rec.sport.baseball`
- `sci.space`
- `talk.politics.mideast`

**Source & docs:**  
- Scikit‑learn dataset docs: https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset  
- Original dataset home: http://qwone.com/~jason/20Newsgroups/

> The code downloads the data programmatically via `sklearn.datasets.fetch_20newsgroups` on first run.

## 📦 Project Structure

```
text-classification-project/
├── artifacts/                         # Saved models, reports, plots
├── src/
│   ├── api.py                         # FastAPI app for /predict
│   ├── config.py                      # Config: categories, paths, vectorizer settings
│   ├── data.py                        # Dataset loader (20 Newsgroups subset)
│   ├── eda.py                         # Simple exploratory analysis + charts
│   ├── preprocess.py                  # Basic text cleaning
│   ├── predict.py                     # CLI prediction script
│   └── train.py                       # Model training + evaluation + saving artifacts
├── requirements.txt
└── README.md
```

## 🔧 Setup & Run


```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m src.train

python -m src.eda

python -m src.predict "The Yankees won their baseball game last night"


uvicorn src.api:app --reload

```

## 📊 What’s Included

- **Data Preparation**: lowercasing, URL/HTML removal, non‑alpha stripping (see `preprocess.py`)
- **Exploratory Analysis**: sample counts per class, top terms per class (TF‑IDF), saved to `artifacts/`
- **Models Compared**: `MultinomialNB`, `LogisticRegression`, `LinearSVC`
- **Features**: TF‑IDF (uni+bi‑grams), English stopwords, sublinear TF
- **Evaluation**: accuracy, weighted precision/recall/F1, confusion matrix plot, classification report
- **Prediction**: CLI tool + optional **FastAPI** endpoint

## 🔍 Notes

- We intentionally re‑split the data (80/20 stratified) to demonstrate the train/test pipeline.
- The best model is chosen by highest test accuracy among the three baselines and persisted as `artifacts/best_model.joblib` along with the TF‑IDF vectorizer.
- All artifacts (reports, metrics, plots) are written into the `artifacts/` folder.

## 🧪 Reproducibility

- Random seeds are controlled in `src/config.py` (`RANDOM_SEED=42`).
- Vectorizer and model settings are defined centrally in `src/config.py`.

## 🐛 Troubleshooting

- If you change Python versions or encounter binary issues, reinstall deps inside a fresh venv.
- If `uvicorn` can’t find the app, ensure you run it from the project root and that training has been run (so the model/vectorizer exist).

