import json
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .config import BEST_MODEL_PATH, VECTORIZER_PATH, LABELS_PATH
from scipy.special import softmax

# Minimal confidence threshold
CONF_THRESHOLD = 0.3  # 30%

app = FastAPI(title="Text Classification API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    predicted_label: str
    predicted_index: int
    top3: list | None = None

# Load artifacts
_model = joblib.load(BEST_MODEL_PATH)
_vectorizer = joblib.load(VECTORIZER_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    _labels = json.load(f)["labels"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    X = _vectorizer.transform([payload.text])
    out = {"predicted_label": "", "predicted_index": -1, "top3": None}

    # MultinomialNB or LogisticRegression (has predict_proba)
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X)[0]
        top_k = probs.argsort()[-3:][::-1]
        out["top3"] = [{"label": _labels[i], "prob": float(probs[i])} for i in top_k]
        if out["top3"][0]["prob"] >= CONF_THRESHOLD:
            out["predicted_label"] = out["top3"][0]["label"]
            out["predicted_index"] = top_k[0]
        else:
            out["predicted_label"] = "UNKNOWN"

    # LinearSVC or other models (use decision_function + softmax)
    elif hasattr(_model, "decision_function"):
        scores = _model.decision_function(X)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probs = softmax(scores[0])
        top_k = probs.argsort()[-3:][::-1]
        out["top3"] = [{"label": _labels[i], "prob": float(probs[i])} for i in top_k]
        if probs[top_k[0]] >= CONF_THRESHOLD:
            out["predicted_label"] = _labels[top_k[0]]
            out["predicted_index"] = int(top_k[0])
        else:
            out["predicted_label"] = "UNKNOWN"

    return out
