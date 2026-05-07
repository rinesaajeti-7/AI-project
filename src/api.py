import json
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.special import softmax
from fastapi.staticfiles import StaticFiles

# Import paths from config (make sure config.py exists and defines these)
try:
    from .config import BEST_MODEL_PATH, VECTORIZER_PATH, LABELS_PATH
except ImportError:
    # Fallback for when running as a script (not as package)
    BEST_MODEL_PATH = "artifacts/best_model.joblib"
    VECTORIZER_PATH = "artifacts/vectorizer.joblib"
    LABELS_PATH = "artifacts/labels.json"

# Pragu minimal i besimit për të pranuar një parashikim
CONF_THRESHOLD = 0.3

# Global variables for loaded artifacts (will be set in lifespan)
_model = None
_vectorizer = None
_labels = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, vectorizer, and labels on startup, gracefully handling missing files."""
    global _model, _vectorizer, _labels

    # Try to load the model
    if os.path.exists(BEST_MODEL_PATH):
        try:
            _model = joblib.load(BEST_MODEL_PATH)
            print(f"✅ Model loaded from {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Model file not found at {BEST_MODEL_PATH}. Prediction endpoint will return 503.")

    # Try to load the vectorizer
    if os.path.exists(VECTORIZER_PATH):
        try:
            _vectorizer = joblib.load(VECTORIZER_PATH)
            print(f"✅ Vectorizer loaded from {VECTORIZER_PATH}")
        except Exception as e:
            print(f"❌ Error loading vectorizer: {e}")
    else:
        print(f"⚠️ Vectorizer file not found at {VECTORIZER_PATH}.")

    # Try to load labels
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                _labels = json.load(f).get("labels", [])
            print(f"✅ Loaded {len(_labels)} labels from {LABELS_PATH}")
        except Exception as e:
            print(f"❌ Error loading labels: {e}")
    else:
        print(f"⚠️ Labels file not found at {LABELS_PATH}.")

    yield
    # Cleanup if needed (nothing here for now)


# Create FastAPI app with lifespan
app = FastAPI(
    title="Text Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files (frontend) with error handling ---
static_dir = "static"
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    print(f"✅ Serving static files from '{static_dir}'")
else:
    print(f"⚠️ Warning: Static directory '{static_dir}' not found. Frontend not served.")

# Pydantic models
class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    predicted_label: str
    predicted_index: int
    top3: list | None = None


# Health check endpoint (simple)
@app.get("/health")
def health():
    """Basic health endpoint (always returns ok even if model is missing)."""
    return {"status": "ok"}


# More detailed health endpoint for monitoring
@app.get("/healthz")
def healthz():
    """Detailed health check – returns model loaded status."""
    model_ok = _model is not None
    vectorizer_ok = _vectorizer is not None
    labels_ok = _labels is not None
    ready = model_ok and vectorizer_ok and labels_ok
    return {
        "status": "ready" if ready else "degraded",
        "model_loaded": model_ok,
        "vectorizer_loaded": vectorizer_ok,
        "labels_loaded": labels_ok,
    }


# Main prediction endpoint
@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    """Classify text. Returns 503 if model artifacts are missing."""
    # Check that all required artifacts are loaded
    if _model is None or _vectorizer is None or _labels is None:
        missing = []
        if _model is None:
            missing.append("model")
        if _vectorizer is None:
            missing.append("vectorizer")
        if _labels is None:
            missing.append("labels")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: missing {', '.join(missing)}. Please ensure model files are present."
        )

    # Transform text
    X = _vectorizer.transform([payload.text])

    out = {"predicted_label": "", "predicted_index": -1, "top3": None}

    # For models with predict_proba
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X)[0]
        top_k = probs.argsort()[-3:][::-1]
        out["top3"] = [{"label": _labels[i], "prob": float(probs[i])} for i in top_k]

        if out["top3"][0]["prob"] >= CONF_THRESHOLD:
            out["predicted_label"] = out["top3"][0]["label"]
            out["predicted_index"] = top_k[0]
        else:
            out["predicted_label"] = "UNKNOWN"

    # For models with decision_function (e.g., LinearSVC)
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