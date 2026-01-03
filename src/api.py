import json
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .config import BEST_MODEL_PATH, VECTORIZER_PATH, LABELS_PATH

app = FastAPI(title="Text Classification API", version="1.0.0")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    predicted_label: str
    predicted_index: int
    top3: list | None = None


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
    pred_idx = int(_model.predict(X)[0])
    out = {
        "predicted_label": _labels[pred_idx],
        "predicted_index": pred_idx,
        "top3": None
    }

    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X)[0]
        top_k = probs.argsort()[-3:][::-1]
        out["top3"] = [{"label": _labels[i], "prob": float(probs[i])} for i in top_k]

    return out
