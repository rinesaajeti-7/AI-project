import json
import sys
import joblib
from pathlib import Path
from .config import BEST_MODEL_PATH, VECTORIZER_PATH, LABELS_PATH

def main():
    # Get text input
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Shkruaj tekstin për klasifikim: ").strip()

    if not text:
        print(json.dumps({
            "error": "Nuk u dha tekst për klasifikim",
            "message": "Përdor: python3 -m src.predict \"Teksti yt këtu\""
        }, indent=2, ensure_ascii=False))
        return

    # Load artifacts
    model = joblib.load(BEST_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    # Transform input text
    X = vectorizer.transform([text])
    pred_idx = int(model.predict(X)[0])
    predicted_label = labels[pred_idx]

    # Try to get decision function scores for top 3
    top3 = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X).ravel()
        if scores.ndim == 0:  # binary classification case
            scores = [scores, -scores]
        top_idx = scores.argsort()[-3:][::-1]
        top3 = [{"label": labels[i], "score": float(scores[i])} for i in top_idx]

    output = {
        "text": text,
        "predicted_label": predicted_label,
        "predicted_index": pred_idx,
        "top3": top3,
        "success": True
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
