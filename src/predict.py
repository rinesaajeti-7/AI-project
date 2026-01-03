import argparse
import json
import joblib
import numpy as np

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["labels"]

def main():
    parser = argparse.ArgumentParser(description="Predict text category")
    parser.add_argument("text", type=str, help="Input text to classify (wrap in quotes)")
    parser.add_argument("--model", default="artifacts/best_model.joblib", help="Path to trained model")
    parser.add_argument("--vectorizer", default="artifacts/tfidf_vectorizer.joblib", help="Path to TF-IDF vectorizer")
    parser.add_argument("--labels", default="artifacts/labels.json", help="Path to label map JSON")
    args = parser.parse_args()

    model = joblib.load(args.model)
    vectorizer = joblib.load(args.vectorizer)
    labels = load_labels(args.labels)

    X = vectorizer.transform([args.text])
    pred = model.predict(X)[0]
    out = {"predicted_index": int(pred), "predicted_label": labels[pred]}

    # If the model supports probabilities, show them
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        top_k = np.argsort(probs)[-3:][::-1]
        out["top3"] = [{"label": labels[i], "prob": float(probs[i])} for i in top_k]

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
