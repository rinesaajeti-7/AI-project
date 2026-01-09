import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import (
    RANDOM_SEED, TEST_SIZE, CATEGORIES, ARTIFACT_DIR,
    NGRAM_RANGE, MIN_DF, MAX_DF, SUBLINEAR_TF,
    VECTORIZER_PATH, BEST_MODEL_PATH, LABELS_PATH,
    METRICS_PATH, REPORT_PATH, CONFUSION_MATRIX_PATH
)
from .data import load_dataset
from .preprocess import basic_clean

def plot_confusion(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Load data
    df, target_names = load_dataset(CATEGORIES)
    print("=== Dataset Preview ===")
    print(df.head(20))   # tregon 20 rreshta të parë
    print(df['label_idx'].map(lambda x: target_names[x]).value_counts())
     # tregon sa shembuj ka për secilën kategori
    print("======================")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values,
        df['label_idx'].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['label_idx'].values
    )

    # Vectorizer
    vectorizer = TfidfVectorizer(
        preprocessor=basic_clean,
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    # Models to compare
    models = {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
        "LinearSVC": LinearSVC(random_state=RANDOM_SEED)
    }

    results = {}
    best_name, best_model, best_acc = None, None, -1.0

    for name, clf in models.items():
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        results[name] = {"accuracy": acc, "precision_w": prec, "recall_w": rec, "f1_w": f1}
        print(f"{name}: acc={acc:.4f} | prec_w={prec:.4f} | rec_w={rec:.4f} | f1_w={f1:.4f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, clf, acc

    # Final evaluation with best model
    best_preds = best_model.predict(X_te)
    cm = confusion_matrix(y_test, best_preds)
    report = classification_report(y_test, best_preds, target_names=target_names, digits=4)

    # Persist artifacts
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, BEST_MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({"labels": target_names}, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_scores": results,
            "best_model": best_name,
            "test_accuracy": float(accuracy_score(y_test, best_preds))
        }, f, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(report)

    plot_confusion(cm, target_names, CONFUSION_MATRIX_PATH)

    print("""
Saved artifacts:
- Vectorizer: {VECTORIZER_PATH}
- Best model ({best_name}): {BEST_MODEL_PATH}
- Labels map: {LABELS_PATH}
- Metrics JSON: {METRICS_PATH}
- Classification report: {REPORT_PATH}
- Confusion matrix image: {CONFUSION_MATRIX_PATH}
""")





if __name__ == "__main__":
    main()
