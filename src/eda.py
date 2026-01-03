import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import CATEGORIES, ARTIFACT_DIR, NGRAM_RANGE, MIN_DF, MAX_DF, SUBLINEAR_TF
from .data import load_dataset
from .preprocess import basic_clean

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df, target_names = load_dataset(CATEGORIES)
    print(f"Total samples: {len(df)}")
    counts = df['label_idx'].value_counts().sort_index()
    for i, c in enumerate(target_names):
        print(f"[{i}] {c:<25} -> {counts.get(i, 0)} samples")

    # Bar chart of samples per category
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(target_names)), counts.values)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=30, ha='right')
    ax.set_ylabel('Samples')
    ax.set_title('Samples per Category')
    fig.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, "samples_per_category.png")
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")

    # Top terms per class by average TF-IDF weight
    vec = TfidfVectorizer(
        preprocessor=basic_clean,
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
    )
    X = vec.fit_transform(df['text'].values)
    vocab = np.array(vec.get_feature_names_out())
    for idx, name in enumerate(target_names):
        mask = (df['label_idx'].values == idx)
        if mask.sum() == 0:
            continue
        class_avg = X[mask].mean(axis=0)  # 1 x V sparse matrix
        class_avg = np.asarray(class_avg).ravel()
        top_idx = np.argsort(class_avg)[-25:][::-1]
        top_terms = [(vocab[i], float(class_avg[i])) for i in top_idx]
        txt_path = os.path.join(ARTIFACT_DIR, f"top_terms_{name.replace(' ', '_')}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for term, weight in top_terms:
                f.write(f"{term}\t{weight:.6f}\n")
        print(f"Saved top terms for '{name}': {txt_path}")

if __name__ == "__main__":
    main()
