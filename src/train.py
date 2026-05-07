# SKEDARI KRYESOR PËR TRAJNIMIN E MODELIT
# Ky skedër kryen të gjithë procesin e trajnimit të modelit të klasifikimit të tekstit:
# 1. Ngarkimi i të dhënave
# 2. Përpunimi dhe vektorizimi i tekstit
# 3. Testimi i tre algoritmeve të ndryshme ML
# 4. Zgjedhja e modelit më të mirë
# 5. Ruajtja e të gjitha artefakteve të nevojshme

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importimi i mjeteve nga scikit-learn
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

# Importimi i konfigurimeve nga projekti
from .config import (
    RANDOM_SEED, TEST_SIZE, CATEGORIES, ARTIFACT_DIR,
    NGRAM_RANGE, MIN_DF, MAX_DF, SUBLINEAR_TF,
    VECTORIZER_PATH, BEST_MODEL_PATH, LABELS_PATH,
    METRICS_PATH, REPORT_PATH, CONFUSION_MATRIX_PATH
)

# Importimi i moduleve tona lokale
from .data import load_dataset
from .preprocess import basic_clean


# FUNKSIONI PËR VIZUALIZIMIN E MATRICËS SË KONFUSIONIT
def plot_confusion(cm, labels, out_path):
    """
    Krijon dhe ruan një vizualizim të matricës së konfuzionit.
    
    Matrica e konfuzionit tregon se sa saktë modeli parashikon secilën kategori:
    - Diagonalja kryesore (nga lart majtas në poshtë djathtas) tregon parashikimet e sakta
    - Qelizat jashtë diagonales tregojnë gabimet e modelit
    
    Args:
        cm (numpy.ndarray): Matrica e konfuzionit (n_classes x n_classes)
        labels (list): Lista e emrave të kategorive
        out_path (str): Rruga ku do të ruhet fotografia
    """
    # Krijo një figurë me dimensione specifike (6 inç gjerësi, 5 inç lartësi)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Shfaq matricën e konfuzionit si imazh me ngjyra
    im = ax.imshow(cm, interpolation='nearest')
    
    # Shto një shkallë ngjyrash (colorbar) në anë për referencë
    ax.figure.colorbar(im, ax=ax)
    
    # Vendos konfigurimet për akset dhe etiketat
    ax.set(xticks=np.arange(cm.shape[1]),          # Vendos tick-et në boshtin x
           yticks=np.arange(cm.shape[0]),          # Vendos tick-et në boshtin y
           xticklabels=labels,                     # Emrat e kategorive për boshtin x
           yticklabels=labels,                     # Emrat e kategorive për boshtin y
           ylabel='True label',                    # Etiketa për boshtin y (etiketa e vërtetë)
           xlabel='Predicted label',               # Etiketa për boshtin x (etiketa e parashikuar)
           title='Confusion Matrix')               # Titulli i grafikut
    
    # Rrotullo etiketat në boshtin x me 30 gradë për ta bërë më të lexueshme
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Llogarit pragun për vendosjen e ngjyrës së tekstit (bardh ose zi)
    thresh = cm.max() / 2.
    
    # Shto numrat brenda çdo qelize të matricës
    for i in range(cm.shape[0]):           # Për çdo rresht (kategori të vërtetë)
        for j in range(cm.shape[1]):       # Për çdo kolonë (kategori të parashikuar)
            # Vendos tekstin në qendër të qelizës
            ax.text(j, i, format(cm[i, j], 'd'),      # 'd' tregon format të numrit të plotë
                    ha="center", va="center",         # Qendro tekstin horizontalisht dhe vertikalisht
                    # Përdor tekst të bardhë nëse sfondi është i errët, të zi nëse është i çelët
                    color="white" if cm[i, j] > thresh else "black")
    
    # Rregullo hapësirat në figurë për ta bërë më të lexueshme
    fig.tight_layout()
    
    # Ruaj figurën si skedar imazhi me rezolucion 160 DPI
    fig.savefig(out_path, dpi=160)
    
    # Mbyll figurën për të liruar memorjen (e rëndësishme për skripta të gjatë)
    plt.close(fig)

# FUNKSIONI KRYESOR PËR TRAJNIMIN E MODELIT
def main():
    """Funksioni kryesor që ekzekuton të gjithë procesin e trajnimit."""
    
    # 1. KRIJO DIRECTORIUM-IN PËR ARTEFAKTE NËSE NUK EKZISTON
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 2. NGARKO TË DHËNAT
    df, target_names = load_dataset(CATEGORIES)
    
    # Printo një preview të të dhënave për verifikim
    print("=== Dataset Preview ===")
    print(df.head(20))   # Shfaq 20 rreshtat e parë të dataset-it
    # Shfaq numrin e shembujve për secilën kategori
    print(df['label_idx'].map(lambda x: target_names[x]).value_counts())
    print("======================")
    
    # 3. NDAJ TË DHËNAT NË TRAIN DHE TEST
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values,           # Veçoritë (tekstet)
        df['label_idx'].values,      # Etiketat (indekset e kategorive)
        test_size=TEST_SIZE,         # Përqindja e të dhënave për test (20%)
        random_state=RANDOM_SEED,    # Për rezultate të riprodhueshme
        stratify=df['label_idx'].values  # Ruaj përpjesën e kategorive edhe në train edhe në test
    )
    
    # 4. VEKTORIZIMI I TEKSTIT ME TF-IDF
    vectorizer = TfidfVectorizer(
        preprocessor=basic_clean,    # Funksioni për pastrimin e tekstit
        stop_words="english",        # Heq fjalët e zakonshme në anglisht
        ngram_range=NGRAM_RANGE,     # Përdor fjalë të vetme dhe çifte fjalësh
        min_df=MIN_DF,              # Heq termat shumë të rrallë
        max_df=MAX_DF,              # Heq termat shumë të zakonshëm
        sublinear_tf=SUBLINEAR_TF,  # Përdor log(1 + tf) në vend të tf
    )
    
    # Trajno vektorizuesin në të dhënat e trajnimit dhe transformoji ato
    X_tr = vectorizer.fit_transform(X_train)
    
    # Transformo të dhënat e testimit duke përdorur vektorizuesin e trajnuar
    X_te = vectorizer.transform(X_test)
    
    # 5. TESTIMI I ALGORITMEVE TË NDRYSHME ML
    # Krijo një dictionary me tre algoritme të ndryshme për të krahasuar
    models = {
        "MultinomialNB": MultinomialNB(alpha=0.1),           # Naïve Bayes - i thjeshtë dhe i shpejtë
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),  # Regresion Logjistik - i balancuar
        "LinearSVC": LinearSVC(random_state=RANDOM_SEED)     # SVM Linear - shpesh më i saktë por më i ngadaltë
    }
    
    # Dictionary për të ruajtur rezultatet e të gjitha modeleve
    results = {}
    
    # Variabla për të ndjekur modelin më të mirë
    best_name, best_model, best_acc = None, None, -1.0
    
    # Testo çdo model një nga një
    for name, clf in models.items():
        # Trajno modelin në të dhënat e trajnimit
        clf.fit(X_tr, y_train)
        
        # Parashiko në të dhënat e testimit
        preds = clf.predict(X_te)
        
        # Llogarit saktësinë (accuracy)
        acc = accuracy_score(y_test, preds)
        
        # Llogarit metrikat e tjera (precision, recall, f1) me peshë mesatare
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, 
            average="weighted",  # Mesatare e ponderuar (merr parasysh madhësinë e çdo klase)
            zero_division=0      # Si të trajtohen pjesëtimet me zero
        )
        
        # Ruaj rezultatet
        results[name] = {
            "accuracy": acc,
            "precision_w": prec,
            "recall_w": rec,
            "f1_w": f1
        }
        
        # Printo rezultatet në konsol
        print(f"{name}: acc={acc:.4f} | prec_w={prec:.4f} | rec_w={rec:.4f} | f1_w={f1:.4f}")
        
        # Kontrollo nëse ky model është më i mirë se të tjerët
        if acc > best_acc:
            best_name, best_model, best_acc = name, clf, acc
    
    # 6. VLERËSIMI FINAL ME MODELIN MË TË MIRË
    # Parashiko me modelin më të mirë
    best_preds = best_model.predict(X_te)
    
    # Llogarit matricën e konfuzionit për modelin më të mirë
    cm = confusion_matrix(y_test, best_preds)
    
    # Gjenero raportin e klasifikimit të detajuar
    report = classification_report(y_test, best_preds, target_names=target_names, digits=4)
    
    # 7. RUAJTJA E ARTEFAKTEVE TË TRAJNIMIT
    # Ruaj vektorizuesin TF-IDF
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    # Ruaj modelin më të mirë
    joblib.dump(best_model, BEST_MODEL_PATH)
    
    # Ruaj emrat e kategorive në një skedar JSON
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({"labels": target_names}, f, indent=2)
    
    # Ruaj të gjitha metrikat në një skedar JSON
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_scores": results,            # Rezultatet e të gjitha modeleve
            "best_model": best_name,            # Emri i modelit më të mirë
            "test_accuracy": float(accuracy_score(y_test, best_preds))  # Saktësia përfundimtare
        }, f, indent=2)
    
    # Ruaj raportin e klasifikimit në një skedar tekst
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(report)
    
    # Krijo dhe ruaj vizualizimin e matricës së konfuzionit
    plot_confusion(cm, target_names, CONFUSION_MATRIX_PATH)
    
    # 8. PRINT REZULTATET PËRFUNDIMTARE
    print("""
Saved artifacts:
- Vectorizer: {VECTORIZER_PATH}
- Best model ({best_name}): {BEST_MODEL_PATH}
- Labels map: {LABELS_PATH}
- Metrics JSON: {METRICS_PATH}
- Classification report: {REPORT_PATH}
- Confusion matrix image: {CONFUSION_MATRIX_PATH}
""")

# EKZEKUTIMI I SCRIPT-IT
if __name__ == "__main__":
    main()