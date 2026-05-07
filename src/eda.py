# ANALIZA EKSPLORATIVE E TË DHËNAVE (EDA)
# Ky skedër kryen analizë eksplorative të dataset-it para trajnimit
# Përfshin: statistikat e distribuimit, vizualizime, dhe analizën e fjalëve karakteristike

import os  # Për operacione me sistemin e skedarëve
import json  # Për manipulim të skedarëve JSON
import numpy as np  # Për llogaritje numerike me vargje
import pandas as pd  # Për manipulim strukturor të të dhënave
import matplotlib.pyplot as plt  # Për krijimin e vizualizimeve grafike

# Importimi i mjeteve për përpunim të tekstit
from sklearn.feature_extraction.text import TfidfVectorizer  # Për kthimin e tekstit në vektorë TF-IDF

# Importimi i konfigurimeve nga skedari i projektit
from .config import CATEGORIES, ARTIFACT_DIR, NGRAM_RANGE, MIN_DF, MAX_DF, SUBLINEAR_TF

# Importimi i moduleve lokalë
from .data import load_dataset  # Funksioni për ngarkimin e dataset-it
from .preprocess import basic_clean  # Funksioni për pastrimin bazë të tekstit

def main(): 
    # 1. KRIJO DIRECTORIUM-IN PËR ARTEFAKTE NËSE NUK EKZISTON
    # 'exist_ok=True' parandalon gabimin nëse directory ekziston tashmë
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 2. NGARKO DATASET-IN
    # Kthen një DataFrame me tekstet dhe indekset e etiketave, plus listën e emrave të etiketave
    df, target_names = load_dataset(CATEGORIES)
    
    # 3. PRINT STATISTIKAT THEMELORE TË DATASET-IT
    print(f"Total samples: {len(df)}")
    
    # Numëro sa shembuj ka për çdo kategori dhe renditi sipas indeksit
    counts = df['label_idx'].value_counts().sort_index()
    
    # Printo një rresht për çdo kategori me emrin dhe numrin e shembujve
    for i, c in enumerate(target_names):
        # :<25 formaton tekstin në 25 karaktere me rreshtim majtas
        print(f"[{i}] {c:<25} -> {counts.get(i, 0)} samples")
    
    # 4. VIZUALIZIM: GRAFIK BAR PËR SHPRËNDARJEN E SHEMBUJVE
    # Krijo një figurë dhe aks me dimensione specifike (8 inç gjerësi, 4 inç lartësi)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Krijo grafikun bar: pozicionet (0,1,2,3) dhe vlerat (numri i shembujve)
    ax.bar(range(len(target_names)), counts.values)
    
    # Vendos shenjat (ticks) në boshtin x në pozicionet 0,1,2,3
    ax.set_xticks(range(len(target_names)))
    
    # Vendos emrat e kategorive si etiketa në boshtin x me rotacion 30 gradë
    ax.set_xticklabels(target_names, rotation=30, ha='right')
    
    # Vendos emër për boshtin y
    ax.set_ylabel('Samples')
    
    # Vendos titull për grafikun
    ax.set_title('Samples per Category')
    
    # Rregullo hapësirat në figurë për ta bërë më të lexueshme
    fig.tight_layout()
    
    # Krijo rrugën e plotë për skedarin e imazhit
    out_path = os.path.join(ARTIFACT_DIR, "samples_per_category.png")
    
    # Ruaj figurën si imazh PNG me rezolucion 160 DPI (dots per inch)
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")
    
    # 5. ANALIZË E FJALËVE/TERMAVE MË KARAKTERISTIKE PËR ÇDO KATEGORI
    # Inicializo vektorizuesin TF-IDF me cilësimet nga konfigurimi
    vec = TfidfVectorizer(
        preprocessor=basic_clean,  # Funksioni për pastrimin e tekstit
        stop_words="english",       # Heq fjalët e zakonshme në anglisht (the, is, and, etj.)
        ngram_range=NGRAM_RANGE,   # Gjatesia e sekuencave të fjalëve (p.sh. (1,2) = fjalë të vetme dhe çifte)
        min_df=MIN_DF,             # Fshin termat që shfaqen në më pak se MIN_DF dokumente
        max_df=MAX_DF,             # Fshin termat që shfaqen në më shumë se MAX_DF dokumente
        sublinear_tf=SUBLINEAR_TF, # Përdor log(1 + tf) në vend të tf për të zvogëluar ndikimin e fjalëve shumë të shpeshta
    )
    
    # Transformo të gjitha tekstet në matricë TF-IDF
    # fit_transform() mëson fjalorin dhe transformon të dhënat
    X = vec.fit_transform(df['text'].values)
    
    # Merr të gjitha termat (fjalët/ngramat) nga vektorizuesi si varg numpy
    vocab = np.array(vec.get_feature_names_out())
    
    # Për çdo kategori në dataset
    for idx, name in enumerate(target_names):
        # Krijo një maskë boolean që tregon se cilat rreshta i përkasin kësaj kategorie
        mask = (df['label_idx'].values == idx)
        
        # Kontrollo nëse ka shembuj në këtë kategori
        if mask.sum() == 0:
            continue  # Kaloj në kategorinë tjetër nëse kjo është bosh
        
        # Njehso mesataren e peshave TF-IDF për të gjithë termat në këtë kategori
        # X[mask] zgjedh vetëm rreshtat që i përkasin kësaj kategorie
        # mean(axis=0) njehson mesataren për secilën kolonë (term)
        class_avg = X[mask].mean(axis=0)  # Kthen një matricë të rrallë 1 x V (V = madhësia e fjalorit)
        
        # Konverto matricën e rrallë në një varg 1D të dendur numpy
        class_avg = np.asarray(class_avg).ravel()
        
        # Gjej indekset e 25 termave me vlera mesatare më të larta TF-IDF
        # np.argsort() kthen indekset e renditura në rend rritës
        # [-25:] merr 25 indekset e fundit (më të lartat)
        # [::-1] i kthen në rend zbritës (më i larti i pari)
        top_idx = np.argsort(class_avg)[-25:][::-1]
        
        # Krijo listën e çifteve (term, peshë) për 25 termat më të lartë
        top_terms = [(vocab[i], float(class_avg[i])) for i in top_idx]
        
        # Krijo emrin e skedarit, duke zëvendësuar hapësirat me nënvizime
        txt_path = os.path.join(ARTIFACT_DIR, f"top_terms_{name.replace(' ', '_')}.txt")
        
        # Shkruaj termat dhe peshat e tyre në skedar
        with open(txt_path, "w", encoding="utf-8") as f:
            for term, weight in top_terms:
                # \t krijon një tab ndërmjet termit dhe peshës
                # :.6f formaton numrin në 6 shifra pas presjes dhjetore
                f.write(f"{term}\t{weight:.6f}\n")
        
        print(f"Saved top terms for '{name}': {txt_path}")

# Ekzekuto funksionin kryesor nëse ky skedër ekzekutohet direkt (jo si modul)
if __name__ == "__main__":
    main()