# SKEDAR PËR PARASHIKIM NGA KOMANDA LINUX
# Ky skedër lejon testimin e modelit direkt nga terminali
# Duke përdorur: python -m src.predict "teksti im"
# Ose: python -m src.predict (dhe shkruan tekstin në prompt)

import json  # Për të formatuar daljen si JSON
import sys  # Për të lexuar argumentet nga komanda
import joblib  # Për të ngarkuar modelin dhe vektorizuesin e ruajtur
from pathlib import Path  # Për të punuar me rrugë të skedarëve (nuk përdoret aktualisht)
from .config import BEST_MODEL_PATH, VECTORIZER_PATH, LABELS_PATH  # Importo rrugët nga konfigurimi

def main():
    # 1. MARRJA E TEKSTIT HYRËS
    # Kontrollo nëse janë dhënë argumente në komandë
    if len(sys.argv) > 1:
        # Bashko të gjitha argumentet në një tekst të vetëm
        # sys.argv[0] është emri i skedarit, kështu që fillojmë nga 1
        text = " ".join(sys.argv[1:])
    else:
        # Nëse nuk ka argumente, kërko tekstin nga përdoruesi
        text = input("Shkruaj tekstin për klasifikim: ").strip()
    
   
    # 2. VALIDIMI I HYRJES
    # Kontrollo nëse teksti është bosh pasi të jetë hequr hapësira e tepërt
    if not text:
        # Nëse teksti është bosh, printo një mesazh gabimi në format JSON
        print(json.dumps({
            "error": "Nuk u dha tekst për klasifikim",  # Përshkrimi i gabimit
            "message": "Përdor: python3 -m src.predict \"Teksti yt këtu\""  # Udhëzime për përdorim
        }, indent=2, ensure_ascii=False))  # indent=2: formatim i bukur, ensure_ascii=False: mbështet karaktere shqip
        return  # Dil nga funksioni pasi të tregohet gabimi
    

    # 3. NGARKIMI I ARTEFAKTEVE TË MODELIT
    # Ngarko modelin e makinës mësimore (LogisticRegression, SVC, ose Naive Bayes)
    model = joblib.load(BEST_MODEL_PATH)
    
    # Ngarko vektorizuesin TF-IDF që është përdorur gjatë trajnimit
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Ngarko emrat e kategorive nga skedari JSON
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]  # labels është lista e emrave të kategorive
    

    # 4. TRANSFORMI I TEKSTIT DHE PARASHIKIMI
    # Transformo tekstin në vektor TF-IDF duke përdorur vektorizuesin e ngarkuar
    # vectorizer.transform e kthen tekstin në një matricë të rrallë (sparse)
    X = vectorizer.transform([text])  # [text] sepse transform pranon një listë teksti
    
    # Parashiko kategorinë duke përdorur modelin
    # model.predict(X) kthen një varg me indekset e kategorive
    # [0] merr vlerën e parë (sepse kemi vetëm një tekst)
    # int() konverton në numër të plotë
    pred_idx = int(model.predict(X)[0])
    
    # Merr emrin e kategorisë duke përdorur indeksin e parashikuar
    predicted_label = labels[pred_idx]
    

    # 5. MARRJA E "TOP 3" PËR MODELE SVM
    # Inicializo top3 si None (do të plotësohet vetëm për modelet SVM)
    top3 = None
    
    # Kontrollo nëse modeli ka atributin decision_function (për SVM)
    if hasattr(model, "decision_function"):
        # Merr skorët e funksionit të vendimit (distancat nga hiperrrafshi)
        # ravel() e sheshon vargun nëse është 2D
        scores = model.decision_function(X).ravel()
        
        # Kontrollo dimensionin e skorëve
        # Në rastin e klasifikimit binar (2 klase), scores mund të jetë një skalar
        if scores.ndim == 0:  # Nëse është skalar (0 dimensionale)
            # Për rastin binar, krijoni një listë me dy skorë
            # Në klasifikim binar, zakonisht kemi një skorë për klasën pozitive
            # dhe negativi i tij për klasën negative
            scores = [scores, -scores]
        
        # Gjej 3 indekset me skorë më të lartë
        # argsort() kthen indekset e renditura në rend rritës
        # [-3:] merr 3 indekset e fundit (më të lartat)
        # [::-1] i kthen në rend zbritës (më i larti i pari)
        top_idx = scores.argsort()[-3:][::-1]
        
        # Krijo listën e "top 3" me emrin e etiketës dhe skorën
        top3 = [{"label": labels[i], "score": float(scores[i])} for i in top_idx]
    
    # 6. FORMATIMI I DALJES
    # Krijo një dictionary me të gjitha informacionet e nevojshme
    output = {
        "text": text,  # Teksti origjinal i dhënë
        "predicted_label": predicted_label,  # Kategoria e parashikuar
        "predicted_index": pred_idx,  # Indeksi i kategorisë së parashikuar
        "top3": top3,  # Lista me 3 parashikimet më të mira (vetëm për SVM)
        "success": True  # Flamur që tregon se parashikimi ishte i suksesshëm
    }
    
    # Printo rezultatin në konsol në format JSON të formatuar mirë
    print(json.dumps(output, indent=2, ensure_ascii=False))


# 7. EKZEKUTIMI KRYESOR
# Kontrollo nëse ky skedër po ekzekutohet direkt (jo si modul)
if __name__ == "__main__":
    # Nis funksionin kryesor
    main()