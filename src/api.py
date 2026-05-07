import json  # Për të lexuar/ruajtur skedarë JSON
import joblib  # Për të ngarkuar modelet e ruajtura
import numpy as np  # Për operacione numerike dhe manipulim me vargje
from fastapi import FastAPI  # Për të krijuar API-n REST
from fastapi.middleware.cors import CORSMiddleware  # Për të lejuar komunikimin ndër-origjine
from pydantic import BaseModel  # Për të definuar skemat e të dhënave
from .config import BEST_MODEL_PATH, VECTORIZER_PATH, LABELS_PATH  # Importo rrugët nga konfigurimi
from scipy.special import softmax  # Funksion për të konvertuar skorët në probabilitete


# Pragu minimal i besimit për të pranuar një parashikim
CONF_THRESHOLD = 0.3  # 30% - nëse probabiliteti është më i ulët, tekst klasifikohet si "UNKNOWN"


# INICIALIZIMI I APLIKACIONIT FASTAPI
app = FastAPI(
    title="Text Classification API",  # Emri i API-së që shfaqet në dokumentacion
    version="1.0.0"  # Versioni i API-së
)


# KONFIGURIMI I CORS (Cross-Origin Resource Sharing)
# Lejon frontend-in të komunikojë me API-n edhe nëse janë në server të ndryshëm
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lejo të gjitha origjinat (në prodhim kjo duhet të kufizohet)
    allow_credentials=True,  # Lejo kredencialet (cookies, headers autentikimi)
    allow_methods=["*"],  # Lejo të gjitha metodat HTTP (GET, POST, etj.)
    allow_headers=["*"],  # Lejo të gjitha header-at
)


# DEFINIMI I MODELEVE TË TË DHËNAVE ME PYDANTIC
class PredictIn(BaseModel):
    """Modeli për të dhënat hyrëse të API-së.
    
    Attributes:
        text (str): Teksti që do të klasifikohet
    """
    text: str


class PredictOut(BaseModel):
    """Modeli për të dhënat dalëse të API-së.
    
    Attributes:
        predicted_label (str): Etiketa e parashikuar
        predicted_index (int): Indeksi i etiketës së parashikuar (0-3)
        top3 (list | None): Lista me 3 parashikimet më të mira me probabilitetet e tyre
    """
    predicted_label: str  # Emri i kategorisë së parashikuar
    predicted_index: int  # Indeksi numerik i kategorisë (0, 1, 2, ose 3)
    top3: list | None = None  # Lista opsionale me 3 rezultatet më të mira


# NGARKIMI I ARTEFAKTEVE TË MODELIT
# Ngarko modelin e makinës mësimore të ruajtur me joblib
_model = joblib.load(BEST_MODEL_PATH)

# Ngarko vektorizuesin TF-IDF të ruajtur
_vectorizer = joblib.load(VECTORIZER_PATH)

# Ngarko emrat e kategorive (etiketat) nga skedari JSON
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    _labels = json.load(f)["labels"]  # _labels është lista: ["comp.sys.mac.hardware", "rec.sport.baseball", ...]


# ENDPOINT-I I SHËNDETIT (HEALTH CHECK)
@app.get("/health")
def health():
    """Endpoint për të kontrolluar nëse API-ja është gjallë dhe funksionale.
    
    Returns:
        dict: Statusi "ok" nëse API-ja funksionon
    """
    return {"status": "ok"}  # Përgjigje e thjeshtë për load balancers dhe monitoring


# ENDPOINT-I KRYESOR PËR PARASHIKIM
@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    """Kryen klasifikimin e tekstit dhe kthen kategorinë më të mundshme.
    
    Args:
        payload (PredictIn): Objekt me fushën 'text' që përmban tekstin për të klasifikuar
    
    Returns:
        PredictOut: Objekt me parashikimin, indeksin dhe listën top3
    """
    # 1. TRANSFORMO TEKSTIN NË VEKTOR NUMERIK ME TF-IDF
    # _vectorizer.transform e kthen tekstin në një matricë të rrallë (sparse) TF-IDF
    X = _vectorizer.transform([payload.text])
    
    # 2. INICIALIZO DICTIONARY PËR REZULTATET
    # Krijon një dictionary me vlera default para se të plotësohet
    out = {"predicted_label": "", "predicted_index": -1, "top3": None}
    
    # 3. PARASHIKIM PËR MODELE ME "predict_proba" (MultinomialNB, LogisticRegression)
    # Kontrollon nëse modeli ka metodën predict_proba (kthe probabilitete)
    if hasattr(_model, "predict_proba"):
        # Merr probabilitetet për të gjitha kategoritë
        probs = _model.predict_proba(X)[0]  # [0] për të marrë vargun e parë (vetëm një mostër)
        
        # Gjej 3 indekset me probabilitete më të larta (renditje zbritëse)
        # argsort() kthen indekset e renditura në rend rritës
        # [-3:] merr 3 elementët e fundit (më të lartët)
        # [::-1] i kthen në rend zbritës (më i larti i pari)
        top_k = probs.argsort()[-3:][::-1]
        
        # Krijon listën top3 me emra etiketash dhe probabilitete
        out["top3"] = [{"label": _labels[i], "prob": float(probs[i])} for i in top_k]
        
        # Kontrollon nëse probabiliteti më i lartë kalon pragun e besimit
        if out["top3"][0]["prob"] >= CONF_THRESHOLD:
            # Cakto kategorinë më të mundshme si parashikim
            out["predicted_label"] = out["top3"][0]["label"]
            out["predicted_index"] = top_k[0]  # Indeksi i kategorisë parashikuar
        else:
            # Probabiliteti shumë i ulët, klasifikoj si "UNKNOWN"
            out["predicted_label"] = "UNKNOWN"
    
    # 4. PARASHIKIM PËR MODELE ME "decision_function" (LinearSVC)
    # Për modelet SVM që nuk japin probabilitete native
    elif hasattr(_model, "decision_function"):
        # Merr skorët e funksionit të vendimit (distancat nga hiperrrafshi)
        scores = _model.decision_function(X)
        
        # Kontrollon dimensionin e skorëve
        # Në disa raste, scores mund të jetë 1D (për klasifikim binar)
        if scores.ndim == 1:
            # Riformaton në 2D (1 rresht, disa kolona) për konsistencë
            scores = scores.reshape(1, -1)
        
        # Konverton skorët në probabilitete duke përdorur softmax
        # Softmax siguron që shuma e të gjitha probabiliteteve të jetë 1
        probs = softmax(scores[0])
        
        # Gjej 3 indekset më të larta si më parë
        top_k = probs.argsort()[-3:][::-1]
        
        # Krijon listën top3
        out["top3"] = [{"label": _labels[i], "prob": float(probs[i])} for i in top_k]
        
        # Kontrollon pragun e besimit
        if probs[top_k[0]] >= CONF_THRESHOLD:
            out["predicted_label"] = _labels[top_k[0]]
            out["predicted_index"] = int(top_k[0])
        else:
            out["predicted_label"] = "UNKNOWN"
    

    return out