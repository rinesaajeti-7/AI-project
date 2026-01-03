# Global configuration for the project

RANDOM_SEED = 42
TEST_SIZE = 0.20

# We'll use a 4-class subset of the 20 Newsgroups dataset to fit the 2–5 category requirement.
# These categories are distinct and produce meaningful classification boundaries.
CATEGORIES = [
    "comp.sys.mac.hardware",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.mideast",
]

# Feature extraction
NGRAM_RANGE = (1, 2)         # unigrams + bigrams
MIN_DF = 2                   # ignore very rare terms
MAX_DF = 0.90                # ignore extremely frequent terms
SUBLINEAR_TF = True          # log scale term frequency

# Paths
ARTIFACT_DIR = "artifacts"
VECTORIZER_PATH = f"{ARTIFACT_DIR}/tfidf_vectorizer.joblib"
BEST_MODEL_PATH = f"{ARTIFACT_DIR}/best_model.joblib"
LABELS_PATH = f"{ARTIFACT_DIR}/labels.json"
METRICS_PATH = f"{ARTIFACT_DIR}/metrics.json"
REPORT_PATH = f"{ARTIFACT_DIR}/classification_report.txt"
CONFUSION_MATRIX_PATH = f"{ARTIFACT_DIR}/confusion_matrix.png"
