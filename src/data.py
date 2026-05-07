from typing import List, Tuple  # Për tipizimin e saktë (type hints) - specifikon se funksioni kthen listë dhe tuple
import pandas as pd  # Për të punuar me të dhëna në formë tabele (DataFrame)
from sklearn.datasets import fetch_20newsgroups  # Për të shkarkuar dataset-in 20 Newsgroups
from .config import RANDOM_SEED  # Importo konstante të përbashkëta nga konfigurimi

# FUNKSIONI KRYESOR PËR NGARKIMIN E DATASET-IT
def load_dataset(categories: List[str]) -> Tuple[pd.DataFrame, list]:
    """Downloads and returns the selected categories of the 20 Newsgroups dataset.

    Returns a DataFrame with columns: text, label_idx and the list of target_names.

    """
        # 1. SHKARKIMI I DATASET-IT 20 NEWSCROUPS
    ds = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    # 2. KONVERTIMI NË DATAFRAME TË PANDAS
    df = pd.DataFrame({ "text": ds.data, "label_idx": ds.target })
    return df, list(ds.target_names)
