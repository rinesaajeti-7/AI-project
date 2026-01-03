from typing import List, Tuple
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from .config import RANDOM_SEED

def load_dataset(categories: List[str]) -> Tuple[pd.DataFrame, list]:
    """Downloads and returns the selected categories of the 20 Newsgroups dataset.

    Returns a DataFrame with columns: text, label_idx and the list of target_names.

    """
    ds = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    df = pd.DataFrame({ "text": ds.data, "label_idx": ds.target })
    return df, list(ds.target_names)
