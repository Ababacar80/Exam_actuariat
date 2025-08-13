import pandas as pd 
from pathlib import Path

def load_raw(filepath: str | Path) -> pd.DataFrame:
    """Charger les donnees CSV."""
    df = pd.read_csv(filepath)
    return df