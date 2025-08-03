import pandas as pd 
from pathlib import Path

def load_raw(filepath: str | Path) -> pd.DataFrame:
    """Charger les donnees excel"""
    df=pd.read_excel(filepath)
    return df
