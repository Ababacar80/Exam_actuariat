import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """Clean the data by removing duplicates and handling missing values."""
    df_clean = df.dropna().drop_duplicates()
    return df_clean

def encodage(df):
    """Encode categorical variables using label encoding."""
    df_clean = clean_data(df)
    df_encoded = df_clean.copy()
    
    # Encode categorical variables
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded