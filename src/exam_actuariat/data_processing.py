import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def clean_data(df):
    """Clean the data by removing duplicates and handling missing values."""
    
    # Drop rows with NaN values in other columns
    df_clean = df.dropna().drop_duplicates()
    
    return df_clean

def encodage(df):
    clean_data(df)  # Clean the data before encoding
    df_encoded = df_clean.copy()
    """Encode categorical variables using one-hot encoding."""
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded