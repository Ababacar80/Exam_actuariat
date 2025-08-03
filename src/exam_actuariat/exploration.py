import pandas as pd
import numpy as np

def analyze_correlation(df_clean):

    correlations = {}
    """Analyze the correlation between numerical features in the DataFrame."""
    if "age" in df_clean.columns and "claim" in df_clean.columns:
        correlations['age_claim'] = df_clean['age'].corr(df_clean['claim'])
    if "bmi" in df_clean.columns and "claim" in df_clean.columns:
        correlations['bmi_claim'] = df_clean['bmi'].corr(df_clean['claim'])
    if "bloodpressure" in df_clean.columns and "claim" in df_clean.columns:
        correlations['bloodpressure_claim'] = df_clean['bloodpressure'].corr(df_clean['claim'])

    return correlations

def analyze_by_gender(df_clean):
    """Analyze the average claim amount"""
    if "gender" in df_clean.columns and "claim" in df_clean.columns:
        return df_clean.groupby('gender')['claim'].agg([mean, np.std, np.var])
    return None

def analyze_smoking_impact(df_clean):
    """Analyze the impact of smoking on claim amounts."""
    if "smoker" in df_clean.columns and "claim" in df_clean.columns:
        mean_smoker =df_clean.loc[df_clean["smoker"]=='Yes  ', 'claim'].mean()
        mean_non_smoker = df_clean.loc[df_clean["smoker"]=='No  ', 'claim'].mean()

        df_clean['smoker_bin'] = df_clean['smoker'].map({'Yes  ': 1, 'No  ': 0})
        corr_smoker  = df_clean['smoker_bin'].corr(df_clean['claim'])

        return {
            'mean_smoker': mean_smoker,
            'mean_non_smoker': mean_non_smoker,
            'correlation_smoker_claim': corr_smoker
        }
    return None