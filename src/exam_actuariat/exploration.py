import pandas as pd
import numpy as np

def analyze_correlation(df):
    """Analyze correlation between features and target."""
    correlations = {}
    
    # Calculate correlations with numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if 'claim' in numerical_cols:
        target_correlations = df[numerical_cols].corr()['claim']
        
        for col in numerical_cols:
            if col != 'claim':
                correlations[f'{col}_claim'] = target_correlations[col]
    
    return correlations

def analyze_by_gender(df):
    """Analyze statistics by gender."""
    if 'gender' in df.columns and 'claim' in df.columns:
        gender_stats = df.groupby('gender')['claim'].describe()
        return gender_stats
    return None

def analyze_smoking_impact(df):
    """Analyze the impact of smoking on claim amounts."""
    if 'smoker' in df.columns and 'claim' in df.columns:
        # Calculate mean claim amounts for smokers and non-smokers
        smoker_mean = df[df['smoker'] == 'Yes']['claim'].mean() if 'Yes' in df['smoker'].values else 0
        non_smoker_mean = df[df['smoker'] == 'No']['claim'].mean() if 'No' in df['smoker'].values else 0
        
        # Calculate correlation between smoking and claim amount
        smoking_binary = df['smoker'].map({'Yes': 1, 'No': 0})
        correlation_smoker_claim = smoking_binary.corr(df['claim'])
        
        return {
            'mean_smoker': smoker_mean,
            'mean_non_smoker': non_smoker_mean,
            'correlation_smoker_claim': correlation_smoker_claim
        }
    return None