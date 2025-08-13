import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def compare_models(results_list):
    """Compare multiple model results."""
    if not results_list:
        raise ValueError("Results list cannot be empty")
    
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values(['r2', 'mae'], ascending=[False, True])
    
    return comparison_df

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_absolute_error'):
    """Perform cross-validation on a model."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    if scoring == 'neg_mean_absolute_error':
        scores = -scores
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    return mean_score, std_score