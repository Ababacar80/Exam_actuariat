import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def get_feature_importance(X, y):
    """Calculate feature importance using Random Forest."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    feature_importances = pd.Series(
        model.feature_importances_, 
        index=X.columns
    ).sort_values(ascending=False)
    
    return feature_importances