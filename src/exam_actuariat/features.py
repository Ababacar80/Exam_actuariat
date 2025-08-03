import pandas as pd
import sklearn.ensemble import RandomForestRegressor

def get_feature_importance(df_encoded, target='claim'):
    """Calculate feature importance using Random Forest."""
    if target not in df_clean.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    # Separate features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    
    # Initialize Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    model.fit(X, y)
    
    # Get feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return feature_importances