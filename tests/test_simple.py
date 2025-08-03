"""
Tests simples pour vérifier que pytest fonctionne et que les modules peuvent être importés.
"""

def test_basic_python():
    """Test très basique pour vérifier que pytest fonctionne."""
    assert 1 + 1 == 2
    assert "hello" == "hello"

def test_imports():
    """Test que les imports Python de base fonctionnent."""
    import sys
    import os
    assert sys.version_info.major >= 3
    assert os.path.exists(".")

def test_pandas_import():
    """Test que pandas peut être importé."""
    import pandas as pd
    import numpy as np
    
    # Créer un DataFrame simple
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']

def test_sklearn_import():
    """Test que scikit-learn peut être importé."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    
    # Test simple
    import numpy as np
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == 3
    mae = mean_absolute_error(y, predictions)
    assert mae >= 0