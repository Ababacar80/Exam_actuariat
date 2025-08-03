import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exam_actuariat import data_processing, exploration, features, models, evaluation


class TestDataProcessing:
    """Tests pour le module data_processing."""
    
    def test_clean_data(self):
        """Test de la fonction clean_data."""
        # Créer des données de test avec des doublons et des valeurs manquantes
        df = pd.DataFrame({
            'age': [25, 30, 25, 35, np.nan],
            'gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
            'claim': [1000, 2000, 1000, 3000, 4000]
        })
        
        df_clean = data_processing.clean_data(df)
        
        # Vérifier qu'il n'y a plus de valeurs manquantes
        assert df_clean.isnull().sum().sum() == 0
        
        # Vérifier qu'il n'y a plus de doublons
        assert df_clean.duplicated().sum() == 0
        
        # Vérifier que la taille est correcte (4 lignes - 1 NaN - 1 doublon = 2)
        assert len(df_clean) == 2
    
    def test_encodage(self):
        """Test de la fonction encodage."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'gender': ['Male', 'Female', 'Male'],
            'smoker': ['Yes', 'No', 'Yes'],
            'claim': [1000, 2000, 3000]
        })
        
        df_encoded = data_processing.encodage(df)
        
        # Vérifier que les colonnes catégorielles ont été encodées
        assert df_encoded['gender'].dtype in [np.int64, np.int32]
        assert df_encoded['smoker'].dtype in [np.int64, np.int32]
        
        # Vérifier que la colonne claim est toujours présente
        assert 'claim' in df_encoded.columns


class TestExploration:
    """Tests pour le module exploration."""
    
    def test_analyze_correlation(self):
        """Test de la fonction analyze_correlation."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'bmi': [22.5, 25.0, 27.5, 30.0],
            'claim': [1000, 2000, 3000, 4000]
        })
        
        correlations = exploration.analyze_correlation(df)
        
        # Vérifier que les corrélations sont calculées
        assert 'age_claim' in correlations
        assert 'bmi_claim' in correlations
        
        # Vérifier que les valeurs sont des nombres
        assert isinstance(correlations['age_claim'], float)
        assert isinstance(correlations['bmi_claim'], float)
    
    def test_analyze_by_gender(self):
        """Test de la fonction analyze_by_gender."""
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'claim': [1000, 2000, 1500, 2500]
        })
        
        result = exploration.analyze_by_gender(df)
        
        # Vérifier que le résultat n'est pas None
        assert result is not None
        
        # Vérifier que les statistiques sont calculées pour les deux genres
        assert 'Male' in result.index
        assert 'Female' in result.index


class TestFeatures:
    """Tests pour le module features."""
    
    def test_get_feature_importance(self):
        """Test de la fonction get_feature_importance."""
        # Créer des données synthétiques
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'claim': np.random.randn(100) * 100 + 1000
        })
        
        importance = features.get_feature_importance(df, target='claim')
        
        # Vérifier que l'importance est calculée pour toutes les features
        assert len(importance) == 3
        assert 'feature1' in importance.index
        assert 'feature2' in importance.index
        assert 'feature3' in importance.index
        
        # Vérifier que les importances sont des nombres positifs
        assert all(importance >= 0)


class TestModels:
    """Tests pour le module models."""
    
    def test_train_random_forest(self):
        """Test de l'entraînement Random Forest."""
        # Créer des données synthétiques
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = np.random.randn(100) * 100 + 1000
        
        model = models.train_random_forest(X, y)
        
        # Vérifier que le modèle a été entraîné
        assert hasattr(model, 'predict')
        
        # Vérifier que le modèle peut faire des prédictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_evaluate_model(self):
        """Test de l'évaluation de modèle."""
        # Créer des données synthétiques
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        y = np.random.randn(50) * 100 + 1000
        
        # Entraîner un modèle simple
        model = models.train_random_forest(X, y)
        
        # Évaluer le modèle
        results = models.evaluate_model(model, X, y, model_name='Test Model')
        
        # Vérifier que toutes les métriques sont présentes
        assert 'model_name' in results
        assert 'mae' in results
        assert 'mse' in results
        assert 'rmse' in results
        assert 'r2' in results
        
        # Vérifier que les valeurs sont numériques
        assert isinstance(results['mae'], float)
        assert isinstance(results['mse'], float)
        assert isinstance(results['rmse'], float)
        assert isinstance(results['r2'], float)


class TestEvaluation:
    """Tests pour le module evaluation."""
    
    def test_compare_models(self):
        """Test de la comparaison de modèles."""
        results = [
            {
                'model_name': 'Model1',
                'mae': 100.0,
                'mse': 15000.0,
                'rmse': 122.47,
                'r2': 0.85
            },
            {
                'model_name': 'Model2',
                'mae': 120.0,
                'mse': 18000.0,
                'rmse': 134.16,
                'r2': 0.80
            }
        ]
        
        comparison = evaluation.compare_models(results)
        
        # Vérifier que le DataFrame de comparaison est créé
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        
        # Vérifier que les colonnes sont présentes
        assert 'model_name' in comparison.columns
        assert 'mae' in comparison.columns
        assert 'r2' in comparison.columns
    
    def test_cross_validate_model(self):
        """Test de la validation croisée."""
        # Créer des données synthétiques
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = np.random.randn(100) * 100 + 1000
        
        # Entraîner un modèle
        model = models.train_random_forest(X, y)
        
        # Effectuer la validation croisée
        mean_score, std_score = evaluation.cross_validate_model(model, X, y, cv=3)
        
        # Vérifier que les scores sont numériques
        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)
        
        # Vérifier que l'écart-type est positif
        assert std_score >= 0


# Fixtures pour les tests
@pytest.fixture
def sample_dataframe():
    """Fixture pour créer un DataFrame de test."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'bmi': [22.5, 25.0, 27.5, 30.0, 32.5],
        'smoker': ['No', 'Yes', 'No', 'Yes', 'No'],
        'claim': [1000, 2500, 1800, 3200, 2100]
    })


def test_full_pipeline(sample_dataframe):
    """Test du pipeline complet."""
    # Nettoyage
    df_clean = data_processing.clean_data(sample_dataframe)
    
    # Encodage
    df_encoded = data_processing.encodage(df_clean)
    
    # Feature importance
    importance = features.get_feature_importance(df_encoded, target='claim')
    
    # Vérifier que le pipeline fonctionne
    assert len(df_encoded) <= len(sample_dataframe)  # Après nettoyage
    assert len(importance) == len(df_encoded.columns) - 1  # Moins la target
    
    # Vérifier que les données encodées sont numériques
    for col in df_encoded.columns:
        assert df_encoded[col].dtype in [np.int64, np.int32, np.float64, np.float32]