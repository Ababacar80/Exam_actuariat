# 🏥 Prédiction de Sinistres d'Assurance

Un package Python pour la prédiction de sinistres dans le domaine de l'assurance santé, développé dans le cadre d'un examen actuariel.

## 🎯 Objectif

Ce projet vise à développer des modèles de machine learning pour prédire les montants de sinistres dans l'assurance santé en utilisant des données démographiques et de santé.

## 📋 Fonctionnalités

- **Chargement de données** : Support des fichiers CSV
- **Preprocessing** : Nettoyage et encodage des données
- **Analyse exploratoire** : Corrélations, statistiques descriptives par genre et impact du tabagisme
- **Modélisation** : Algorithmes de boosting avancés (XGBoost, LightGBM)
- **Évaluation** : Métriques complètes et validation croisée


## 📁 Structure du Projet

```
exam-actuariat/
├── src/
│   └── exam_actuariat/
│       ├── __init__.py
│       ├── data_loading.py      # Chargement des données
│       ├── data_processing.py   # Preprocessing
│       ├── exploration.py       # Analyse exploratoire
│       ├── features.py         # Extraction de features
│       ├── models.py          # Modèles ML
│       └── evaluation.py     # Évaluation
├── scripts/
│   └── train.py              # Script d'entraînement
├── data/
│   └──                   # Données brutes
├── models/                   # Modèles sauvegardés
├── tests/                    # Tests unitaires
├── pyproject.toml
└── README.md
```

## 🚀 Utilisation

### 1. Préparer les données
Placez le fichier de données CSV dans `data/`. 

### 2. Exécuter l'analyse complète
```bash
python scripts/train.py
```

### 3. Utilisation programmatique
```python
from src.exam_actuariat import data_loading, data_processing, models, features
from sklearn.model_selection import train_test_split

# Charger les données
df = data_loading.load_raw('data/insurance-demographic-health.csv')

# Preprocessing
df_clean = data_processing.clean_data(df)
df_encoded = data_processing.encodage(df_clean)

# Préparer les données
X = df_encoded.drop(columns=['claim'])
y = df_encoded['claim']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Analyse des features
feature_importances = features.get_feature_importance(X_train, y_train)
print(feature_importances.head())

# Entraîner un modèle
model = models.train_xgboost(X_train, y_train)
results = models.evaluate_model(model, X_test, y_test, model_name='XGBoost')
print(f"MAE: {results['mae']:.2f}, R²: {results['r2']:.3f}")
```

## 📊 Modules Disponibles

### `data_loading`
- `load_raw(filepath)` : Charge les données depuis un fichier CSV

### `data_processing`
- `clean_data(df)` : Nettoie les données (supprime doublons et valeurs manquantes)
- `encodage(df)` : Encode les variables catégorielles

### `exploration`
- `analyze_correlation(df)` : Calcule les corrélations
- `analyze_by_gender(df)` : Statistiques par genre
- `analyze_smoking_impact(df)` : Impact du tabagisme

### `features`
- `get_feature_importance(X, y)` : Calcule l'importance des variables avec Random Forest

### `models`
- `train_xgboost(X_train, y_train)` : Entraîne un modèle XGBoost
- `train_lightgbm(X_train, y_train)` : Entraîne un modèle LightGBM
- `evaluate_model(model, X_test, y_test, model_name)` : Évalue un modèle
- `save_model(model, filepath)` : Sauvegarde un modèle

### `evaluation`
- `compare_models(results_list)` : Compare plusieurs modèles
- `cross_validate_model(model, X, y, cv)` : Validation croisée

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/test_simple.py

# Tests basiques uniquement
pytest tests/test_simple.py::test_basic_python
```

## 📈 Métriques d'Évaluation

Le projet utilise plusieurs métriques pour évaluer les modèles :
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **MSE** (Mean Squared Error) : Erreur quadratique moyenne
- **RMSE** (Root Mean Squared Error) : Racine de l'erreur quadratique moyenne
- **R²** (Coefficient de détermination) : Variance expliquée

## 🔧 Configuration

Les paramètres des modèles peuvent être ajustés directement aux fonctions d'entraînement :

```python
# Exemple de configuration XGBoost
model = models.train_xgboost(
    X_train, y_train,
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)
```

## ⚡ Algorithmes Supportés

### XGBoost (Extreme Gradient Boosting)
- **Avantages** : Très performant, gestion des interactions non-linéaires
- **Usage** : Champion des compétitions Kaggle
- **Installation** : `pip install xgboost`

### LightGBM (Light Gradient Boosting Machine)
- **Avantages** : Rapide, efficace en mémoire
- **Usage** : Optimal pour les gros datasets
- **Installation** : `pip install lightgbm`

## 🤝 Contribution

1. Fork le projet
2. Créer une branche pour votre feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 👨‍💻 Auteur

**Ababacar Sagna**
- Email: ababacarsagna10@gmail.com

## 🐛 Problèmes Connus

- XGBoost et LightGBM doivent être installés séparément
- Le fichier de données doit être au format CSV

## 📚 Documentation

Pour plus de détails sur l'utilisation des modules, consultez les docstrings dans le code.

## 🗃️ Pipeline d'Analyse

Le script `train.py` exécute automatiquement :

1. **Chargement** des données CSV
2. **Nettoyage** (suppression doublons/NaN)
3. **Analyse exploratoire** (corrélations, statistiques par genre, impact tabagisme)
4. **Encodage** des variables catégorielles
5. **Split** train/test (80/20)
6. **Calcul** de l'importance des features
7. **Entraînement** des modèles (XGBoost, LightGBM)
8. **Évaluation** et comparaison
9. **Validation croisée** du meilleur modèle
10. **Sauvegarde** du modèle optimal

## 📊 Exemple de Sortie

```
Chargement des données...
Données chargées: (1340, 11)
Nettoyage des données...
Données nettoyées: (1332, 11)

Analyse des corrélations...
age_claim: -0.029
bmi_claim: 0.200
bloodpressure_claim: 0.531
children_claim: 0.064

Analyse par genre...
Statistiques par genre:
        count          mean           std      min        25%       50%        75%       max
gender                                                                                      
female  662.0  12569.578897  11128.703817  1607.51  4885.1625  9412.965  14454.690  63770.43
male    670.0  14071.891060  12971.546624  1121.87  4676.6400  9439.495  19160.175  62592.87

Analyse de l'impact du tabagisme...
Smoking Impact Analysis
Mean Claim Amount for Smokers: 32050.23
Mean Claim Amount for Non-Smokers: 8475.86
Correlation tabagisme: 0.787

Encodage des variables...
Préparation des données...
Train: (1065, 10), Test: (267, 10)

Calcul de l'importance des features...
Top 5 features:
bloodpressure    0.350
smoker           0.285
bmi              0.180
age              0.125
children         0.060

Entraînement des modèles...
Entraînement XGBoost...
Entraînement LightGBM...

Résultats de comparaison:
  model_name         mae        rmse        r2
0    XGBoost  2845.67  4205.23  0.885
1   LightGBM  3102.14  4598.45  0.863

Validation croisée pour le meilleur modèle: XGBoost
XGBoost Cross-Validation MAE: 2967.42 ± 245.18

==================================================
ANALYSE TERMINÉE AVEC SUCCÈS !
Meilleur modèle: XGBoost
MAE: 2845.67
R²: 0.885
==================================================
```
