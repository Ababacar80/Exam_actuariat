# 🏥 Exam Actuariat - Prédiction de Sinistres d'Assurance

Un package Python pour la prédiction de sinistres dans le domaine de l'assurance santé, développé dans le cadre d'un examen actuariel.

## 🎯 Objectif

Ce projet vise à développer des modèles de machine learning pour prédire les montants de sinistres dans l'assurance santé en utilisant des données démographiques et de santé.

## 📋 Fonctionnalités

- **Chargement de données** : Support des fichiers Excel et CSV
- **Preprocessing** : Nettoyage, encodage et normalisation des données
- **Analyse exploratoire** : Corrélations, statistiques descriptives et détection d'outliers
- **Visualisation** : Graphiques interactifs et heatmaps
- **Modélisation** : Algorithmes de boosting avancés (XGBoost, LightGBM)
- **Évaluation** : Métriques complètes et validation croisée

## 🔧 Installation

### Prérequis
- Python 3.11 ou supérieur
- Poetry (recommandé) ou pip

### Installation avec Poetry
```bash
# Cloner le repository
git clone <url-du-repo>
cd exam-actuariat

# Installer les dépendances
poetry install

# Activer l'environnement virtuel
poetry shell
```

### Installation avec pip
```bash
# Cloner le repository
git clone <url-du-repo>
cd exam-actuariat

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -e .
```

### Dépendances optionnelles
```bash
# Pour le développement
poetry install --extras dev

# Pour la documentation
poetry install --extras docs

# Installer les algorithmes de boosting
pip install xgboost lightgbm
```

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
│       ├── visualization.py    # Visualisations
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
Placez le fichier de données dans `data/`. 

### 2. Exécuter l'analyse complète
```bash
python scripts/train.py
```

### 3. Utilisation programmatique
```python
from src.exam_actuariat import data_loading, data_processing, models, features

# Charger les données
df = data_loading.load_raw('data/insurance-demographic-health.csv')

# Preprocessing
df_clean = data_processing.clean_data(df)
df_encoded = data_processing.encodage(df_clean)

# Analyse des features
feature_importances = features.get_feature_importance(df_encoded, target='claim')
print(feature_importances.head())

# Entraîner un modèle
from sklearn.model_selection import train_test_split
X = df_encoded.drop(columns=['claim'])
y = df_encoded['claim']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.train_xgboost(X_train, y_train)
results = models.evaluate_model(model, X_test, y_test)
print(results)
```

## 📊 Modules Disponibles

### `data_loading`
- `load_raw(filepath)` : Charge les données depuis un fichier Excel/CSV

### `data_processing`
- `clean_data(df)` : Nettoie les données (supprime doublons et valeurs manquantes)
- `encodage(df)` : Encode les variables catégorielles
- `scale_features(df)` : Normalise les features numériques
- `apply_smote(X, y)` : Applique SMOTE pour l'équilibrage des classes

### `exploration`
- `analyze_correlation(df)` : Calcule les corrélations
- `analyze_by_gender(df)` : Statistiques par genre
- `analyze_smoking_impact(df)` : Impact du tabagisme
- `detect_outliers(df, column)` : Détection d'outliers

### `features`
- `get_feature_importance(df, target)` : Calcule l'importance des variables avec Random Forest

### `models`
- `train_xgboost(X, y)` : Entraîne un modèle XGBoost
- `train_lightgbm(X, y)` : Entraîne un modèle LightGBM
- `train_linear_regression(X, y)` : Entraîne une régression linéaire
- `evaluate_model(model, X, y)` : Évalue un modèle
- `save_model(model, filepath)` : Sauvegarde un modèle
- `load_model(filepath)` : Charge un modèle

### `evaluation`
- `compare_models(results)` : Compare plusieurs modèles
- `cross_validate_model(model, X, y)` : Validation croisée
- `plot_predictions_vs_actual(y_true, y_pred)` : Visualise les prédictions

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Avec couverture de code
pytest --cov=src/exam_actuariat
```

## 📈 Métriques d'Évaluation

Le projet utilise plusieurs métriques pour évaluer les modèles :
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **MSE** (Mean Squared Error) : Erreur quadratique moyenne
- **RMSE** (Root Mean Squared Error) : Racine de l'erreur quadratique moyenne
- **R²** (Coefficient de détermination) : Variance expliquée
- **MAPE** (Mean Absolute Percentage Error) : Erreur absolue en pourcentage

## 🔧 Configuration

Les paramètres des modèles peuvent être ajustés dans le script `train.py` ou passés directement aux fonctions d'entraînement :

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
- **Installation** : `poetry add xgboost`

### LightGBM (Light Gradient Boosting Machine)
- **Avantages** : Rapide, efficace en mémoire
- **Usage** : Optimal pour les gros datasets
- **Installation** : `poetry add lightgbm`

### Régression Linéaire
- **Avantages** : Simple, interprétable
- **Usage** : Baseline et comparaison

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
- Les visualisations nécessitent un environnement graphique

## 📚 Documentation

Pour plus de détails sur l'utilisation des modules, consultez les docstrings dans le code ou générez la documentation :

```bash
# Installer les dépendances de documentation
poetry install --extras docs

# Générer la documentation
cd docs
make html
```