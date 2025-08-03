import pathlib as Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from exam_actuariat import(data_loading, data_processing, features, visualization,exploration)

def main():
    # Configuration
    DATA_PATH = ('data/raw/insurance_demographic-health.csv')
    TARGET = 'claim'

    # Load data
    df = data_loading.load_raw(DATA_PATH)

    # Nettoyage
    df_clean= preprocessing.clean_data(df)  

    # Analyse correlations
    correlations = exploration.analyze_correlation(df_clean)
    for key, value in correlations.items():
        print(f"{key}: {value:.3f}")

    # Analyse par genre
    gender_stats = exploration.analyze_by_gender(df_clean)
    if gender_stats is not None:
        print("Average claim amount")
        print(gender_stats)

    # Analyse de l'impact du tabagisme
    smoking_stats = exploration.analyze_smoking_impact(df_clean)
    if smoking_stats:
        print("Smoking Impact Analysis")
        print(f"Mean Claim Amount for Smokers: {smoking_stats['mean_smoker']:.2f}")
        print(f"Mean Claim Amount for Non-Smokers: {smoking_stats['mean_non_smoker']:.2f}")
        print(f"Correlation between Smoking and Claim Amount: {smoking_stats['correlation_smoker_claim']:.3f}")
        print(f"Correlation tabagisme": {smoking_stats[correlation]})

    # Encodage des variables et feature importance
    df_encoded = data_processing.encodage(df_clean)
    feature_importances = features.get_feature_importance(df_encoded, target=TARGET)
    print("Feature Importances:")
    print(feature_importances.head())

    # Visualisation
    visualization.plot_correlation(df_encoded)
    visualization.plot_age_vs_claim(df_clean)
    visualization.plot_smoking_impact(smoking_stats)
    visualization.plot_feature_importance(feature_importances)

    # Modelisation
    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Entrainement du modèle XGBoost
    xgb_model = models.train_xgboost(X_train, y_train)

    # Entrainement du modèle LightGBM
    lgb_model = models.train_lightgbm(X_train, y_train)

    #Evaluation des modèles
    results = []
    results.append(models.evaluate_model(xgb_model, X_test, y_test, model_name='XGBoost'))
    results.append(models.evaluate_model(lgb_model, X_test, y_test, model_name='LightGBM')) 

    comparaison = evaluation.compare_models(results)    
    print("Model Comparaison Results:")
    print(comparaison)

    # Validation croisée    
    mae_cv_xgb, std_cv_xgb = evaluation.cross_validate_model(xgb_model, X, y, cv=5) 
    mae_cv_lgb, std_cv_lgb = evaluation.cross_validate_model(lgb_model, X, y, cv=5)
    print(f"XGBoost Cross-Validation MAE: {mae_cv_xgb:.2f} ± {std_cv_xgb:.2f}")
    print(f"LightGBM Cross-Validation MAE: {mae_cv_lgb:.2f} ± {std_cv_lgb:.2f}")    


    print("Analyse terminée avec succès !")

if __name__ == "__main__":
    main()
    # Exécute le script principal
    # pour lancer l'analyse des données et l'entraînement des modèles.
