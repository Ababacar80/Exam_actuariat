import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df_encoded):
    """Heatmap to visualize correlation between features."""
    numerical_features = df_encoded.select_dtypes(exclude=['object']).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_encoded[numerical_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_age_vs_claim(df_clean):
    """Scatter plot of age vs claim amount."""
    if 'age' in df_clean.columns and 'claim' in df_clean.columns:
        plt.figure(figsize=(10, 6))
        if 'gender' in df_clean.columns :
            sns.scatterplot(data=df_clean, x='age', y='claim', hue='gender', style='smoker', alpha=0.7)
        else:
            sns.scatterplot(data=df_clean, x='age', y='claim', alpha=0.7)
        plt.title('Age vs Claim Amount')    
        plt.xlabel('Age')
        plt.ylabel('Claim Amount')
        plt.show()
        
def plot_smoking_impact(smoking_stats):
    """Bar plot to visualize the impact of smoking on claim amounts."""
    if smoking_stats:
        plt.figure(figsize=(8, 5))
        plt.bar(['Smoker', 'Non-Smoker'], [smoking_stats['mean_smoker'], smoking_stats['mean_non_smoker']], color=['red', 'blue'])
        plt.title('Cout Moyen selon le tabagisme ')
        plt.ylabel('Mean Claim Amount ')
        plt.xlabel('Smoking Status')
        plt.show()
    else:
        print("No data available for smoking impact analysis.")

def plot_feature_importance(feature_importances, top_n=10):
    """Bar plot to visualize feature importance."""
    plt.figure(figsize=(10, 6))
    feature_importances.head(top_n).plot(kind='barh', color='skyblue')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} des variables les plus importantes')
    plt.show()

    