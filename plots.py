import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def plot_actual_vs_predicted(y_test, predictions, save_path):
    """
    Plots actual vs. predicted values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual MIPS24 Magnitude')
    plt.ylabel('Predicted MIPS24 Magnitude')
    plt.title('Actual vs. Predicted MIPS24 Magnitudes')
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path, X_train=None, y_train=None, scaler=None):
    """
    Plots feature importance. Handles XGBoost, NGBoost, and MLP models.
    For NGBoost, averages importances across internal boosters.
    For MLP, uses permutation importance.
    """
    if hasattr(model, 'feature_importances_'):
        # XGBoost case
        importance_values = model.feature_importances_
    elif hasattr(model, 'scalers_'):
        # NGBoost case: Average importances from scalers
        importances = [scaler.feature_importances_ for scaler in model.scalers_]
        importance_values = np.mean(importances, axis=0)
    else:
        # MLP case: Use permutation importance
        if X_train is None or y_train is None or scaler is None:
            raise ValueError("X_train, y_train, and scaler must be provided for MLP feature importance")
        X_train_scaled = scaler.transform(X_train)
        result = permutation_importance(model, X_train_scaled, y_train, n_repeats=10, random_state=42)
        importance_values = result.importances_mean
    
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title('Feature Importance for MIPS24 Prediction')
    plt.savefig(save_path)
    plt.close()