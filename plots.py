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
    For XGBoost, uses built-in feature_importances_.
    For NGBoost and MLP, uses permutation importance.
    """
    if hasattr(model, 'feature_importances_'):
        # XGBoost case
        importance_values = model.feature_importances_
    else:
        # NGBoost or MLP case: Use permutation importance
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided for permutation importance")
        X_perm = scaler.transform(X_train) if scaler is not None else X_train
        result = permutation_importance(model, X_perm, y_train, n_repeats=10, random_state=42)
        importance_values = result.importances_mean
    
    # Ensure importance_values is 1D and matches feature_names length
    importance_values = np.array(importance_values).flatten()
    if len(importance_values) == len(feature_names):
        
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
        plt.title('Feature Importance for MIPS24 Prediction')
        plt.savefig(save_path)
        plt.close()

def plot_loss_curves(history, save_path):
    """
    Plots train and validation RMSE over iterations with a logarithmic y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['train'], label='Train RMSE')
    plt.plot(history['val'], label='Validation RMSE')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Iterations')
    plt.ylabel('RMSE (Log Scale)')
    plt.title('Learning Curves (RMSE)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_test, predictions, save_path):
    """
    Plots residuals (predicted - actual) against actual values.
    """
    residuals = predictions - y_test
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=residuals, alpha=0.6)
    plt.axhline(0, color='r', ls='--')
    plt.xlabel('Actual MIPS24 Magnitude')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(y_test, predictions, save_path):
    """
    Plots the distribution of residuals.
    """
    residuals = predictions - y_test
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Prediction Errors')
    plt.savefig(save_path)
    plt.close()

def plot_features_vs_error(X_test, y_test, predictions, save_path):
    """
    Plots each feature against absolute prediction error.
    """
    residuals = np.abs(predictions - y_test)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i, col in enumerate(X_test.columns):
        sns.scatterplot(x=X_test[col], y=residuals, ax=axs[i], alpha=0.5)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Absolute Error')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spatial_error(X_test, y_test, predictions, save_path):
    """
    Plots galactic coordinates colored by absolute prediction error.
    """
    residuals = np.abs(predictions - y_test)
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_test['GAL_LONG'], X_test['GAL_LAT'], c=residuals, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Absolute Error')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.title('Spatial Distribution of Prediction Errors')
    plt.savefig(save_path)
    plt.close()