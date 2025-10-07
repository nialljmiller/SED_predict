import configparser
import os
from data_loader import load_and_split_data
from plots import (
    plot_actual_vs_predicted, 
    plot_feature_importance, 
    plot_loss_curves, 
    plot_residuals, 
    plot_error_distribution, 
    plot_features_vs_error, 
    plot_spatial_error
)

def main():
    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('inlist')
    
    # Extract parameters
    data_file = config['paths']['data_file']
    learning_rate = float(config['hyperparameters']['learning_rate'])
    n_estimators = int(config['hyperparameters']['n_estimators'])
    max_depth = int(config['hyperparameters']['max_depth'])
    test_size = float(config['general']['test_size'])
    val_size = float(config['general']['val_size'])
    random_state = int(config['general']['random_state'])
    output_dir = config['general']['output_dir']
    model_type = config['general'].get('model_type', 'xgboost')  # Default to xgboost
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        data_file, test_size, val_size, random_state
    )
    
    # Train and evaluate based on model_type
    scaler = None
    history = None
    if model_type == 'ngboost':
        from ngboost_model import train_ngboost, evaluate_model
        model, history = train_ngboost(
            X_train, y_train, X_val, y_val, learning_rate, n_estimators, max_depth
        )
        predictions = evaluate_model(model, X_test, y_test)
    elif model_type == 'xgboost':
        from xgboost_model import train_xgboost, evaluate_model
        model, history = train_xgboost(
            X_train, y_train, X_val, y_val, learning_rate, n_estimators, max_depth
        )
        predictions = evaluate_model(model, X_test, y_test)
    elif model_type == 'mlp':
        from mlp_model import train_mlp, evaluate_model
        model, scaler, history = train_mlp(
            X_train, y_train, X_val, y_val, learning_rate, n_estimators, max_depth
        )
        predictions = evaluate_model(model, scaler, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'xgboost', 'ngboost', or 'mlp'.")
    
    # Generate plots
    plot_actual_vs_predicted(y_test, predictions, os.path.join(output_dir, 'actual_vs_predicted.png'))
    plot_feature_importance(
        model, 
        X_train.columns, 
        os.path.join(output_dir, 'feature_importance.png'),
        X_train=X_train if model_type in ['mlp', 'ngboost'] else None,
        y_train=y_train if model_type in ['mlp', 'ngboost'] else None,
        scaler=scaler if model_type == 'mlp' else None
    )
    plot_loss_curves(history, os.path.join(output_dir, 'loss_curves.png'))
    plot_residuals(y_test, predictions, os.path.join(output_dir, 'residuals.png'))
    plot_error_distribution(y_test, predictions, os.path.join(output_dir, 'error_distribution.png'))
    plot_features_vs_error(X_test, y_test, predictions, os.path.join(output_dir, 'features_vs_error.png'))
    plot_spatial_error(X_test, y_test, predictions, os.path.join(output_dir, 'spatial_error.png'))
    
    print("Training and evaluation complete. Plots saved in", output_dir)

if __name__ == "__main__":
    main()