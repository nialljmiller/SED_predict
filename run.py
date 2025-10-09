import configparser
import os
from data_loader import load_and_split_data
from posterior import generate_posterior  # Import the new function
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd 

from plots import (
    plot_actual_vs_predicted, 
    plot_feature_importance, 
    plot_loss_curves, 
    plot_residuals, 
    plot_error_distribution, 
    plot_features_vs_error, 
    plot_spatial_error,
    plot_uncertainty_comparison,
    plot_posterior_distributions,
    plot_residual_distributions
)

def main():
    # Load configuration from inlist
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

    # Control parallelism to avoid exhausting system resources during CV or model training
    search_n_jobs = max(1, int(config['general'].get('search_n_jobs', '1')))
    booster_n_jobs = max(1, int(config['general'].get('booster_n_jobs', str(max(1, os.cpu_count() or 1)))))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    # First get the full data
    df = pd.read_csv(data_file)
    features = ['GAL_LONG', 'GAL_LAT', 'Ks_mag', 'I1_mag', 'I2_mag', 'I3_mag', 'I4_mag', 'alpha']
    target = 'Mips_24_mag'
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]
    
    # Split into train_val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Split train_val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=random_state)
    
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
        import xgboost as xgb
        from xgboost_model import train_xgboost, evaluate_model
        
        # Base model for search (without early stopping, as CV handles validation)
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=booster_n_jobs
        )
        
        # Parameter distribution for RandomizedSearchCV
        param_dist = {
            'learning_rate': [0.005, 0.01, 0.05],
            'n_estimators': [500, 1000, 2000],
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=50,  # Number of parameter settings sampled; adjust as needed
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=search_n_jobs,
            random_state=random_state
        )
        
        random_search.fit(X_train_val, y_train_val)
        
        best_params = random_search.best_params_
        print("Best parameters found: ", best_params)
        
        # Train final model with best params using original train/val for history and early stopping
        model, history = train_xgboost(
            X_train, y_train, X_val, y_val,
            learning_rate=best_params['learning_rate'],
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_child_weight=best_params['min_child_weight'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree']
        )
        
        predictions = evaluate_model(model, X_test, y_test)
    elif model_type == 'mlp':
        from mlp_model import train_mlp, evaluate_model
        model, scaler, history = train_mlp(
            X_train, y_train, X_val, y_val, learning_rate, n_estimators, max_depth
        )
        predictions = evaluate_model(model, scaler, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")
    
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

    print("... Doing posterior shit...")
    # Extract alphas and derive stages for quantification
    alphas = X_test['alpha'].values
    stages = []  # Derive stages as in generate_posterior
    for alpha in alphas:
        if alpha > 0.3: stages.append('Class0')
        elif alpha > -0.3: stages.append('ClassI')
        elif alpha > -1.6: stages.append('ClassII')
        else: stages.append('ClassIII')

    # Generate posteriors with decompositions
    from posterior import generate_posterior
    posteriors, model_samples_list, deltas_list = generate_posterior(
        model, model_type, X_test, predictions, alphas, history
    )

    # Quantify uncertainties
    from posterior import quantify_uncertainties
    uncertainty_df, aggregates = quantify_uncertainties(
        posteriors, model_samples_list, deltas_list, stages,
        os.path.join(output_dir, 'uncertainty_quantification.csv')
    )
    print("Uncertainty Quantification Aggregates:\n", aggregates)

    # Plot decomposed posteriors
    plot_posterior_distributions(posteriors, model_samples_list, deltas_list, y_test.values, 
                                 os.path.join(output_dir, 'posterior_dist.png'))

    plot_uncertainty_comparison(aggregates, os.path.join(output_dir, 'uncertainty_comparison.png'))

    plot_residual_distributions(y_test, predictions, model_samples_list, deltas_list, 
                                os.path.join(output_dir, 'residual_distributions.png'))

    # Optional: Save posteriors
    np.save(os.path.join(output_dir, 'posteriors.npy'), np.array(posteriors, dtype=object))

if __name__ == "__main__":
    main()