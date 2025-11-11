import configparser
import os

from joblib import dump

from data_loader import load_and_split_data
from posterior import generate_posterior  # Import the new function
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

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
    test_size = float(config['general']['test_size'])
    val_size = float(config['general']['val_size'])
    random_state = int(config['general']['random_state'])
    output_dir = config['general']['output_dir']
    model_type = config['general'].get('model_type', 'xgboost')  # Default to xgboost
    hyperparameter_tuning = config['general'].getboolean('hyperparameter_tuning', fallback=False)

    hyperparam_values = {}
    if config.has_section('hyperparameters'):
        for key, value in config['hyperparameters'].items():
            entries = [item.strip() for item in value.split(',') if item.strip()]
            if not entries:
                continue

            def smart_cast(item):
                lower_item = item.lower()
                if lower_item in {'true', 'false'}:
                    return lower_item == 'true'
                try:
                    int_val = int(item)
                    return int_val
                except ValueError:
                    try:
                        float_val = float(item)
                        return float_val
                    except ValueError:
                        return item

            hyperparam_values[key] = [smart_cast(entry) for entry in entries]

    feature_columns = None
    target_column = None
    if config.has_section('columns'):
        raw_features = config['columns'].get('feature_columns', fallback='')
        feature_columns = [col.strip() for col in raw_features.split(',') if col.strip()] or None
        target_column = config['columns'].get('target_column', fallback=None)
        if target_column:
            target_column = target_column.strip() or None

    # Control parallelism to avoid exhausting system resources during CV or model training
    booster_n_jobs = max(1, int(config['general'].get('booster_n_jobs', str(max(1, os.cpu_count() or 1)))))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data using data_loader for consistency
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        data_file,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    # Train and evaluate based on model_type
    scaler = None
    history = None
    if model_type == 'ngboost':
        from ngboost_model import train_ngboost, evaluate_model
        ngb_defaults = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 3,
        }
        ngb_param_grid = {
            key: hyperparam_values.get(key, [default])
            for key, default in ngb_defaults.items()
        }

        if hyperparameter_tuning:
            best_score = float('inf')
            best_result = None
            for params in ParameterGrid(ngb_param_grid):
                candidate_model, candidate_history = train_ngboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                )
                val_predictions = candidate_model.predict(X_val)
                val_rmse = mean_squared_error(y_val, val_predictions)
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_result = (params, candidate_model, candidate_history)

            assert best_result is not None, "Hyperparameter tuning failed to produce a model."
            best_params, model, history = best_result
            print(f"Selected NGBoost hyperparameters: {best_params}")
        else:
            selected_params = {key: values[0] for key, values in ngb_param_grid.items()}
            model, history = train_ngboost(
                X_train,
                y_train,
                X_val,
                y_val,
                learning_rate=selected_params['learning_rate'],
                n_estimators=selected_params['n_estimators'],
                max_depth=selected_params['max_depth'],
            )
            best_params = selected_params

        predictions = evaluate_model(model, X_test, y_test)
    elif model_type == 'xgboost':
        import xgboost as xgb
        from xgboost_model import train_xgboost, evaluate_model

        xgb_defaults = {
            'learning_rate': 0.01,
            'n_estimators': 2000,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
        xgb_param_grid = {
            key: hyperparam_values.get(key, [default])
            for key, default in xgb_defaults.items()
        }
        xgb_param_grid['n_jobs'] = [booster_n_jobs]

        if hyperparameter_tuning:
            best_score = float('inf')
            best_result = None
            for params in ParameterGrid(xgb_param_grid):
                candidate_model, candidate_history = train_xgboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_child_weight=params['min_child_weight'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    n_jobs=params['n_jobs'],
                )
                val_predictions = candidate_model.predict(X_val)
                val_rmse = mean_squared_error(y_val, val_predictions)
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_result = (params, candidate_model, candidate_history)

            assert best_result is not None, "Hyperparameter tuning failed to produce a model."
            best_params, model, history = best_result
            print(f"Selected XGBoost hyperparameters: {best_params}")
        else:
            selected_params = {key: values[0] for key, values in xgb_param_grid.items()}
            model, history = train_xgboost(
                X_train,
                y_train,
                X_val,
                y_val,
                learning_rate=selected_params['learning_rate'],
                n_estimators=selected_params['n_estimators'],
                max_depth=selected_params['max_depth'],
                min_child_weight=selected_params['min_child_weight'],
                subsample=selected_params['subsample'],
                colsample_bytree=selected_params['colsample_bytree'],
                reg_alpha=selected_params['reg_alpha'],
                reg_lambda=selected_params['reg_lambda'],
                n_jobs=selected_params['n_jobs'],
            )
            best_params = selected_params

        predictions = evaluate_model(model, X_test, y_test)
    elif model_type == 'mlp':
        from mlp_model import train_mlp, evaluate_model
        mlp_defaults = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 3,
        }
        mlp_param_grid = {
            key: hyperparam_values.get(key, [default])
            for key, default in mlp_defaults.items()
        }

        if hyperparameter_tuning:
            best_score = float('inf')
            best_result = None
            for params in ParameterGrid(mlp_param_grid):
                candidate_model, candidate_scaler, candidate_history = train_mlp(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                )
                val_predictions = candidate_model.predict(candidate_scaler.transform(X_val))
                val_rmse = mean_squared_error(y_val, val_predictions)
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_result = (params, candidate_model, candidate_scaler, candidate_history)

            assert best_result is not None, "Hyperparameter tuning failed to produce a model."
            best_params, model, scaler, history = best_result
            print(f"Selected MLP hyperparameters: {best_params}")
        else:
            selected_params = {key: values[0] for key, values in mlp_param_grid.items()}
            model, scaler, history = train_mlp(
                X_train,
                y_train,
                X_val,
                y_val,
                learning_rate=selected_params['learning_rate'],
                n_estimators=selected_params['n_estimators'],
                max_depth=selected_params['max_depth'],
            )
            best_params = selected_params

        predictions = evaluate_model(model, scaler, X_test, y_test)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")
    
    # Persist trained artefacts for later inference
    model_path = os.path.join(output_dir, f"{model_type}_model.joblib")
    dump(model, model_path)
    print(f"Saved trained {model_type} model to {model_path}")

    scaler_path = None
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f"{model_type}_scaler.joblib")
        dump(scaler, scaler_path)
        print(f"Saved feature scaler to {scaler_path}")

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
