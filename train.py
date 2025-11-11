import configparser
import os

from joblib import dump

from data_loader import load_and_split_data
from posterior import generate_posterior  # Import the new function
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid


DEFAULT_GENERAL = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 69,
    'output_dir': 'outputs/',
    'model_type': 'xgboost',
    'hyperparameter_tuning': False,
    'search_n_jobs': 1,
    'booster_n_jobs': max(1, os.cpu_count() or 1),
}

DEFAULT_PATHS = {
    'data_file': 'data/yso_training_data.csv',
}

MODEL_DEFAULTS = {
    'ngboost': {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 3,
    },
    'xgboost': {
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': max(1, os.cpu_count() or 1),
    },
    'mlp': {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 3,
    },
}


def smart_cast(item):
    lower_item = item.lower()
    if lower_item in {'true', 'false'}:
        return lower_item == 'true'
    try:
        return int(item)
    except ValueError:
        try:
            return float(item)
        except ValueError:
            return item


def parse_value_list(raw_value, default=None):
    if raw_value is None:
        return [default] if default is not None else []
    entries = [entry.strip() for entry in raw_value.split(',') if entry.strip()]
    if not entries:
        return [default] if default is not None else []
    return [smart_cast(entry) for entry in entries]


def build_param_grid(config, model_type):
    defaults = MODEL_DEFAULTS.get(model_type, {})
    param_values = {key: [value] for key, value in defaults.items()}

    for section in ('hyperparameters', model_type):
        if not config.has_section(section):
            continue
        for key, raw_value in config[section].items():
            parsed_values = parse_value_list(raw_value, defaults.get(key))
            if parsed_values:
                param_values[key] = parsed_values

    return param_values

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
    data_file = DEFAULT_PATHS['data_file']
    if config.has_section('paths'):
        data_file = config['paths'].get('data_file', data_file)

    general_section = config['general'] if config.has_section('general') else {}
    test_size = float(general_section.get('test_size', DEFAULT_GENERAL['test_size']))
    val_size = float(general_section.get('val_size', DEFAULT_GENERAL['val_size']))
    random_state = int(general_section.get('random_state', DEFAULT_GENERAL['random_state']))
    output_dir = general_section.get('output_dir', DEFAULT_GENERAL['output_dir']).strip() or DEFAULT_GENERAL['output_dir']
    model_type = general_section.get('model_type', DEFAULT_GENERAL['model_type']).strip().lower()
    hyperparameter_toggle_raw = general_section.get('hyperparameter_tuning', str(DEFAULT_GENERAL['hyperparameter_tuning']))
    hyperparameter_tuning = str(hyperparameter_toggle_raw).lower() in {'true', '1', 'yes', 'on'}

    booster_n_jobs_raw = str(general_section.get('booster_n_jobs', DEFAULT_GENERAL['booster_n_jobs'])).strip()
    booster_n_jobs = max(1, int(float(booster_n_jobs_raw)))

    feature_columns = None
    target_column = None
    if config.has_section('columns'):
        raw_features = config['columns'].get('feature_columns', fallback='')
        feature_columns = [col.strip() for col in raw_features.split(',') if col.strip()] or None
        target_column = config['columns'].get('target_column', fallback=None)
        if target_column:
            target_column = target_column.strip() or None

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
        ngb_param_values = build_param_grid(config, 'ngboost')
        ngb_keys = ['learning_rate', 'n_estimators', 'max_depth']
        ngb_param_grid = {key: ngb_param_values.get(key, [MODEL_DEFAULTS['ngboost'][key]]) for key in ngb_keys}

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

        xgb_param_values = build_param_grid(config, 'xgboost')
        explicit_n_jobs = (
            (config.has_section('xgboost') and config.has_option('xgboost', 'n_jobs')) or
            (config.has_section('hyperparameters') and config.has_option('hyperparameters', 'n_jobs'))
        )
        if explicit_n_jobs:
            xgb_param_values['n_jobs'] = [max(1, int(value)) for value in xgb_param_values.get('n_jobs', [booster_n_jobs])]
        else:
            xgb_param_values['n_jobs'] = [booster_n_jobs]

        xgb_keys = [
            'learning_rate',
            'n_estimators',
            'max_depth',
            'min_child_weight',
            'subsample',
            'colsample_bytree',
            'reg_alpha',
            'reg_lambda',
            'n_jobs',
        ]
        xgb_param_grid = {key: xgb_param_values.get(key, [MODEL_DEFAULTS['xgboost'][key]]) for key in xgb_keys}

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
        mlp_param_values = build_param_grid(config, 'mlp')
        mlp_keys = ['learning_rate', 'n_estimators', 'max_depth']
        mlp_param_grid = {key: mlp_param_values.get(key, [MODEL_DEFAULTS['mlp'][key]]) for key in mlp_keys}

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
