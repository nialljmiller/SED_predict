import argparse
import configparser
import os
from typing import Optional

from joblib import load

from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_features_for_inference
import pandas as pd
import numpy as np

# Add imports for plotting
from plots import (
    plot_actual_vs_predicted, plot_residuals, plot_error_distribution,
    plot_spatial_error, plot_uncertainty_comparison, plot_posterior_distributions,
    plot_residual_distributions, plot_color_color_with_target,
    plot_galactic_position_with_band
)
from posterior import generate_posterior, quantify_uncertainties


def _read_config(path: str) -> configparser.ConfigParser:
    """Safely read a configuration file, returning an empty parser if missing."""
    parser = configparser.ConfigParser()
    if os.path.exists(path):
        parser.read(path)
    return parser


def _config_get(parser: configparser.ConfigParser, section: str, option: str, fallback=None):
    if parser.has_option(section, option):
        return parser.get(section, option)
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a previously trained model.")
    parser.add_argument(
        "--config",
        default="inlist",
        help="Path to the configuration file used during training (default: inlist)."
    )
    parser.add_argument(
        "--data-file",
        dest="data_file",
        help="CSV file containing sources to predict. If omitted, falls back to the training data file."
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="Path to the trained model weights. Defaults to <output_dir>/<model_type>_model.joblib."
    )
    parser.add_argument(
        "--scaler-path",
        dest="scaler_path",
        help="Optional path to a fitted scaler (required for MLP models if not in the default location)."
    )
    parser.add_argument(
        "--model-type",
        dest="model_type",
        choices=["xgboost", "ngboost", "mlp"],
        help="Override the model type stored in the config file."
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        help="Where to save the predictions CSV. Defaults to <output_dir>/inference_results.csv."
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Directory containing training artefacts. Defaults to the training output_dir."
    )

    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace, config: configparser.ConfigParser):
    model_type = args.model_type or _config_get(config, 'general', 'model_type', fallback='xgboost')
    output_dir = args.output_dir or _config_get(config, 'general', 'output_dir', fallback='outputs')
    os.makedirs(output_dir, exist_ok=True)

    model_path = args.model_path or os.path.join(output_dir, f"{model_type}_model.joblib")
    scaler_path: Optional[str] = None
    if model_type == 'mlp':
        scaler_path = args.scaler_path or os.path.join(output_dir, f"{model_type}_scaler.joblib")

    output_file = args.output_file or os.path.join(output_dir, 'inference_results.csv')

    return model_type, output_dir, model_path, scaler_path, output_file


def main():
    args = parse_args()
    config = _read_config(args.config)

    data_file = args.data_file or _config_get(config, 'paths', 'data_file')
    if data_file is None:
        raise ValueError("No data file provided. Specify --data-file or define paths.data_file in the config.")

    model_type, output_dir, model_path, scaler_path, output_file = _resolve_paths(args, config)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training or provide --model-path.")

    if model_type == 'mlp':
        if scaler_path is None or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "MLP models require a fitted scaler. Provide --scaler-path or ensure the default scaler exists."
            )

    original_df, feature_frame = load_features_for_inference(data_file)

    model = load(model_path)
    scaler = None
    if model_type == 'mlp':
        scaler = load(scaler_path)
        transformed_features = scaler.transform(feature_frame)
        predictions = model.predict(transformed_features)
        results_df = original_df.copy()
        results_df[f"{TARGET_COLUMN}_pred"] = predictions
        pred_dist = None  # No distribution for MLP
    elif model_type == 'ngboost':
        pred_dist = model.pred_dist(feature_frame)
        predictions = pred_dist.loc
        std_devs = pred_dist.scale
        results_df = original_df.copy()
        results_df[f"{TARGET_COLUMN}_pred"] = predictions
        results_df[f"{TARGET_COLUMN}_pred_std"] = std_devs
    else:  # xgboost
        predictions = model.predict(feature_frame)
        results_df = original_df.copy()
        results_df[f"{TARGET_COLUMN}_pred"] = predictions
        pred_dist = None  # No distribution for XGBoost

    results_df.to_csv(output_file, index=False)

    print(f"Inference complete. Saved predictions for {len(results_df)} sources to {output_file}")

    print("Generating inference plots...")
    y_true = original_df[TARGET_COLUMN] if TARGET_COLUMN in original_df.columns else None
    has_ground_truth = y_true is not None

    # Common plots (adapt spatial_error to use predictions if no y_true)
    if has_ground_truth:
        plot_actual_vs_predicted(y_true, predictions, os.path.join(output_dir, 'inf_actual_vs_predicted.png'))
        plot_residuals(y_true, predictions, os.path.join(output_dir, 'inf_residuals.png'))
        plot_error_distribution(y_true, predictions, os.path.join(output_dir, 'inf_error_distribution.png'))
        plot_spatial_error(feature_frame, y_true, predictions, os.path.join(output_dir, 'inf_spatial_error.png'))
    else:
        # Fallback: Plot spatial distribution of predictions (modify plot_spatial_error accordingly if needed)
        plot_spatial_error(feature_frame, predictions, predictions, os.path.join(output_dir, 'inf_spatial_predictions.png'))  # Use predictions as proxy for errors

    predicted_column = f"{TARGET_COLUMN}_pred"
    try:
        plot_color_color_with_target(
            results_df,
            predicted_column,
            os.path.join(output_dir, 'inf_color_color.png')
        )
    except ValueError as exc:
        print(f"Skipping color-color plot: {exc}")

    try:
        plot_galactic_position_with_band(
            results_df,
            predicted_column,
            os.path.join(output_dir, 'inf_galactic_position.png')
        )
    except ValueError as exc:
        print(f"Skipping galactic position plot: {exc}")

    # Uncertainty plots for NGBoost or if history is available (assume history can be loaded or approximated)
    if model_type == 'ngboost' and pred_dist is not None:
        # Approximate history if needed; here we use a placeholder (load from training if available)
        history = {'val': [np.mean(std_devs)]}  # Placeholder; enhance as needed
        alphas = feature_frame['alpha'].values if 'alpha' in feature_frame.columns else np.zeros(len(predictions))
        posteriors, model_samples_list, deltas_list = generate_posterior(
            model, model_type, feature_frame, predictions, alphas, history
        )
        stages = []  # Derive as in run.py
        for alpha in alphas:
            if alpha > 0.3: stages.append('Class0')
            elif alpha > -0.3: stages.append('ClassI')
            elif alpha > -1.6: stages.append('ClassII')
            else: stages.append('ClassIII')
        uncertainty_df, aggregates = quantify_uncertainties(
            posteriors, model_samples_list, deltas_list, stages,
            os.path.join(output_dir, 'inf_uncertainty_quantification.csv')
        )
        plot_uncertainty_comparison(aggregates, os.path.join(output_dir, 'inf_uncertainty_comparison.png'))
        plot_posterior_distributions(posteriors, model_samples_list, deltas_list, y_true.values if has_ground_truth else None,
                                     os.path.join(output_dir, 'inf_posterior_dist.png'))
        if has_ground_truth:
            plot_residual_distributions(y_true, predictions, model_samples_list, deltas_list,
                                        os.path.join(output_dir, 'inf_residual_distributions.png'))

    print(f"Plots saved in {output_dir} with prefix 'inf_' for inference-specific visualizations.")


if __name__ == "__main__":
    main()