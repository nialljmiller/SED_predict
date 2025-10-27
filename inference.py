import argparse
import configparser
import os
from typing import Optional

from joblib import load

from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_features_for_inference


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

    if model_type == 'mlp':
        scaler = load(scaler_path)
        transformed_features = scaler.transform(feature_frame)
        predictions = model.predict(transformed_features)
        results_df = original_df.copy()
        results_df[f"{TARGET_COLUMN}_pred"] = predictions
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

    # Keep only original columns plus predictions to avoid duplicating engineered features unless desired
    base_columns = [col for col in original_df.columns if col not in FEATURE_COLUMNS or col == TARGET_COLUMN]
    export_columns = base_columns + [col for col in results_df.columns if col not in original_df.columns]
    export_df = results_df[export_columns]

    export_df.to_csv(output_file, index=False)

    print(f"Inference complete. Saved predictions for {len(export_df)} sources to {output_file}")


if __name__ == "__main__":
    main()
