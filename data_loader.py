import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    'GAL_LAT', 'GAL_LONG',                  #these to is not so good 
    'GAL_LONG_sin', 'GAL_LONG_cos', 
    'Ks_mag', 'I1_mag', 'I2_mag', 'I3_mag', 'I4_mag', 'alpha',
    'Ks_I1', 'I1_I2', 'I2_I3', 'I3_I4', 'I4_mag_sq'
]

TARGET_COLUMN = 'Mips_24_mag'


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with all derived features required by the models."""

    engineered = df.copy()

    # Add derived features: color indices
    engineered['Ks_I1'] = engineered['Ks_mag'] - engineered['I1_mag']
    engineered['I1_I2'] = engineered['I1_mag'] - engineered['I2_mag']
    engineered['I2_I3'] = engineered['I2_mag'] - engineered['I3_mag']
    engineered['I3_I4'] = engineered['I3_mag'] - engineered['I4_mag']

    # Cyclic encoding for GAL_LONG (replacing original GAL_LONG)
    engineered['GAL_LONG_sin'] = np.sin(2 * np.pi * engineered['GAL_LONG'] / 360)
    engineered['GAL_LONG_cos'] = np.cos(2 * np.pi * engineered['GAL_LONG'] / 360)

    # Polynomial term for I4_mag (to help with non-linearity in faint sources)
    engineered['I4_mag_sq'] = engineered['I4_mag'] ** 2

    return engineered


def load_and_split_data(data_file, test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads the CSV data, preprocesses it, and splits into train/validation/test sets.

    Assumes the CSV has columns: GAL_LONG, GAL_LAT, Ks_mag, I1_mag, I2_mag, I3_mag, I4_mag, Mips_24_mag, alpha.
    Features: All except Mips_24_mag (target).
    Handles missing values by dropping rows (if any; adjust as needed for prediction on missing targets).
    """
    df = pd.read_csv(data_file)
    df = _add_engineered_features(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the input data.")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_features_for_inference(data_file):
    """Load data for inference and return the original dataframe along with model features."""

    df = pd.read_csv(data_file)
    df = _add_engineered_features(df)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns for inference: {missing_features}")

    feature_frame = df[FEATURE_COLUMNS]
    return df, feature_frame
