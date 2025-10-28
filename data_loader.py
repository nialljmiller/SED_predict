import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Optional

FEATURE_COLUMNS = [
    'GAL_LAT', 'GAL_LONG',
    'GAL_LONG_sin', 'GAL_LONG_cos',
    'Ks_mag', 'I1_mag', 'I2_mag', 'I3_mag', 'I4_mag', 'alpha',
    'Ks_I1', 'I1_I2', 'I2_I3', 'I3_I4', 'I4_mag_sq'
]

TARGET_COLUMN = 'Mips_24_mag'

BASE_FEATURES = ['GAL_LAT', 'GAL_LONG', 'Ks_mag', 'I1_mag', 'I2_mag', 'I3_mag', 'I4_mag', 'alpha']

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    # Ensure base features exist, filling with NaN if missing
    for col in BASE_FEATURES:
        if col not in engineered.columns:
            engineered[col] = np.nan
        engineered[col] = pd.to_numeric(engineered[col], errors='coerce')
    # Compute derived features safely
    engineered['Ks_I1'] = engineered['Ks_mag'] - engineered['I1_mag']
    engineered['I1_I2'] = engineered['I1_mag'] - engineered['I2_mag']
    engineered['I2_I3'] = engineered['I2_mag'] - engineered['I3_mag']
    engineered['I3_I4'] = engineered['I3_mag'] - engineered['I4_mag']
    engineered['GAL_LONG_sin'] = np.sin(2 * np.pi * engineered['GAL_LONG'] / 360.0)
    engineered['GAL_LONG_cos'] = np.cos(2 * np.pi * engineered['GAL_LONG'] / 360.0)
    engineered['I4_mag_sq'] = engineered['I4_mag'] ** 2
    return engineered

def _series_bad_mask(s: pd.Series, bad_sentinels: Optional[List[float]]) -> pd.Series:
    s_num = pd.to_numeric(s, errors='coerce')
    mask = s_num.isna() | ~np.isfinite(s_num)
    if bad_sentinels:
        mask = mask | s_num.isin(bad_sentinels)
    return mask

def _prepare_data(
    data_file: str,
    mode: str,  # 'train' or 'inference'
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    bad_sentinels: Optional[List[float]] = None
):
    """
    Universal path:
      - Keep all feature columns, replace bad values with NaN.
      - Train: drop rows with any bad in features OR bad target.
      - Inference: drop rows only if all features are bad; ignore target quality.
    """
    df = pd.read_csv(
        data_file,
        na_values=["", "NA", "NaN", "nan", "Inf", "-Inf", "NULL", "null", "None"]
    )
    df = _add_engineered_features(df)
    # Coerce target to numeric in BOTH modes so dtype is never object
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    elif mode == 'train':
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found; required for training.")
    else:  # inference
        df[TARGET_COLUMN] = np.nan  # Add dummy if missing

    # Replace bad sentinels in target if provided
    if bad_sentinels:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(bad_sentinels, np.nan)

    # Keep all features, replace inf with NaN
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            X[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if bad_sentinels:
                X[col] = X[col].replace(bad_sentinels, np.nan)
        else:
            X[col] = np.nan

    if mode == 'inference':
        # Keep rows unless all features are bad; ignore target
        row_all_bad = X.isna().all(axis=1)
        keep_rows = ~row_all_bad
        return df.loc[keep_rows].copy(), X.loc[keep_rows]

    # mode == 'train'
    y = df[TARGET_COLUMN]
    y_bad = y.isna() | ~np.isfinite(y)
    row_any_bad = X.isna().any(axis=1)
    keep_rows = ~(row_any_bad | y_bad)
    X = X.loc[keep_rows]
    y = y.loc[keep_rows]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Public API
def load_and_split_data(
    data_file: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    bad_sentinels: Optional[List[float]] = None
):
    return _prepare_data(
        data_file=data_file,
        mode='train',
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        bad_sentinels=bad_sentinels
    )

def load_features_for_inference(
    data_file: str,
    bad_sentinels: Optional[List[float]] = None
):
    return _prepare_data(
        data_file=data_file,
        mode='inference',
        bad_sentinels=bad_sentinels
    )