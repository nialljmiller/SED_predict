# data_loader.py
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


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
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


def _drop_bad_feature_columns(
    df: pd.DataFrame,
    candidate_features: List[str],
    bad_sentinels: Optional[List[float]]
) -> List[str]:
    kept = []
    for col in candidate_features:
        if col not in df.columns:
            continue
        if _series_bad_mask(df[col], bad_sentinels).any():
            continue  # drop the entire feature column if any bad exists
        df[col] = pd.to_numeric(df[col], errors='coerce')  # force numeric dtype
        kept.append(col)
    return kept


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
      - Drop feature columns if any element in them is bad (NaN/±inf/sentinel). Never drop target for this reason.
      - Train: drop rows with any bad in features OR bad target.
      - Inference: ignore target quality entirely; keep rows with clean features (target is still numeric NaN where junk).
    """
    df = pd.read_csv(
        data_file,
        na_values=["", "NA", "NaN", "nan", "Inf", "-Inf", "NULL", "null", "None"]
    )
    df = _add_engineered_features(df)

    if mode == 'train':
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found; required for training.")
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    elif mode == 'inference':
        if TARGET_COLUMN not in df.columns:
            df[TARGET_COLUMN] = np.nan  # Add dummy target column for consistency
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')


    # Coerce target to numeric in BOTH modes so dtype is never object
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    elif mode == 'train':
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found; required for training.")

    # Decide usable feature columns
    candidate_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    kept_features = _drop_bad_feature_columns(df, candidate_features, bad_sentinels)
    if not kept_features:
        raise ValueError("No usable feature columns remain after dropping bad columns.")

    X = df[kept_features].copy().replace([np.inf, -np.inf], np.nan)

    if mode == 'inference':
        # Keep rows with fully valid features; ignore target quality
        good_rows = ~X.isna().any(axis=1)
        return df.loc[good_rows].copy(), X.loc[good_rows]

    # mode == 'train'
    y = df[TARGET_COLUMN]
    y_bad = y.isna() | ~np.isfinite(y)
    row_bad_features = X.isna().any(axis=1)
    keep_rows = ~(row_bad_features | y_bad)
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
