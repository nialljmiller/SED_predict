import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(data_file, test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads the CSV data, preprocesses it, and splits into train/validation/test sets.
    
    Assumes the CSV has columns: GAL_LONG, GAL_LAT, Ks_mag, I1_mag, I2_mag, I3_mag, I4_mag, Mips_24_mag, alpha.
    Features: All except Mips_24_mag (target).
    Handles missing values by dropping rows (if any; adjust as needed for prediction on missing targets).
    """
    df = pd.read_csv(data_file)
    
    # Add derived features: color indices
    df['Ks_I1'] = df['Ks_mag'] - df['I1_mag']
    df['I1_I2'] = df['I1_mag'] - df['I2_mag']
    df['I2_I3'] = df['I2_mag'] - df['I3_mag']
    df['I3_I4'] = df['I3_mag'] - df['I4_mag']
    
    # Cyclic encoding for GAL_LONG (replacing original GAL_LONG)
    df['GAL_LONG_sin'] = np.sin(2 * np.pi * df['GAL_LONG'] / 360)
    df['GAL_LONG_cos'] = np.cos(2 * np.pi * df['GAL_LONG'] / 360)
    
    # Polynomial term for I4_mag (to help with non-linearity in faint sources)
    df['I4_mag_sq'] = df['I4_mag'] ** 2
    
    # Define updated features and target
    features = [
        'GAL_LONG_sin', 'GAL_LONG_cos', 'GAL_LAT', 
        'Ks_mag', 'I1_mag', 'I2_mag', 'I3_mag', 'I4_mag', 'alpha',
        'Ks_I1', 'I1_I2', 'I2_I3', 'I3_I4', 'I4_mag_sq'
    ]
    target = 'Mips_24_mag'
    
    # Drop rows with missing target
    df = df.dropna(subset=[target])
    
    X = df[features]
    y = df[target]
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test