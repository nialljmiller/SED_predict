import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_file, test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads the CSV data, preprocesses it, and splits into train/validation/test sets.
    
    Assumes the CSV has columns: GAL_LONG, GAL_LAT, Ks_mag, I1_mag, I2_mag, I3_mag, I4_mag, Mips_24_mag, alpha.
    Features: All except Mips_24_mag (target).
    Handles missing values by dropping rows (if any; adjust as needed for prediction on missing targets).
    """
    df = pd.read_csv(data_file)
    
    # Define features and target
    features = ['GAL_LONG', 'GAL_LAT', 'Ks_mag', 'I1_mag', 'I2_mag', 'I3_mag', 'I4_mag', 'alpha']
    target = 'Mips_24_mag'
    
    # Drop rows with missing target (for training; for prediction, filter separately if needed)
    df = df.dropna(subset=[target])
    
    X = df[features]
    y = df[target]
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test