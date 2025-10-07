import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_xgboost(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=3):
    """
    Trains an XGBoost regressor model.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and returns predictions.
    Prints RMSE and MAE for assessment.
    """
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return predictions