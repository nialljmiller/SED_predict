import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_xgboost(X_train, y_train, X_val, y_val, learning_rate=0.1, n_estimators=100, max_depth=3):
    """
    Trains an XGBoost regressor model and returns the model along with RMSE history.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    evals_result = model.evals_result()
    history = {
        'train': evals_result['validation_0']['rmse'],
        'val': evals_result['validation_1']['rmse']
    }
    return model, history

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