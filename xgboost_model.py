import inspect

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_xgboost(X_train, y_train, X_val, y_val, learning_rate=0.01, n_estimators=2000, max_depth=6,
                  min_child_weight=1, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=1.0):
    """
    Trains an XGBoost regressor model and returns the model along with RMSE history.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        eval_metric='rmse'
    )
    eval_set = [(X_train, y_train), (X_val, y_val)]
    fit_signature = inspect.signature(model.fit)
    supports_callbacks = 'callbacks' in fit_signature.parameters and hasattr(xgb, 'callback')
    supports_early_stopping = 'early_stopping_rounds' in fit_signature.parameters

    fit_kwargs = {
        'eval_set': eval_set,
        'verbose': False,
    }

    if supports_callbacks and hasattr(xgb.callback, 'EarlyStopping'):
        fit_kwargs['callbacks'] = [
            xgb.callback.EarlyStopping(
                rounds=50,
                metric_name='rmse',
                data_name='validation_1',
                save_best=True
            )
        ]
    elif supports_early_stopping:
        fit_kwargs['early_stopping_rounds'] = 50
    else:
        print("Warning: Installed XGBoost does not support early stopping; proceeding without it.")

    model.fit(
        X_train,
        y_train,
        **fit_kwargs
    )
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
