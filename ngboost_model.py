import ngboost as ngb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def train_ngboost(X_train, y_train, X_val, y_val, learning_rate=0.1, n_estimators=100, max_depth=3):
    """
    Trains an NGBoost regressor model using a normal distribution for predictions.
    Computes RMSE history post-training using staged predictions.
    """
    base_learner = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model = ngb.NGBoost(
        Base=base_learner,
        Dist=ngb.distns.Normal,
        Score=ngb.scores.LogScore,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbose=False  # Set to False to avoid console clutter; change as needed
    )
    model.fit(X_train, y_train)
    
    # Compute staged RMSE for train and val
    train_rmse = []
    val_rmse = []
    for i in range(1, n_estimators + 1):
        pred_train = model.predict(X_train, max_iter=i)
        rmse_train = mean_squared_error(y_train, pred_train)
        train_rmse.append(rmse_train)
        
        pred_val = model.predict(X_val, max_iter=i)
        rmse_val = mean_squared_error(y_val, pred_val)
        val_rmse.append(rmse_val)
    
    history = {'train': train_rmse, 'val': val_rmse}
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the NGBoost model on the test set and returns point predictions (means).
    Prints RMSE and MAE for assessment based on point estimates.
    """
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return predictions