import ngboost as ngb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def train_ngboost(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=3):
    """
    Trains an NGBoost regressor model using a normal distribution for predictions.
    """
    base_learner = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model = ngb.NGBoost(
        Base=base_learner,
        Dist=ngb.distns.Normal,
        Score=ngb.scores.LogScore,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbose=True
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the NGBoost model on the test set and returns point predictions (means).
    Prints RMSE and MAE for assessment based on point estimates.
    """
    pred_dist = model.pred_dist(X_test)
    predictions = pred_dist.loc  # Use mean (loc) for point predictions
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return predictions