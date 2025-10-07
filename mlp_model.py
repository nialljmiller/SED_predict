from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_mlp(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=3):
    """
    Trains an MLP regressor model with feature scaling.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Configure MLP with max_depth as proxy for hidden layer size
    hidden_layer_sizes = (max_depth * 10,)  # Simple heuristic: max_depth * 10 neurons
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        max_iter=n_estimators,
        random_state=42,
        activation='relu',
        solver='adam'
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluates the MLP model on the test set and returns predictions.
    Prints RMSE and MAE for assessment.
    """
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return predictions