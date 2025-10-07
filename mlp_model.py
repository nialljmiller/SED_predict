from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_mlp(X_train, y_train, X_val, y_val, learning_rate=0.1, n_estimators=100, max_depth=3):
    """
    Trains an MLP regressor model with feature scaling and returns RMSE history.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Configure MLP
    hidden_layer_sizes = (max_depth * 10,)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        random_state=42,
        activation='relu',
        solver='adam'
    )
    
    # Train iteratively with partial_fit and track RMSE
    train_rmse = []
    val_rmse = []
    for _ in range(n_estimators):
        model.partial_fit(X_train_scaled, y_train)
        pred_train = model.predict(X_train_scaled)
        rmse_train = mean_squared_error(y_train, pred_train)
        train_rmse.append(rmse_train)
        
        pred_val = model.predict(X_val_scaled)
        rmse_val = mean_squared_error(y_val, pred_val)
        val_rmse.append(rmse_val)
    
    history = {'train': train_rmse, 'val': val_rmse}
    return model, scaler, history

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