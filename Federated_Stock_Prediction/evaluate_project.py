import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

X_test_seq = joblib.load("processed_data/X_test_seq.pkl")
y_test_seq = joblib.load("processed_data/y_test_seq.pkl")
close_scaler = joblib.load("processed_data/close_scaler.pkl")

weights = joblib.load("models/global_model_weights_round_5.pkl")  # Updated to round 5
model = Sequential([
    Input(shape=(60, 1)),
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32, return_sequences=False, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.set_weights(weights)

predictions = model.predict(X_test_seq)
predicted_prices = close_scaler.inverse_transform(predictions)
actual_prices = close_scaler.inverse_transform(y_test_seq.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
print(f"Federated Model RMSE: ${rmse:.2f}")
print(f"Federated Model MAPE: {mape:.2f}%")