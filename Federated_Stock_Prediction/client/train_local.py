import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib  # For saving the scaler

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Get root project directory
DATA_PATH = os.path.join(BASE_DIR, "data", "stock_data", "TSLA.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")

# Ensure the 'models/' directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

# Path to save model weights
MODEL_WEIGHTS_PATH = os.path.join(MODELS_PATH, "client_model_weights.pkl")


def load_data():
    """Load stock data and preprocess it."""
    df = pd.read_csv(DATA_PATH)

    # Assuming 'Close' price is used for prediction
    data = df[['Close']].values  

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Save scaler for later use
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

    # Create sequences
    X, y = [], []
    lookback = 60  # Use 60 days of past data for prediction
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    
    return X, y


def create_model(input_shape):
    """Create and return an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_local_model(epochs=10, batch_size=16):
    """Train the LSTM model locally and save weights."""
    X_train, y_train = load_data()

    model = create_model((X_train.shape[1], 1))

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save model weights
    joblib.dump(model.get_weights(), MODEL_WEIGHTS_PATH)
    print(f"✅ Model weights saved at: {MODEL_WEIGHTS_PATH}")

    return model.get_weights()  # Return weights for federated training


if __name__ == "__main__":
    train_local_model(epochs=1, batch_size=16)  # Run training for testing
