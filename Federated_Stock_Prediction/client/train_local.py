import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "stock_data", "TSLA.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "close_scaler.pkl")

os.makedirs(MODELS_PATH, exist_ok=True)

def load_data(client_id, num_clients=4):
    df = pd.read_csv(DATA_PATH)
    data = df[['Close']]
    scaler = joblib.load(SCALER_PATH)
    data_scaled = scaler.transform(data)
    
    total_len = len(data_scaled)
    client_size = int(total_len / (num_clients * 0.8))
    overlap = int(client_size * 0.2)
    start = client_id * (client_size - overlap)
    if start < 0:
        start = 0
    end = min(start + client_size, total_len)
    client_data = data_scaled[start:end]
    
    X, y = [], []
    lookback = 60
    if len(client_data) <= lookback:
        raise ValueError(f"Client {client_id} data too small ({len(client_data)} points) for lookback {lookback}.")
    for i in range(lookback, len(client_data)):
        X.append(client_data[i-lookback:i, 0])
        y.append(client_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    return np.reshape(X, (X.shape[0], X.shape[1], 1)), y, scaler

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
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
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
    return model

def train_local_model(client_id, epochs=20, batch_size=32):  # Reduced to 20
    X_train, y_train, scaler = load_data(client_id)
    if X_train.shape[0] == 0:
        raise ValueError(f"Client {client_id} has insufficient data after sequencing.")
    
    model = create_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_split=0.2, callbacks=[early_stopping], verbose=1)
    print(f"Client {client_id} training complete.")
    
    return model.get_weights(), scaler

if __name__ == "__main__":
    weights, scaler = train_local_model(client_id=0, epochs=20)