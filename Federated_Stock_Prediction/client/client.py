import requests
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

SERVER_URL = "http://127.0.0.1:5000"
GET_WEIGHTS_URL = f"{SERVER_URL}/get_global_weights"
UPLOAD_WEIGHTS_URL = f"{SERVER_URL}/upload_weights"

def fetch_global_model():
    response = requests.get(GET_WEIGHTS_URL)
    if response.status_code == 200:
        weights = pickle.loads(response.content)
        model = load_model("../models/best_model.h5")
        model.set_weights(weights)
        print("✅ Fetched global model weights from server!")
        return model
    else:
        raise Exception(f"Failed to fetch global weights: {response.text}")

def send_model_to_server(model, sample_size):
    weights = model.get_weights()
    data = {'weights': weights, 'sample_size': sample_size}
    serialized_data = pickle.dumps(data)
    files = {"weights": ('weights.pkl', serialized_data, 'application/octet-stream')}
    response = requests.post(UPLOAD_WEIGHTS_URL, files=files)
    
    if response.status_code == 200:
        print("✅ Model weights successfully sent to server!")
    else:
        print(f"❌ Failed to send model. Server response: {response.text}")

def train_local_model(client_id, epochs=10, batch_size=32):
    X_train = joblib.load(f"../processed_data/X_client_{client_id}.pkl")
    y_train = joblib.load(f"../processed_data/y_client_{client_id}.pkl")
    sample_size = len(y_train)
    
    model = fetch_global_model()
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='mean_squared_error', metrics=['mae'])  # Reduced to 0.00005
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_split=0.2, callbacks=[early_stopping], verbose=1)
    print(f"Client {client_id} fine-tuning complete.")
    
    return model, sample_size

if __name__ == "__main__":
    for round_num in range(5):
        print(f"Starting round {round_num + 1}")
        for client_id in range(4):
            model, sample_size = train_local_model(client_id=client_id, epochs=10)
            send_model_to_server(model, sample_size)