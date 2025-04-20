import os
from flask import Flask, request, Response
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
import joblib

app = Flask(__name__)
HOST = "127.0.0.1"
PORT = 5000
MODELS_PATH = "../models/"
NUM_CLIENTS_EXPECTED = 4
NUM_ROUNDS = 5
round_count = 0

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

client_weights_list = []
client_sample_sizes = []
global_model = None

def create_global_model():
    model = load_model(os.path.join(MODELS_PATH, "best_model.h5"))
    return model

def weighted_federated_averaging(client_weights_list, client_sample_sizes, centralized_weights):
    num_clients = len(client_weights_list)
    if num_clients == 0:
        raise ValueError("No client weights received for aggregation.")
    
    total_samples = sum(client_sample_sizes)
    aggregated_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    
    for weights, sample_size in zip(client_weights_list, client_sample_sizes):
        weight_factor = sample_size / total_samples
        for i, w in enumerate(weights):
            aggregated_weights[i] += w * weight_factor
    
    # Blend with centralized weights (50% client, 50% centralized)
    for i in range(len(aggregated_weights)):
        aggregated_weights[i] = 0.5 * aggregated_weights[i] + 0.5 * centralized_weights[i]
    
    return aggregated_weights

def save_global_model_weights(weights, path="models/global_model_weights.pkl"):
    joblib.dump(weights, path)
    print(f"✅ Global model weights saved at {path}")

@app.route('/get_global_weights', methods=['GET'])
def get_global_weights():
    global global_model
    if global_model is None:
        global_model = create_global_model()
    weights = global_model.get_weights()
    return Response(pickle.dumps(weights), mimetype='application/octet-stream')

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    global client_weights_list, client_sample_sizes, global_model, round_count
    data = pickle.loads(request.files['weights'].read())
    weights = data['weights']
    sample_size = data['sample_size']
    client_weights_list.append(weights)
    client_sample_sizes.append(sample_size)
    print(f"Received weights from client. Total: {len(client_weights_list)}/{NUM_CLIENTS_EXPECTED}")

    if len(client_weights_list) >= NUM_CLIENTS_EXPECTED:
        print(f"Aggregating client weights for round {round_count + 1}...")
        centralized_weights = create_global_model().get_weights()
        aggregated_weights = weighted_federated_averaging(client_weights_list, client_sample_sizes, centralized_weights)
        global_model.set_weights(aggregated_weights)
        save_global_model_weights(aggregated_weights, path=os.path.join(MODELS_PATH, f"global_model_weights_round_{round_count + 1}.pkl"))
        client_weights_list.clear()
        client_sample_sizes.clear()
        round_count += 1
        print(f"Aggregation complete for round {round_count}/{NUM_ROUNDS}.")
        if round_count >= NUM_ROUNDS:
            print("All rounds complete!")
    
    return "ACK", 200

if __name__ == "__main__":
    global_model = create_global_model()
    print(f"Server running on {HOST}:{PORT} for {NUM_ROUNDS} rounds with centralized weights")
    app.run(host=HOST, port=PORT)