import numpy as np
import joblib

def federated_averaging(client_weights):
    num_clients = len(client_weights)
    if num_clients == 0:
        raise ValueError("No client weights received for aggregation.")
    aggregated_weights = [np.zeros_like(w) for w in client_weights[0]]
    for weights in client_weights:
        for i, w in enumerate(weights):
            aggregated_weights[i] += w
    aggregated_weights = [w / num_clients for w in aggregated_weights]
    return aggregated_weights

def save_global_model_weights(weights, path="models/global_model_weights.pkl"):
    joblib.dump(weights, path)
    print(f"✅ Global model weights saved at {path}")

if __name__ == "__main__":
    pass