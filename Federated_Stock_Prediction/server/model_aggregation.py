import os
import pickle

AGGREGATED_MODEL_PATH = "models/global_model.pkl"
RECEIVED_MODELS_PATH = "received_models/"

def aggregate_models():
    model_files = [f for f in os.listdir(RECEIVED_MODELS_PATH) if f.endswith(".pkl")]

    if not model_files:
        print("❌ No models received for aggregation!")
        return
    
    print(f"🔄 Aggregating {len(model_files)} client models...")

    # Load the first model (since we have only one client for now)
    model_path = os.path.join(RECEIVED_MODELS_PATH, model_files[0])
    with open(model_path, "rb") as f:
        model_weights = pickle.load(f)
    
    # Save it as the global model
    os.makedirs("models", exist_ok=True)
    with open(AGGREGATED_MODEL_PATH, "wb") as f:
        pickle.dump(model_weights, f)

    print(f"✅ Global model saved at {AGGREGATED_MODEL_PATH}.")
import numpy as np
import joblib

def federated_averaging(client_weights):
    """
    Aggregates weights from multiple clients using Federated Averaging.
    
    Parameters:
        client_weights (list of list of np.array): Weights from multiple clients.

    Returns:
        aggregated_weights (list of np.array): Aggregated global model weights.
    """
    num_clients = len(client_weights)
    if num_clients == 0:
        raise ValueError("No client weights received for aggregation.")

    # Initialize aggregated weights with zeros
    aggregated_weights = [np.zeros_like(w) for w in client_weights[0]]

    # Sum up weights from all clients
    for weights in client_weights:
        for i, w in enumerate(weights):
            aggregated_weights[i] += w

    # Compute average
    aggregated_weights = [w / num_clients for w in aggregated_weights]

    return aggregated_weights

def save_global_model_weights(weights, path="models/global_model_weights.pkl"):
    """Save aggregated global model weights."""
    joblib.dump(weights, path)
    print(f"Global model weights saved at {path}")


if __name__ == "__main__":
    aggregate_models()
