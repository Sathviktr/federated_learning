import requests
import pickle
import json
import numpy as np
import tensorflow as tf
from train_local import train_model  # Will create this file next

SERVER_URL = "http://127.0.0.1:5000/upload_weights"  # Adjust if needed

def send_model_to_server(model):
    """Serialize and send model weights to the server."""
    weights = model.get_weights()  # Extract trained model weights
    serialized_weights = pickle.dumps(weights)  # Convert to byte format

    response = requests.post(SERVER_URL, files={"weights": serialized_weights})
    
    if response.status_code == 200:
        print("✅ Model weights successfully sent to server!")
    else:
        print(f"❌ Failed to send model. Server response: {response.text}")

if __name__ == "__main__":
    model = train_model()  # Train the local model
    send_model_to_server(model)  # Send model weights to the server
