import socket
import pickle
import numpy as np

HOST = "localhost"
PORT = 5000

# Create dummy model weights (same structure as LSTM model)
dummy_weights = [np.random.rand(*shape) for shape in [(1, 200), (50, 200), (200,), (50, 200), (50, 200), (200,), (50, 25), (25,), (25, 1), (1,)]]

# Send weights to server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(pickle.dumps(dummy_weights))  # Serialize and send weights
    response = s.recv(1024)
    print("Server Response:", response.decode())
