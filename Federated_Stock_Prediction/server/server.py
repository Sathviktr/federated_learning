import os
import socket
import joblib
import pickle
import threading
from model_aggregation import federated_averaging, save_global_model_weights

# Server Configuration
HOST = "localhost"  # Change this if running on different machines
PORT = 5000
MODELS_PATH = "models/"

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

client_weights_list = []  # Store weights from clients

def handle_client(conn, addr):
    """Handles incoming client connections and receives model weights."""
    global client_weights_list
    print(f"Client {addr} connected.")

    try:
        data = conn.recv(4096)  # Receive data from client
        client_weights = pickle.loads(data)  # Deserialize received weights
        client_weights_list.append(client_weights)
        print(f"Received weights from client {addr}")
        
        conn.sendall(b"ACK")  # Send acknowledgment

    except Exception as e:
        print(f"Error receiving data from {addr}: {e}")
    finally:
        conn.close()

def start_server():
    """Starts the federated learning server."""
    global client_weights_list

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)  # Allow up to 5 clients to connect
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=handle_client, args=(conn, addr)).start()

            # If we receive weights from at least 2 clients, aggregate them
            if len(client_weights_list) >= 2:
                print("Aggregating client weights...")
                aggregated_weights = federated_averaging(client_weights_list)
                save_global_model_weights(aggregated_weights)

                client_weights_list = []  # Reset for next round
                print("Aggregation complete.")

if __name__ == "__main__":
    start_server()
