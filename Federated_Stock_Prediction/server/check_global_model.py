import pickle
import os

GLOBAL_MODEL_PATH = "models/global_model.pkl"

if not os.path.exists(GLOBAL_MODEL_PATH):
    print("❌ ERROR: Global model file not found!")
else:
    with open(GLOBAL_MODEL_PATH, "rb") as f:
        global_weights = pickle.load(f)
    
    print("✅ Global model loaded successfully!")
    print("🔎 First 5 Model Weights:", global_weights[:5])  # Print first 5 weights
