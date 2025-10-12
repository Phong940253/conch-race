# ===========================================
# 🐚 Conch Race Winner Prediction Script
# ===========================================

# --- 1. Setup ---
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import urllib.parse
from common import ConchPredictor, parse_rate_emoji, SHEET_ID, SHEET_NAME

# --- 2. Load Model and Data ---
MODEL_PATH = "conch_race_model.pt"

# Load the saved model checkpoint
# Set weights_only=False as the checkpoint contains non-tensor data like the label encoder.
checkpoint = torch.load(MODEL_PATH, weights_only=False)

# Extract saved components
input_dim = checkpoint["input_dim"]
num_classes = checkpoint["num_classes"]
label_encoder_classes = checkpoint["label_encoder"]
players = checkpoint["players"]

# Initialize model
model = ConchPredictor(input_dim, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set model to evaluation mode

# Restore LabelEncoder
label_enc = LabelEncoder()
label_enc.classes_ = label_encoder_classes

print(f"✅ Model '{MODEL_PATH}' loaded successfully.")

# --- 3. Fetch latest data ---
encoded_name = urllib.parse.quote(SHEET_NAME)
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={encoded_name}"

# Load into pandas
origin_df = pd.read_csv(csv_url)
print("✅ Loaded Google Sheet successfully! Shape:", origin_df.shape)

# Get the latest row
latest_race = origin_df.iloc[-1]

# --- 4. Prediction ---
def predict_winner(model, row, players, label_encoder):
    features = []
    for p in players:
        rate, emo = parse_rate_emoji(row[p])
        features += [rate, emo]
    
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        
    return label_encoder.inverse_transform([pred])[0]

# Predict on the latest race
predicted_winner = predict_winner(model, latest_race, players, label_enc)

print("\n--- Latest Race Data ---")
print(latest_race)
print("\n--- Prediction ---")
print(f"🔮 Predicted Winner: {predicted_winner}")
