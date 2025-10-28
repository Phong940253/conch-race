# ===========================================
# 🐚 Conch Race Winner Prediction Script
# ===========================================

# --- 1. Setup ---
import pandas as pd
import torch
import torch.nn.functional as F
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
        probabilities = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        
    winner = label_encoder.inverse_transform([pred])[0]
    return winner, probabilities

# Predict on the latest race
predicted_winner, predictions = predict_winner(model, latest_race, players, label_enc)

print("\n--- Latest Race Data ---")
print(latest_race)
print("\n--- Prediction ---")
print(f"🔮 Predicted Winner: {predicted_winner}")

print("\n--- Prediction Rates ---")
for i, conch in enumerate(label_enc.classes_):
    rate = predictions[0][i].item() * 100
    print(f"{conch}: {rate:.2f}%")

# predict all races and show to table
all_predictions = []
for idx, row in origin_df.iterrows():
    winner, _ = predict_winner(model, row, players, label_enc)
    all_predictions.append(winner)
new_df = origin_df.copy()
new_df["Predicted Winner"] = all_predictions
print("\n--- All Races with Predictions ---")
# remove column has name Predict
new_df = new_df.loc[:, ~new_df.columns.str.contains('^Predict$', case=False)]
print(new_df.head(30))

# save to csv
new_df.to_csv("conch_race_predictions.csv", index=False)