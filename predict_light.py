# ===========================================
# 🐚 Conch Race Winner Prediction Script (LightGBM)
# ===========================================

# --- 1. Setup ---
import pandas as pd
import numpy as np
import joblib
import urllib.parse
from common import parse_rate_emoji, SHEET_ID, SHEET_NAME

# --- 2. Load Model and Data ---
MODEL_PATH = "conch_race_lightgbm_model.pkl"

# Load the saved model and other objects
data = joblib.load(MODEL_PATH)
model = data["model"]
label_enc = data["label_encoder"]
players = data["players"]

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
    
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    
    probabilities = model.predict(x)
    pred_index = np.argmax(probabilities, axis=1)[0]
        
    winner = label_encoder.inverse_transform([pred_index])[0]
    return winner, probabilities

# Predict on the latest race
predicted_winner, predictions = predict_winner(model, latest_race, players, label_enc)

print("\n--- Latest Race Data ---")
print(latest_race)
print("\n--- Prediction ---")
print(f"🔮 Predicted Winner: {predicted_winner}")

print("\n--- Prediction Rates ---")
for i, conch in enumerate(label_enc.classes_):
    rate = predictions[0][i] * 100
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
new_df.to_csv("conch_race_lightgbm_predictions.csv", index=False)
