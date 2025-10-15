# ===========================================
# 🐚 Conch Race Winner Prediction Notebook
# Using PyTorch Lightning
# ===========================================

# --- 1. Setup ---
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import urllib.parse
from common import ConchPredictor, parse_rate_emoji, SHEET_ID, SHEET_NAME

# --- 2. Load data ---
# --- Convert sheet name to encoded URL format ---
encoded_name = urllib.parse.quote(SHEET_NAME)

# --- Create CSV export link ---
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={encoded_name}"

# --- Load into pandas ---
origin_df = pd.read_csv(csv_url)
print("✅ Loaded Google Sheet successfully! Shape:", origin_df.shape)
origin_df.head()

# filter Top 1 not none
df = origin_df[origin_df['Top 1'].notna() & (origin_df['Top 1'] != "")]

# --- 3. Transform dataset ---
players = [c for c in df.columns if c not in ["Time", "Top 1"]]

X_data = []
y_data = []

for _, row in df.iterrows():
    row_features = []
    for p in players:
        rate, emo = parse_rate_emoji(row[p])
        row_features += [rate, emo]
    X_data.append(row_features)
    y_data.append(row["Top 1"])

X = np.array(X_data, dtype=np.float32)
y = np.array(y_data)

# Encode winners
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

num_classes = len(label_enc.classes_)
input_dim = X.shape[1]

print(f"✅ Data ready: {X.shape}, classes = {num_classes}")

# --- 5. Dataset and DataLoader ---
class ConchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.1, random_state=42)
train_ds = ConchDataset(X_train, y_train)
val_ds = ConchDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# --- 6. Train model ---
model = ConchPredictor(input_dim, num_classes)

trainer = pl.Trainer(max_epochs=100, enable_checkpointing=False, log_every_n_steps=1)
trainer.fit(model, train_loader, val_loader)

# --- 7. Test prediction ---
def predict_winner(model, row):
    features = []
    for p in players:
        rate, emo = parse_rate_emoji(row[p])
        features += [rate, emo]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    pred = model(x).argmax(dim=1).item()
    return label_enc.inverse_transform([pred])[0]

# Test on last row
sample = origin_df.iloc[-1]
print(sample)
print("🔮 Predicted:", predict_winner(model, sample))

# --- 8. Save trained model ---

MODEL_PATH = "conch_race_model.pt"

# Save the model weights
torch.save({
    "model_state_dict": model.state_dict(),
    "label_encoder": label_enc.classes_,  # Save winner label mapping too
    "players": players,                   # Save column order
    "input_dim": input_dim,
    "num_classes": num_classes
}, MODEL_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
