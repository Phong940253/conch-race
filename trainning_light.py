# ===========================================
# 🐚 Conch Race Winner Prediction
# Using LightGBM
# ===========================================

# --- 1. Imports ---
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from lightgbm import early_stopping, log_evaluation
from oauth2client.service_account import ServiceAccountCredentials
import gspread

import config
from config import load_config
from common import parse_rate_emoji, SHEET_NAME, SHEET


# ===========================================
# --- 2. Config & Data Loading ---
# ===========================================

def load_sheet_as_dataframe(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """Load Google Sheet safely using gspread (table-safe)."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        config.CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)
    worksheet = client.open(sheet_name).worksheet(worksheet_name)

    values = worksheet.get_all_values()
    if len(values) < 2:
        raise ValueError("Google Sheet has no data rows")

    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)


# Load config FIRST
load_config("config.ini")
assert config.CREDENTIALS_PATH, "CREDENTIALS_PATH is not loaded"

# Load data
origin_df = load_sheet_as_dataframe(SHEET, SHEET_NAME)
print("✅ Loaded Google Sheet:", origin_df.shape)

# Keep only rows with winner
df = origin_df[
    origin_df["Top 1"].notna()
    & (origin_df["Top 1"].astype(str).str.strip() != "")
].copy()

print("✅ Training rows:", df.shape)


# ===========================================
# --- 3. Feature Engineering ---
# ===========================================

EXCLUDE_COLUMNS = {"Time", "Top 1", "Predict"}
players = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

# ---- Feature names ----
# Safe names for LightGBM
lgb_feature_names = []
# Pretty names for plotting
plot_feature_names = []

for idx, name in enumerate(players):
    safe_base = f"p{idx}"
    lgb_feature_names.append(f"{safe_base}_rate")
    lgb_feature_names.append(f"{safe_base}_emoji")

    plot_feature_names.append(f"{name} (rate)")
    plot_feature_names.append(f"{name} (emoji)")

X_data, y_data = [], []

for _, row in df.iterrows():
    features = []
    for player in players:
        rate, emoji = parse_rate_emoji(row[player])
        features.extend([rate, emoji])
    X_data.append(features)
    y_data.append(row["Top 1"])

X = np.asarray(X_data, dtype=np.float32)
y = np.asarray(y_data)

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)
print(f"✅ Features: {X.shape} | Classes: {num_classes}")

assert X.shape[1] == len(lgb_feature_names)


# ===========================================
# --- 4. Train / Validation Split ---
# ===========================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.1, random_state=42
)


# ===========================================
# --- 5. Train LightGBM ---
# ===========================================

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=lgb_feature_names,
)
val_data = lgb.Dataset(
    X_val,
    label=y_val,
    reference=train_data,
)

params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbose": -1,
}

print("🚀 Training LightGBM...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50),
    ],
)


# ===========================================
# --- 6. Evaluation ---
# ===========================================

y_pred = np.argmax(model.predict(X_val), axis=1)
accuracy = accuracy_score(y_val, y_pred)

print(f"✅ Validation Accuracy: {accuracy:.4f}")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)


# ===========================================
# --- 7. Feature Importance (PRETTY NAMES) ---
# ===========================================

importance = model.feature_importance()
indices = np.argsort(importance)[::-1][:20]

fig, ax = plt.subplots(figsize=(12, 6))
lgb.plot_importance(
    model,
    max_num_features=20,
    ax=ax,
)

ax.set_yticks(range(len(indices)))
ax.set_yticklabels([plot_feature_names[i] for i in indices])
ax.set_title("Top 20 Feature Importances")

plt.tight_layout()
plt.show()


# ===========================================
# --- 8. Sanity Prediction ---
# ===========================================

def predict_winner(model, row: pd.Series) -> str:
    features = []
    for player in players:
        rate, emoji = parse_rate_emoji(row[player])
        features.extend([rate, emoji])
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    pred = np.argmax(model.predict(X), axis=1)[0]
    return label_encoder.inverse_transform([pred])[0]


sample = origin_df.iloc[-1]
print("🔮 Sample prediction:", predict_winner(model, sample))


# ===========================================
# --- 9. Save Model ---
# ===========================================

MODEL_PATH = "conch_race_lightgbm_model.pkl"
joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "players": players,
    },
    MODEL_PATH,
)

print(f"✅ Model saved to {MODEL_PATH}")
