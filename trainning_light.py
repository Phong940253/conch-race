# ===========================================
# 🐚 Conch Race Winner Prediction Notebook
# Using LightGBM
# ===========================================

# --- 1. Setup ---
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import urllib.parse
import joblib
from common import parse_rate_emoji, SHEET_NAME, SHEET
from lightgbm import early_stopping, log_evaluation
import config
from config import load_config

# --- 2. Load data ---
# --- 2. Load data (SAFE: same logic as runtime) ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def load_sheet_as_dataframe(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        config.CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)
    sheet = client.open(sheet_id).worksheet(worksheet_name)

    values = sheet.get_all_values()
    if not values or len(values) < 2:
        raise ValueError("Sheet has no data rows")

    header = values[0]
    rows = values[1:]

    df = pd.DataFrame(rows, columns=header)
    return df

# --- Load config FIRST ---
load_config("config.ini")
assert config.CREDENTIALS_PATH is not None

origin_df = load_sheet_as_dataframe(SHEET, SHEET_NAME)
print("✅ Loaded Google Sheet via gspread. Shape:", origin_df.shape)

# Filter rows where winner exists
df = origin_df[
    origin_df["Top 1"].notna()
    & (origin_df["Top 1"].astype(str).str.strip() != "")
].copy()

print("✅ Filtered training rows:", df.shape)


# Filter rows where 'Top 1' (winner) exists
df = origin_df[origin_df['Top 1'].notna() & (origin_df['Top 1'] != "")]
print("✅ Filtered data:", df.shape)

# --- 3. Transform dataset ---
players = [c for c in df.columns if c not in ["Time", "Top 1", "Predict"]]

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

# Encode target labels
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

num_classes = len(label_enc.classes_)
input_dim = X.shape[1]

print(f"✅ Data ready: X={X.shape}, Classes={num_classes}")

# --- 4. Train/Validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.1, random_state=42)

# --- 5. Train LightGBM model ---
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

print("🚀 Training LightGBM model...")
model = lgb.train(
    params=params,
    train_set=train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=500,
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)

# --- 6. Evaluate model ---
y_pred_val = np.argmax(model.predict(X_val), axis=1)
acc = accuracy_score(y_val, y_pred_val)
print(f"✅ Validation Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred_val,
    target_names=label_enc.classes_,
    zero_division=0
))

# --- 7. Feature importance ---
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.show()

# --- 8. Test prediction ---
def predict_winner(model, row):
    features = []
    for p in players:
        rate, emo = parse_rate_emoji(row[p])
        features += [rate, emo]
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    pred = np.argmax(model.predict(features), axis=1)[0]
    return label_enc.inverse_transform([pred])[0]

# Test on the last row
sample = origin_df.iloc[-1]
print("Sample row:\n", sample)
print("🔮 Predicted Winner:", predict_winner(model, sample))

# --- 9. Save model ---
MODEL_PATH = "conch_race_lightgbm_model.pkl"
joblib.dump({
    "model": model,
    "label_encoder": label_enc,
    "players": players
}, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# --- Optional: Visualize class distribution ---
# pd.Series(y).value_counts().plot(kind='bar', title='Winner Class Distribution')
# plt.show()