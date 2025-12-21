# ===========================================
# 🐚 Conch Race Winner Prediction (RANKING)
# LightGBM LambdaRank - FULL PIPELINE
# ===========================================

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from typing import List, Tuple

from sklearn.preprocessing import LabelEncoder

from oauth2client.service_account import ServiceAccountCredentials
import gspread

import config
from config import load_config
from common import SHEET, SHEET_NAME

# ===========================================
# 1. Emoji sentiment map
# ===========================================

EMOJI_SENTIMENT = {
    "😡": (-1.0, 1.0),   # angry: negative + high arousal
    "😢": (-1.0, -1.0),  # sad
    "🖤": (-0.5, 0.0),   # dark / passive
    "😎": (1.0, 0.5),    # confident
    "😁": (1.0, 1.0),   # happy / hype
}

def parse_rate_emoji(cell: str) -> Tuple[float, float, float]:
    """
    Parse cell like: '24.7% 😎'
    Return: rate, valence, arousal
    """
    if not cell or str(cell).strip() == "":
        return 0.0, 0.0, 0.0

    text = str(cell)
    rate = 0.0
    valence, arousal = 0.0, 0.0

    try:
        if "%" in text:
            rate = float(text.split("%")[0].strip())
    except Exception:
        rate = 0.0

    for emoji, (v, a) in EMOJI_SENTIMENT.items():
        if emoji in text:
            valence, arousal = v, a
            break

    return rate, valence, arousal


# ===========================================
# 2. Load Google Sheet
# ===========================================

def load_sheet(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        config.CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)
    ws = client.open(sheet_name).worksheet(worksheet_name)

    values = ws.get_all_values()
    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)


# ===========================================
# 3. Reshape to ranking dataset
# ===========================================

def build_ranking_dataset(df: pd.DataFrame):
    EXCLUDE = {"Time", "Top 1", "Predict"}
    players = [c for c in df.columns if c not in EXCLUDE]

    X, y, group, race_ids = [], [], [], []

    for race_id, row in df.iterrows():
        rates = []
        temp = []

        for p in players:
            rate, val, aro = parse_rate_emoji(row[p])
            rates.append(rate)
            temp.append((p, rate, val, aro))

        rates = np.array(rates)
        rate_mean = rates.mean()
        rate_std = rates.std() + 1e-6

        group.append(len(players))

        for p, rate, val, aro in temp:
            features = [
                rate,
                val,
                aro,
                rate - rate_mean,         # relative rate
                rate / (rate_mean + 1e-6),
                (rate - rate_mean) / rate_std,
            ]

            label = 1 if row["Top 1"] == p else 0

            X.append(features)
            y.append(label)
            race_ids.append(race_id)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(group, dtype=np.int32),
        players,
    )


# ===========================================
# 4. Train / Val split by time
# ===========================================

def time_split(df: pd.DataFrame, ratio=0.9):
    df = df.sort_values("Time")
    cut = int(len(df) * ratio)
    return df.iloc[:cut], df.iloc[cut:]


# ===========================================
# 5. Top-1 Accuracy (race-level)
# ===========================================

def top1_accuracy(preds: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    correct = 0
    idx = 0
    for g in group:
        group_preds = preds[idx : idx + g]
        group_labels = y[idx : idx + g]

        if np.argmax(group_preds) == np.argmax(group_labels):
            correct += 1
        idx += g
    return correct / len(group)


# ===========================================
# 6. MAIN
# ===========================================

def main():
    load_config("config.ini")
    assert config.CREDENTIALS_PATH

    print("📥 Loading Google Sheet...")
    origin_df = load_sheet(SHEET, SHEET_NAME)

    df = origin_df[
        origin_df["Top 1"].notna()
        & (origin_df["Top 1"].astype(str).str.strip() != "")
    ].copy()

    print("✅ Races:", len(df))

    train_df, val_df = time_split(df)

    X_train, y_train, group_train, players = build_ranking_dataset(train_df)
    X_val, y_val, group_val, _ = build_ranking_dataset(val_df)

    print("🎯 Players:", players)
    print("📊 Train rows:", X_train.shape)
    print("📊 Val rows:", X_val.shape)

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=group_train,
        feature_name=[
            "rate",
            "emoji_valence",
            "emoji_arousal",
            "rate_minus_mean",
            "rate_div_mean",
            "rate_zscore",
        ],
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=group_val,
        reference=train_data,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
        "verbose": -1,
    }

    print("🚀 Training Ranker...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    val_preds = model.predict(X_val)
    acc = top1_accuracy(val_preds, y_val, group_val)

    print(f"🏆 Top-1 Accuracy (VAL): {acc:.4f}")

    joblib.dump(
        {
            "model": model,
            "players": players,
            "features": train_data.feature_name,
        },
        "conch_race_ranker.pkl",
    )

    print("💾 Model saved: conch_race_ranker.pkl")


# ===========================================
# Entry
# ===========================================

if __name__ == "__main__":
    main()
