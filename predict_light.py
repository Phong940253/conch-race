# ===========================================
# 🐚 Conch Race Winner Prediction (RANKING)
# LightGBM LambdaRank - PREDICT SCRIPT
# ===========================================

import numpy as np
import pandas as pd
import joblib

from oauth2client.service_account import ServiceAccountCredentials
import gspread

import config
from config import load_config
from common import SHEET, SHEET_NAME

# ===========================================
# 1. Emoji sentiment (MUST MATCH TRAINING)
# ===========================================

EMOJI_SENTIMENT = {
    "😡": (-1.0, 1.0),
    "😢": (-1.0, -1.0),
    "🖤": (-0.5, 0.0),
    "😎": (1.0, 0.5),
    "😁": (1.0, 1.0),
}


def parse_rate_emoji(cell: str):
    if not cell or str(cell).strip() == "":
        return 0.0, 0.0, 0.0

    text = str(cell)
    rate = 0.0
    valence = arousal = 0.0

    try:
        if "%" in text:
            rate = float(text.split("%")[0].strip())
    except Exception:
        pass

    for emoji, (v, a) in EMOJI_SENTIMENT.items():
        if emoji in text:
            valence, arousal = v, a
            break

    return rate, valence, arousal


# ===========================================
# 2. Load Google Sheet (IDENTICAL TO TRAINING)
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
# 3. Load Model
# ===========================================

MODEL_PATH = "conch_race_ranker.pkl"

data = joblib.load(MODEL_PATH)
model = data["model"]
players = data["players"]
features = data["features"]

print(f"✅ Loaded ranking model: {MODEL_PATH}")
print("🎯 Players:", players)


# ===========================================
# 4. Ranking Prediction Logic (SAME AS TRAINING)
# ===========================================

def predict_race(row: pd.Series):
    rates = []
    temp = []

    for p in players:
        rate, val, aro = parse_rate_emoji(row.get(p, ""))
        rates.append(rate)
        temp.append((p, rate, val, aro))

    rates = np.array(rates, dtype=np.float32)
    mean = rates.mean()
    std = rates.std() + 1e-6

    X = []
    names = []

    for p, rate, val, aro in temp:
        X.append([
            rate,
            val,
            aro,
            rate - mean,
            rate / (mean + 1e-6),
            (rate - mean) / std,
        ])
        names.append(p)

    X = np.asarray(X, dtype=np.float32)

    scores = model.predict(X)

    ranking = sorted(
        zip(names, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # softmax → pseudo-probabilities
    score_arr = np.array([s for _, s in ranking])
    exp_scores = np.exp(score_arr - score_arr.max())
    probs = exp_scores / exp_scores.sum()

    probabilities = [
        (name, prob * 100)
        for (name, _), prob in zip(ranking, probs)
    ]

    winner = ranking[0][0]
    return winner, ranking, probabilities


# ===========================================
# 5. MAIN
# ===========================================

def main():
    load_config("config.ini")
    assert config.CREDENTIALS_PATH

    print("📥 Loading Google Sheet (training-safe)...")
    origin_df = load_sheet(SHEET, SHEET_NAME)

    # same filtering as training
    df = origin_df[
        origin_df["Top 1"].notna()
        & (origin_df["Top 1"].astype(str).str.strip() != "")
    ].copy()

    print("✅ Total races:", len(df))

    # ---- Predict latest race ----
    latest_race = df.iloc[-1]

    winner, ranking, probabilities = predict_race(latest_race)

    print("\n🔮 Latest Race Prediction")
    print("Winner:", winner)

    print("\n📊 Win Confidence (Ranking)")
    for name, prob in probabilities:
        print(f"{name}: {prob:.2f}%")

    # ---- Predict all races ----
    all_predictions = []
    for _, row in df.iterrows():
        w, _, _ = predict_race(row)
        all_predictions.append(w)

    result_df = df.copy()
    result_df["Predicted Winner"] = all_predictions

    # remove old Predict column if exists
    result_df = result_df.loc[
        :, ~result_df.columns.str.contains("^Predict$", case=False)
    ]

    print("\n🧾 Sample predictions")
    print(result_df.head(20))

    OUTPUT = "conch_race_ranker_predictions.csv"
    result_df.to_csv(OUTPUT, index=False)
    print(f"\n💾 Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
