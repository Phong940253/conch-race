import joblib
import numpy as np
import logging
import re
from typing import Dict, Tuple

# SAME emoji map used in training
EMOJI_SENTIMENT = {
    "😡": (-1.0, 1.0),
    "😢": (-1.0, -1.0),
    "🖤": (-0.5, 0.0),
    "😎": (1.0, 0.5),
    "😁": (1.0, 1.0),
}

_RATE_RE = re.compile(r"([-+]?\d+(?:[\.,]\d+)?)")


def _extract_rate_percent(text: str) -> float:
    if not text:
        return 0.0
    s = str(text).strip()
    if not s:
        return 0.0
    if "%" in s:
        s = s.split("%", 1)[0]
    m = _RATE_RE.search(s)
    if not m:
        return 0.0
    num = m.group(1).replace(",", ".")
    try:
        return float(num)
    except Exception:
        return 0.0


logger = logging.getLogger(__name__)

def parse_rate_emoji_rank(cell: str) -> Tuple[float, float, float]:
    if not cell or str(cell).strip() == "":
        return 0.0, 0.0, 0.0

    text = str(cell)
    rate = _extract_rate_percent(text)
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


def load_model(model_path, model_type="lightgbm"):
    if model_type != "lightgbm":
        raise ValueError("Ranking model supports LightGBM only")

    data = joblib.load(model_path)
    logger.info("Loaded ranking model: %s", model_path)
    return data["model"], data["players"], data["features"]


def predict_winner(
    model,
    players,
    features,
    ocr_data: Dict[str, Dict[str, str]],
):
    """
    Ranking-based prediction
    Returns:
        winner_name
        sorted_scores [(name, score)]
    """

    rows = []
    names = []

    # Build per-conch rows
    rates = []
    temp = []

    for p in players:
        info = ocr_data.get(p)
        if info:
            rate, val, aro = parse_rate_emoji_rank(
                f"{info['rate']} {info['emoji']}"
            )
        else:
            rate, val, aro = 0.0, 0.0, 0.0

        rates.append(rate)
        temp.append((p, rate, val, aro))

    rates = np.array(rates)
    mean = rates.mean()
    std = rates.std() + 1e-6

    for p, rate, val, aro in temp:
        row = [
            rate,
            val,
            aro,
            rate - mean,
            rate / (mean + 1e-6),
            (rate - mean) / std,
        ]
        rows.append(row)
        names.append(p)

        # Very verbose; keep in DEBUG only
        logger.debug("Predict features for %s: %s", p, row)

    X = np.asarray(rows, dtype=np.float32)

    scores = model.predict(X)

    ranking = sorted(
        zip(names, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    winner = ranking[0][0]

    return winner, ranking
