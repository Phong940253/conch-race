# ===========================================
# 🐚 Conch Race Prediction Explainer (RANKING)
# LightGBM LambdaRank - VISUALIZE + EXPLAIN
# ===========================================

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oauth2client.service_account import ServiceAccountCredentials
import gspread

import config
from config import load_config
from common import SHEET as DEFAULT_SHEET_NAME, SHEET_NAME as DEFAULT_WORKSHEET_NAME


FEATURE_NAMES = [
    "rate",
    "emoji_valence",
    "emoji_arousal",
    "rate_minus_mean",
    "rate_div_mean",
    "rate_zscore",
]


# MUST MATCH training/model.py
EMOJI_SENTIMENT: Dict[str, Tuple[float, float]] = {
    "😡": (-1.0, 1.0),
    "😢": (-1.0, -1.0),
    "🖤": (-0.5, 0.0),
    "😎": (1.0, 0.5),
    "😁": (1.0, 1.0),
}


def parse_rate_emoji(cell: str) -> Tuple[float, float, float]:
    if not cell or str(cell).strip() == "":
        return 0.0, 0.0, 0.0

    text = str(cell)
    rate = 0.0
    valence = arousal = 0.0

    try:
        if "%" in text:
            rate = float(text.split("%")[0].strip().replace(",", "."))
    except Exception:
        pass

    for emoji, (v, a) in EMOJI_SENTIMENT.items():
        if emoji in text:
            valence, arousal = v, a
            break

    return rate, valence, arousal


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
    if not values:
        return pd.DataFrame()

    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)


def build_features_for_race(row: pd.Series, players: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    rates: List[float] = []
    temp: List[Tuple[str, float, float, float]] = []

    for p in players:
        rate, val, aro = parse_rate_emoji(row.get(p, ""))
        rates.append(rate)
        temp.append((p, rate, val, aro))

    rates_arr = np.asarray(rates, dtype=np.float32)
    mean = float(rates_arr.mean()) if len(rates_arr) else 0.0
    std = float(rates_arr.std()) + 1e-6

    X: List[List[float]] = []
    names: List[str] = []

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

    return np.asarray(X, dtype=np.float32), names


def softmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    e = np.exp(s - np.max(s))
    return e / np.sum(e)


@dataclass
class RaceExplanation:
    timestamp: str
    ranking_df: pd.DataFrame
    winner: str


def explain_race(model, row: pd.Series, players: Sequence[str]) -> RaceExplanation:
    X, names = build_features_for_race(row, players)
    scores = np.asarray(model.predict(X), dtype=np.float64)
    probs = softmax(scores) * 100.0

    # LightGBM pred_contrib returns (n_samples, n_features + 1) with bias term last.
    contrib = np.asarray(model.predict(X, pred_contrib=True), dtype=np.float64)
    if contrib.shape[1] == len(FEATURE_NAMES) + 1:
        contrib_feature = contrib[:, : len(FEATURE_NAMES)]
        contrib_bias = contrib[:, -1]
    else:
        # Fallback: shape unexpected; still provide basic score/prob table.
        contrib_feature = np.zeros((len(names), len(FEATURE_NAMES)), dtype=np.float64)
        contrib_bias = np.zeros((len(names),), dtype=np.float64)

    features_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    contrib_df = pd.DataFrame(contrib_feature, columns=[f"Δ {c}" for c in FEATURE_NAMES])

    df = pd.DataFrame({
        "name": names,
        "score": scores,
        "prob%": probs,
        "bias": contrib_bias,
    })

    df = pd.concat([df, features_df, contrib_df], axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    winner = str(df.iloc[0]["name"]) if len(df) else ""

    # best-effort timestamp
    ts = str(row.get("Timestamp", row.get("Time", "")) or "")
    ts = ts.strip() or "(no timestamp column)"

    return RaceExplanation(timestamp=ts, ranking_df=df, winner=winner)


def _fmt_top_contrib(row: pd.Series, top_n: int = 3) -> str:
    contrib_cols = [c for c in row.index if c.startswith("Δ ")]
    pairs = [(c.replace("Δ ", ""), float(row[c])) for c in contrib_cols]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    parts = []
    for name, val in pairs[:top_n]:
        sign = "+" if val >= 0 else ""
        parts.append(f"{name}={sign}{val:.3f}")
    return ", ".join(parts)


def print_explanation(expl: RaceExplanation, top_k: int = 6) -> None:
    df = expl.ranking_df
    print("\n" + "=" * 72)
    print(f"🕒 {expl.timestamp}")
    print(f"🏆 Winner: {expl.winner}")

    view = df.head(top_k).copy()
    view["top_contrib"] = view.apply(_fmt_top_contrib, axis=1)

    cols = ["name", "score", "prob%", "rate", "emoji_valence", "emoji_arousal", "top_contrib"]
    print(view[cols].to_string(index=False, justify="left", float_format=lambda x: f"{x:0.3f}"))

    if len(df) >= 2:
        first = df.iloc[0]
        second = df.iloc[1]
        gap = float(first["score"]) - float(second["score"])
        print(f"\nScore gap vs #2: {gap:.3f}")


def plot_race(expl: RaceExplanation, outdir: str, show: bool) -> None:
    df = expl.ranking_df
    if df.empty:
        return

    safe_ts = expl.timestamp.replace(":", "-").replace("/", "-").replace("\\", "-")
    safe_ts = safe_ts.replace(" ", "_")
    base = f"explain_{safe_ts}"

    # --- Plot 1: score/prob ranking ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = df["name"].tolist()[::-1]
    scores = df["score"].tolist()[::-1]
    probs = df["prob%"].tolist()[::-1]

    colors = ["#c9c9c9"] * len(names)
    if expl.winner in df["name"].values:
        win_idx = names.index(expl.winner)  # reversed list
        colors[win_idx] = "#2ecc71"

    ax.barh(names, scores, color=colors)
    ax.set_title(f"Conch Race Ranker — Scores ({expl.timestamp})")
    ax.set_xlabel("LightGBM score")

    for i, (s, p) in enumerate(zip(scores, probs)):
        ax.text(s, i, f"  {p:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path1 = os.path.join(outdir, base + "_scores.png")
    fig.savefig(path1, dpi=160)

    # --- Plot 2: winner contributions ---
    winner_row = df.iloc[0]
    contrib_cols = [c for c in df.columns if c.startswith("Δ ")]
    contrib_vals = [(c.replace("Δ ", ""), float(winner_row[c])) for c in contrib_cols]
    contrib_vals.sort(key=lambda t: abs(t[1]), reverse=True)

    top = contrib_vals[:6]
    feat = [t[0] for t in top][::-1]
    val = [t[1] for t in top][::-1]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    c2 = ["#3498db" if v >= 0 else "#e74c3c" for v in val]
    ax2.barh(feat, val, color=c2)
    ax2.axvline(0, color="#666", linewidth=1)
    ax2.set_title(f"Winner feature contributions — {expl.winner}")
    ax2.set_xlabel("Contribution to score")
    fig2.tight_layout()
    path2 = os.path.join(outdir, base + "_winner_contrib.png")
    fig2.savefig(path2, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Explain LightGBM LambdaRank predictions with plots + per-feature contributions."
    )
    ap.add_argument("--config", default="config.ini", help="Path to config.ini")
    ap.add_argument("--model", default="conch_race_ranker.pkl", help="Path to ranker model")

    ap.add_argument("--sheet", default=DEFAULT_SHEET_NAME, help="Google Sheet file name")
    ap.add_argument("--worksheet", default=DEFAULT_WORKSHEET_NAME, help="Worksheet name")
    ap.add_argument("--csv", default=None, help="Optional local CSV file instead of Google Sheets")

    ap.add_argument("--last", type=int, default=1, help="Explain last N rows")
    ap.add_argument(
        "--iloc",
        default=None,
        help="Comma-separated pandas iloc indices (supports negative). Example: --iloc -1,-2",
    )

    ap.add_argument("--outdir", default="explain_outputs", help="Output directory for plots")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")

    args = ap.parse_args()

    load_config(args.config)

    data = joblib.load(args.model)
    model = data["model"]
    players = data["players"]

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        if not config.CREDENTIALS_PATH:
            raise RuntimeError("config.CREDENTIALS_PATH is empty; check config.ini")
        df = load_sheet(args.sheet, args.worksheet)

    if df.empty:
        print("No rows found.")
        return

    # Drop rows that have no data for any player column (common in partially-filled sheets)
    player_cols = [p for p in players if p in df.columns]
    if player_cols:
        mask_any = df[player_cols].astype(str).apply(lambda c: c.str.strip().ne(""), axis=0).any(axis=1)
        df = df[mask_any].copy()

    if df.empty:
        print("No usable rows found (player columns are empty).")
        return

    if args.iloc:
        indices = _parse_int_list(args.iloc)
        rows = [df.iloc[i] for i in indices]
    else:
        rows = [df.iloc[i] for i in range(max(0, len(df) - args.last), len(df))]

    for row in rows:
        expl = explain_race(model, row, players)
        print_explanation(expl, top_k=6)
        plot_race(expl, outdir=args.outdir, show=args.show)

    print(f"\nSaved plots to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
