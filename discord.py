import requests
import logging
import traceback
import numpy as np


# =======================
# Helpers
# =======================

def shorten_and_center(name: str, width: int) -> str:
    if len(name) > width:
        name = name[: width - 1] + "…"
    return center_cell(name, width)


def center_cell(text: str, width: int) -> str:
    text = text or ""
    if len(text) >= width:
        return text[:width]
    left = (width - len(text)) // 2
    right = width - len(text) - left
    return " " * left + text + " " * right


def reorder_emojis_by_race(row_data, sheet_conch_order, race_conch_order):
    """
    row_data: one row from sheet
    sheet_conch_order: LIST_CONCH (sheet column order)
    race_conch_order: OCR order (current race)
    """

    # strip timestamp + winner
    sheet_emojis = row_data[1:-1]

    sheet_map = {
        conch: emoji
        for conch, emoji in zip(sheet_conch_order, sheet_emojis)
        if emoji and emoji.strip()
    }

    return [sheet_map.get(conch, "") for conch in race_conch_order]

def format_ranking_with_gap(ranking):
    """
    ranking: List[(name, score)] sorted desc
    """
    winner_score = ranking[0][1]
    lines = []

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]

    for i, (name, score) in enumerate(ranking):
        gap = score - winner_score
        medal = medals[i] if i < len(medals) else f"{i+1}."
        if i == 0:
            lines.append(f"{medal} {name}  (score {score:.2f})")
        else:
            lines.append(f"{medal} {name}  ({gap:.2f})")

    return "\n".join(lines)

def format_wsi_padded(ranking, width_name=22, width_bar=10):
    """
    ranking: List[(name, score)] TOP-6 already
    returns: formatted string for Discord code block
    """
    # compute WSI
    scores = np.array([s for _, s in ranking], dtype=np.float32)
    max_s = scores.max()
    min_s = scores.min()

    if max_s == min_s:
        wsi_values = [100] * len(ranking)
    else:
        wsi_values = ((scores - min_s) / (max_s - min_s) * 100).round().astype(int)

    lines = []
    for (name, _), wsi in zip(ranking, wsi_values):
        bar_filled = int(round(wsi / 100 * width_bar))
        bar = "█" * bar_filled + "░" * (width_bar - bar_filled)

        line = (
            f"{name.ljust(width_name)} "
            f"{bar} "
            f"{str(wsi).rjust(3)}"
        )
        lines.append(line)

    return "```\n" + "\n".join(lines) + "\n```"

# =======================
# Main Discord Function
# =======================

def send_discord_notification(
    data,
    prediction,
    ranking,
    debug=False,
    matched_rows=None,
):
    """
    data: OCR data
    prediction: predicted winner name
    ranking: List[(conch_name, rank_score)]
    """

    from config import LIST_CONCH, WEBHOOK_URL

    try:
        embed = {
            "title": "🏁 Conch Race Results",
            "description": "A new race has been processed!",
            "color": 0x00FF00,
            "fields": [],
            "footer": {"text": "Conch Race OCR Bot"},
        }

        if debug:
            embed["title"] = "🐞 Debug Mode — " + embed["title"]
            embed["color"] = 0xFF0000

        # =======================
        # OCR Results
        # =======================
        for name, info in data.items():
            embed["fields"].append({
                "name": name,
                "value": f"Rate: {info['rate']} {info['emoji']}",
                "inline": True,
            })

        # =======================
        # Prediction
        # =======================
        if prediction:
            embed["fields"].append({
                "name": "🔮 Predicted Winner",
                "value": prediction,
                "inline": False,
            })

        # =======================
        # Duplicate Detection
        # =======================
        num_conch = len(data)
        PERFECT_MATCH_SCORE = num_conch
        has_perfect_match = False

        # =======================
        # Ranking Probabilities
        # =======================
        if ranking:
            top_ranking = ranking[:6]
            ranking_text = format_ranking_with_gap(top_ranking)

            embed["fields"].append({
                "name": "📊 Rank & Confidence Gap",
                "value": ranking_text,
                "inline": False,
            })
            
            wsi_tables = format_wsi_padded(top_ranking)

            embed["fields"].append({
                "name": "💪 Win Strength Index (WSI)",
                "value": wsi_tables,
                "inline": False,
            })
            
        # =======================
        # Historical Match Table
        # =======================
        if matched_rows:
            MAX_COL_WIDTH = 5
            MAX_COL_EMOJI_WIDTH = 1
            MAX_WINNER_WIDTH = 5

            conch_names = list(data.keys())
            short_names = [
                shorten_and_center(name, MAX_COL_WIDTH)
                for name in conch_names
            ]

            header_cols = (
                [center_cell("Row", 3)]
                + short_names
                + [center_cell("Winner", MAX_WINNER_WIDTH)]
                + [center_cell("Score", 5)]
            )

            header_line = " | ".join(header_cols)
            table_lines = [header_line, "-" * len(header_line)]

            for m in matched_rows:
                if m.get("score") == PERFECT_MATCH_SCORE:
                    has_perfect_match = True

                row_num = center_cell(str(m["row_number"]), 3)
                score = center_cell(f"{m['score']}/{num_conch}", 5)

                emojis = reorder_emojis_by_race(
                    m["row_data"],
                    LIST_CONCH,
                    list(data.keys()),
                )

                emoji_cells = [
                    center_cell(e or "", MAX_COL_EMOJI_WIDTH)
                    for e in emojis
                ]

                winner = m["row_data"][-1] if m["row_data"] else ""
                winner_cell = shorten_and_center(winner, MAX_WINNER_WIDTH)

                row_line = " | ".join(
                    [row_num]
                    + emoji_cells
                    + [winner_cell]
                    + [score]
                )

                table_lines.append(row_line)

            table_text = "```\n" + "\n".join(table_lines)[:1800] + "\n```"

            if has_perfect_match:
                embed["fields"].append({
                    "name": "⚠️ Duplicate Detected",
                    "value": f"Winner was: **{matched_rows[0]['row_data'][-1]}**",
                    "inline": False,
                })
            else:
                embed["fields"].append({
                    "name": "📜 Historical Match Table",
                    "value": table_text,
                    "inline": False,
                })

            embed["color"] = 0xFFFF00 if has_perfect_match else embed["color"]

        payload = {"embeds": [embed]}

        if has_perfect_match:
            payload["allowed_mentions"] = {"parse": ["everyone"]}

        # =======================
        # Send to Discord
        # =======================
        for url in WEBHOOK_URL:
            response = requests.post(url, json=payload)
            response.raise_for_status()

            if has_perfect_match:
                requests.post(
                    url,
                    json={"content": "@everyone\n⚠️ Duplicate data detected!"},
                )

            logging.info(f"Discord notification sent successfully to {url}")

    except Exception:
        logging.error(traceback.format_exc())
