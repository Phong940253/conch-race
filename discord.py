import requests
import logging
import traceback

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
    row_data: 1 dòng sheet
    sheet_conch_order: LIST_CONCH (thứ tự cột sheet)
    race_conch_order: list(data.keys()) (thứ tự race hiện tại)
    """

    # Lấy emoji theo sheet (bỏ timestamp, bỏ winner)
    sheet_emojis = row_data[1:-1]

    # Map: tên tay đua -> emoji lịch sử
    sheet_map = {
        conch: emoji
        for conch, emoji in zip(sheet_conch_order, sheet_emojis)
        if emoji and emoji.strip()
    }

    # Reorder theo race hiện tại
    ordered_emojis = [
        sheet_map.get(conch, "")
        for conch in race_conch_order
    ]

    return ordered_emojis


def send_discord_notification(data, prediction, probabilities, label_encoder, webhook_url, debug=False, matched_rows=None):
    """Sends a notification to a Discord webhook with the race results and prediction rates."""
    
    from config import (LIST_CONCH)
    
    try:
        
        embed = {
            "title": "🏁 Conch Race Results",
            "description": "A new race has been processed!",
            "color": 0x00ff00,  # Green
            "fields": [],
            "footer": {
                "text": "Conch Race OCR Bot"
            }
        }

        if debug:
            embed["title"] = "🐞 Debug Mode: " + embed["title"]
            embed["color"] = 0xff0000  # Red

        for name, info in data.items():
            embed["fields"].append({
                "name": name,
                "value": f"Rate: {info['rate']} {info['emoji']}",
                "inline": True
            })

        if prediction:
            embed["fields"].append({
                "name": "🔮 Predicted Winner",
                "value": prediction,
                "inline": False
            })
            
        conch_names = list(data.keys())   # thứ tự đã đúng theo OCR
        num_conch = len(conch_names)      # thường = 6

            
        PERFECT_MATCH_SCORE = num_conch
        has_perfect_match = False

        if matched_rows:
            conch_names = list(data.keys())
            num_conch = len(conch_names)

            # Header
            MAX_COL_WIDTH = 5
            MAX_COL_EMOJI_WIDTH = 1

            conch_names = list(data.keys())
            num_conch = len(conch_names)

            short_names = [
                shorten_and_center(name, MAX_COL_WIDTH)
                for name in conch_names
            ]

            MAX_WINNER_WIDTH = 5

            header_cols = (
                [center_cell("Row", 3)] +
                short_names +
                [center_cell("Winner", MAX_WINNER_WIDTH)] +
                [center_cell("Score", 5)]
            )

            header_line = " | ".join(header_cols)
            table_lines = [header_line]
            table_lines.append("-" * len(header_line))

            for m in matched_rows:
                # 🔥 FIX: detect perfect match
                if m.get("score") == PERFECT_MATCH_SCORE:
                    has_perfect_match = True

                row_num = center_cell(str(m["row_number"]), 3)
                score = center_cell(f"{m['score']}/{num_conch}", 5)

                emojis = reorder_emojis_by_race(
                    m["row_data"],
                    LIST_CONCH,              # thứ tự cột sheet
                    list(data.keys())        # thứ tự race hiện tại
                )
                emoji_cells = [
                    center_cell(e or "", MAX_COL_EMOJI_WIDTH)
                    for e in emojis
                ]

                winner = m["row_data"][-1] if m["row_data"] else ""
                winner_cell = shorten_and_center(winner, MAX_WINNER_WIDTH)

                row_line = " | ".join(
                    [row_num] +
                    emoji_cells +
                    [winner_cell] +
                    [score]
                )
                table_lines.append(row_line)

            table_text = "```\n" + "\n".join(table_lines)[:1800] + "\n```"
            
            if has_perfect_match:
                embed["fields"].append({
                    "name": "⚠️ Duplicate Detected",
                    "value": f"Winner was: **{matched_rows[0]['row_data'][-1]}**",
                    "inline": False
                })
            else:
                embed["fields"].append({
                    "name": "⚠️ Historical Match Table",
                    "value": table_text,
                    "inline": False
                })
                    

            # Màu sắc theo mức độ
            embed["color"] = 0x00ff00 if not has_perfect_match else 0xffff00
            
        
        if probabilities is not None:
            # Create a list of (conch, rate) tuples
            conch_rates = []
            for i, conch in enumerate(label_encoder.classes_):
                rate = probabilities[0][i].item() * 100
                conch_rates.append((conch, rate))
            
            # Sort the list by rate in descending order
            conch_rates.sort(key=lambda x: x[1], reverse=True)
            
            # Format the sorted rates into a string
            rates_message = ""
            for conch, rate in conch_rates:
                rates_message += f"{conch}: {rate:.2f}%\n"
            
            if rates_message:
                embed["fields"].append({
                    "name": "📊 Prediction Rates",
                    "value": rates_message,
                    "inline": False
                })
            
            payload = {"embeds": [embed]}
            if has_perfect_match:
                payload["allowed_mentions"] = {"parse": ["everyone"]}

        response = requests.post(webhook_url, json=payload)
        
        # tag everyone if duplicate
        if has_perfect_match:
            new_payload = {
                "content": f"@everyone\n⚠️ Duplicate data detected!",
            }
            requests.post(webhook_url, json=new_payload)
            
        response.raise_for_status()
        logging.info("Discord notification sent successfully.")
    except Exception as e:
        logging.error(traceback.format_exc())
    