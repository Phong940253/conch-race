import requests
import logging
from config import WEBHOOK_URL

def send_discord_notification(data, prediction, debug=False):
    """Sends a notification to a Discord webhook with the race results."""
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

        payload = {"embeds": [embed]}
        response = requests.post(WEBHOOK_URL, json=payload)
        response.raise_for_status()
        logging.info("Discord notification sent successfully.")
    except Exception as e:
        logging.error(f"Error sending Discord notification: {e}")
