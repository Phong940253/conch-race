import requests
import logging

def send_discord_notification(data, prediction, probabilities, label_encoder, webhook_url, debug=False, duplicate_row=None):
    """Sends a notification to a Discord webhook with the race results and prediction rates."""
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
            
        if duplicate_row:
            # If duplicate, highlight the winner
            embed["fields"].append({
                "name": "⚠️ Duplicate Detected",
                "value": f"Winner was: **{duplicate_row}**",
                "inline": False
            })
            # yellow color
            embed["color"] = 0xffff00
            
        
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
            if duplicate_row:
                payload["allowed_mentions"] = {"parse": ["everyone"]}

        response = requests.post(webhook_url, json=payload)
        
        # tag everyone if duplicate
        if duplicate_row:
            new_payload = {
                "content": f"@everyone\n⚠️ Duplicate data detected."
            }
            requests.post(webhook_url, json=new_payload)
            
        response.raise_for_status()
        logging.info("Discord notification sent successfully.")
    except Exception as e:
        logging.error(f"Error sending Discord notification: {e}")
