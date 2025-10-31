import logging
import pyautogui
import pygetwindow as gw
import mss
import numpy as np
import cv2
import os
from datetime import datetime
import time

def click_image(image_path, confidence=0.7, region=None, sleep_time=2):
    """Finds and clicks the center of an image on the screen."""
    try:
        location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence, region=region)
        if location:
            pyautogui.click(location)
            # logging.info(f"Clicked on {image_path}.")
            time.sleep(sleep_time)  # Wait for a short duration after clicking
            return True
        else:
            # logging.warning(f"{image_path} not found on screen.")
            return False
    except Exception as e:
        # logging.error(f"An error occurred while trying to click {image_path}: {e}")
        return False

def auto_bet(predicted_winner, conch_regions):
    """Automates the betting process based on the predicted winner."""
    if not predicted_winner or not conch_regions:
        logging.error("Auto-betting skipped: No prediction or region data.")
        return

    # logging.info(f"Starting auto-bet for predicted winner: {predicted_winner}")

    # 1. Click the support button
    if not click_image('support.png'):
        logging.error("Could not find support button.")
        return

    # 2. Click the increase button for the predicted winner
    winner_region = conch_regions.get(predicted_winner)
    logging.info(f"Winner region: {winner_region}")
    logging.info(f"Predicted winner: {predicted_winner}")
    logging.info(f"Conch regions: {conch_regions}")
    if not winner_region:
        logging.error(f"Could not find region for predicted winner: {predicted_winner}")
        return
    
    # click button increase 3 times
    for _ in range(3):
        if not click_image('increase.png', region=winner_region, sleep_time=0.5):
            logging.error(f"Could not find increase button for {predicted_winner}")
            return

    # 3. Click the first confirm button
    if not click_image('confirm1.png'):
        logging.error("Could not find first confirm button.")
        return

    # 4. Click the second confirm button
    if not click_image('confirm2.png'):
        logging.error("Could not find second confirm button.")
        return
    
    logging.info(f"Auto-bet process completed for {predicted_winner}.")

def click_refresh_button():
    """Finds and clicks the refresh button on the screen."""
    try:
        refresh_button_location = pyautogui.locateCenterOnScreen('refresh.png', confidence=0.8)
        if refresh_button_location:
            pyautogui.click(refresh_button_location)
            logging.info("Clicked the refresh button.")
            return True
        else:
            logging.warning("Refresh button not found on the screen.")
            return False
    except Exception as e:
        logging.error(f"An error occurred while trying to click the refresh button: {e}")
        return False

def capture_window(title='Crystal of Atlan  '):
    """Captures a screenshot of the specified window."""
    try:
        window = gw.getWindowsWithTitle(title)[0]
        if window:
            window.activate()
            with mss.mss() as sct:
                monitor = {
                    "top": window.top,
                    "left": window.left,
                    "width": window.width,
                    "height": window.height,
                }
                img = np.array(sct.grab(monitor))
                
                # Ensure the split-data directory exists
                output_dir = "split-data"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the image with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"capture_{timestamp}.png")
                cv2.imwrite(filename, img)
                logging.info(f"Screenshot saved to {filename}")
                
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            logging.warning(f"Window with title '{title}' not found.")
            return None
    except IndexError:
        logging.warning(f"Window with title '{title}' not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred during window capture: {e}")
        return None
