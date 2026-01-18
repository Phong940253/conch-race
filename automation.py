import logging
import pyautogui
import pygetwindow as gw
import mss
import win32api
import win32con
import numpy as np
import cv2
import os
from datetime import datetime
import time
import traceback


logger = logging.getLogger(__name__)

def activate_window(title='Crystal of Atlan  '):
    """Activates the specified window."""
    try:
        window = gw.getWindowsWithTitle(title)[0]
        if window:
            window.activate()
            return True
        else:
            logger.warning("Window with title '%s' not found.", title)
            return False
    except IndexError:
        logger.warning("Window with title '%s' not found.", title)
        return False
    except Exception:
        logger.exception("Failed to activate window '%s'", title)
        return False

def click(x, y):
    """Moves the mouse to the specified coordinates and performs a left-click."""
    try:
        win32api.SetCursorPos((x, y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    except Exception:
        logger.exception("Failed to click at (%s, %s)", x, y)

def click_image(image_path, confidence=0.7, region=None, sleep_time=2):
    """Finds and clicks the center of an image on the screen."""
    try:
        if not activate_window():
            return False
        location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence, region=region)
        if location:
            click(location.x, location.y)
            logger.debug("Clicked image '%s'", image_path)
            time.sleep(sleep_time)  # Wait for a short duration after clicking
            return True
        else:
            logger.debug("Image '%s' not found on screen.", image_path)
            return False
    except Exception:
        logger.exception("click_image failed for '%s'", image_path)
        return False

def auto_bet(predicted_winner, conch_regions):
    """Automates the betting process based on the predicted winner."""
    if not predicted_winner or not conch_regions:
        logger.error("Auto-betting skipped: No prediction or region data.")
        return

    # logging.info(f"Starting auto-bet for predicted winner: {predicted_winner}")

    # 1. Click the support button
    if not click_image('support.png'):
        logger.error("Could not find support button.")
        return

    # 2. Click the increase button for the predicted winner
    winner_region = conch_regions.get(predicted_winner)
    logger.debug("Winner region: %s", winner_region)
    logger.debug("Predicted winner: %s", predicted_winner)
    logger.debug("Conch regions: %s", conch_regions)
    if not winner_region:
        logger.error("Could not find region for predicted winner: %s", predicted_winner)
        return
    
    # click button increase 3 times
    for _ in range(1):
        if not click_image('increase.png', region=winner_region, sleep_time=0.5):
            logger.error("Could not find increase button for %s", predicted_winner)
            return

    # 3. Click the first confirm button
    if not click_image('confirm1.png'):
        logger.error("Could not find first confirm button.")
        return

    # 4. Click the second confirm button
    if not click_image('confirm2.png'):
        logger.error("Could not find second confirm button.")
        return
    
    logger.info("Auto-bet completed for %s", predicted_winner)

def click_refresh_button():
    """Finds and clicks the refresh button on the screen."""
    try:
        if not activate_window():
            return False
        refresh_button_location = pyautogui.locateCenterOnScreen('refresh.png', confidence=0.8)
        if refresh_button_location:
            click(refresh_button_location.x, refresh_button_location.y)
            logger.info("Clicked refresh button")
            return True
        else:
            logger.warning("Refresh button not found on the screen.")
            return False
    except Exception:
        logger.exception("Failed to click refresh button")
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
                logger.debug("Screenshot saved to %s", filename)
                
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            logger.warning("Window with title '%s' not found.", title)
            return None
    except IndexError:
        logger.warning("Window with title '%s' not found.", title)
        return None
    except Exception:
        logger.exception("Window capture failed")
        return None
