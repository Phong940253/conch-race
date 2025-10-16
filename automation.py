import logging
import pyautogui
import pygetwindow as gw
import mss
import numpy as np
import cv2

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
                # save the image for debugging
                cv2.imwrite("debug_capture.png", img)
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
