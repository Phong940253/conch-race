import cv2
import os
from fuzzywuzzy import process
import pyautogui
from PIL import Image

def detect_emoji(image, dict_emoji, emoji_threshold):
    """Detects an emoji in the image using pyautogui."""
    # Convert the OpenCV image (NumPy array) to a Pillow image
    # PyAutoGUI's image recognition works with Pillow images
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for emoji_info in dict_emoji.values():
        try:
            # Use pyautogui.locate to find the emoji in the given image region
            if pyautogui.locate(emoji_info['path'], pil_image, confidence=emoji_threshold):
                return emoji_info['icon']
        except pyautogui.PyAutoGUIException as e:
            # This can happen if the template image is not found, etc.
            # print(f"Could not process emoji {emoji_info['path']}: {e}")
            continue
            
    # Return a default emoji if no match is found
    return dict_emoji['sad']['icon']

def find_best_match(text, choices, score_cutoff):
    """Finds the best match for a text from a list of choices."""
    match = process.extractOne(text, choices)
    return match[0] if match and match[1] >= score_cutoff else None

def perform_ocr_on_region(reader, image):
    """Performs OCR on a cropped region of the image."""
    return reader.readtext(image)

def draw_ocr_results(image, results, x_offset, y_offset, bbox_color, text_color):
    """Draws OCR results on the image."""
    for detection in results:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]
        abs_top_left = (top_left[0] + x_offset, top_left[1] + y_offset)
        cv2.rectangle(image, abs_top_left, (bottom_right[0] + x_offset, bottom_right[1] + y_offset), bbox_color, 2)
        cv2.putText(image, text, abs_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

def preprocess_for_ocr(image):
    """Applies preprocessing steps to an image to improve OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Only apply light enhancement
    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    return enhanced
