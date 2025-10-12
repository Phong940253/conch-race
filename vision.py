import cv2
import os
from fuzzywuzzy import process
from config import (
    EMOJI_THRESHOLD, DICT_EMOJI, SCORE_CUTOFF, BBOX_COLOR, TEXT_COLOR
)

def detect_emoji(image):
    """Detects an emoji in the image using template matching."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_match = {'icon': DICT_EMOJI['sad']['icon'], 'score': EMOJI_THRESHOLD}

    for emoji_info in DICT_EMOJI.values():
        if not os.path.exists(emoji_info['path']):
            continue
        template = cv2.imread(emoji_info['path'], 0)
        if template is None:
            continue
        
        h, w = template.shape
        if h > gray_image.shape[0] or w > gray_image.shape[1]:
            continue

        res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_match['score']:
            best_match = {'icon': emoji_info['icon'], 'score': max_val}
            
    return best_match['icon']

def find_best_match(text, choices):
    """Finds the best match for a text from a list of choices."""
    match = process.extractOne(text, choices)
    return match[0] if match and match[1] >= SCORE_CUTOFF else None

def perform_ocr_on_region(reader, image):
    """Performs OCR on a cropped region of the image."""
    return reader.readtext(image)

def draw_ocr_results(image, results, x_offset, y_offset):
    """Draws OCR results on the image."""
    for detection in results:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]
        abs_top_left = (top_left[0] + x_offset, top_left[1] + y_offset)
        cv2.rectangle(image, abs_top_left, (bottom_right[0] + x_offset, bottom_right[1] + y_offset), BBOX_COLOR, 2)
        cv2.putText(image, text, abs_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

def preprocess_for_ocr(image):
    """Applies preprocessing steps to an image to improve OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Only apply light enhancement
    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    return enhanced