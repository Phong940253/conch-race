import cv2
import os
from fuzzywuzzy import process
import pyautogui
from PIL import Image
from typing import List, Sequence, Tuple

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
    raw = reader.readtext(image)
    return _merge_adjacent_ocr_boxes(raw)


def _merge_adjacent_ocr_boxes(
    results: Sequence[Tuple[Sequence[Sequence[float]], str, float]],
    *,
    max_x_gap_px: int = 18,
    max_y_center_gap_px: int = 18,
    min_vertical_overlap_ratio: float = 0.55,
) -> List[Tuple[List[List[int]], str, float]]:
    """Merge OCR boxes that belong to the same line and are close horizontally.

    EasyOCR often splits a single name into 2 adjacent detections (e.g. two boxes
    sitting next to each other). This merges those back into one detection so
    downstream logic can treat it as a single string.
    """

    def _rect(bbox: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        return x1, y1, x2, y2

    def _bbox_points(x1: int, y1: int, x2: int, y2: int) -> List[List[int]]:
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _vertical_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ay1, ay2 = a[1], a[3]
        by1, by2 = b[1], b[3]
        overlap = max(0, min(ay2, by2) - max(ay1, by1))
        denom = max(1, min(ay2 - ay1, by2 - by1))
        return overlap / denom

    items = []
    for bbox, text, conf in results:
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        r = _rect(bbox)
        y_center = (r[1] + r[3]) // 2
        items.append({
            "bbox": r,
            "y_center": y_center,
            "text": text,
            "conf": float(conf) if conf is not None else 0.0,
        })

    if not items:
        return []

    # Sort top-to-bottom, left-to-right for stable merging and predictable order.
    items.sort(key=lambda d: (d["y_center"], d["bbox"][0]))

    merged: List[dict] = []

    for cur in items:
        if not merged:
            merged.append(cur)
            continue

        prev = merged[-1]
        prev_bbox = prev["bbox"]
        cur_bbox = cur["bbox"]

        y_center_gap = abs(cur["y_center"] - prev["y_center"])
        x_gap = cur_bbox[0] - prev_bbox[2]
        v_overlap = _vertical_overlap(prev_bbox, cur_bbox)

        same_line = y_center_gap <= max_y_center_gap_px and v_overlap >= min_vertical_overlap_ratio
        close_horiz = 0 <= x_gap <= max_x_gap_px

        if same_line and close_horiz:
            joiner = "" if x_gap <= 3 else " "
            prev["text"] = (prev["text"] + joiner + cur["text"]).strip()
            prev["conf"] = min(prev["conf"], cur["conf"])
            prev["bbox"] = (
                min(prev_bbox[0], cur_bbox[0]),
                min(prev_bbox[1], cur_bbox[1]),
                max(prev_bbox[2], cur_bbox[2]),
                max(prev_bbox[3], cur_bbox[3]),
            )
            prev["y_center"] = (prev["bbox"][1] + prev["bbox"][3]) // 2
        else:
            merged.append(cur)

    out: List[Tuple[List[List[int]], str, float]] = []
    for m in merged:
        x1, y1, x2, y2 = m["bbox"]
        out.append((_bbox_points(x1, y1, x2, y2), m["text"], m["conf"]))
    return out

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
