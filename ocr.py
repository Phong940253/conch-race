import easyocr
import cv2
from matplotlib import pyplot as plt
import sys
from fuzzywuzzy import process
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
import logging
import configparser
import argparse

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
config = configparser.ConfigParser()
config.read('config.ini')

# --- Constants ---
# Paths
IMAGE_PATH = config.get('Paths', 'image_path')
OUTPUT_PATH = config.get('Paths', 'output_path')
CREDENTIALS_PATH = config.get('Paths', 'credentials_path')

# OCR Grid
START_X = config.getint('OCRGrid', 'start_x')
START_Y = config.getint('OCRGrid', 'start_y')
RECT_WIDTH = config.getint('OCRGrid', 'rect_width')
RECT_HEIGHT = config.getint('OCRGrid', 'rect_height')
PADDING = config.getint('OCRGrid', 'padding')
ROWS = config.getint('OCRGrid', 'rows')
COLS = config.getint('OCRGrid', 'cols')

# Google Sheets
SHEET_NAME = config.get('GoogleSheets', 'sheet_name')
WORKSHEET_NAME = config.get('GoogleSheets', 'worksheet_name')
DATA_WORKSHEET_NAME = config.get('GoogleSheets', 'data_worksheet_name')

# Settings
SCORE_CUTOFF = config.getint('Settings', 'score_cutoff')
EMOJI_THRESHOLD = config.getfloat('Settings', 'emoji_threshold')

# Drawing Colors (BGR format)
BBOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 0, 0)  # Blue
GRID_COLOR = (0, 0, 255)  # Red

LIST_CONCH = [
    "Karl, the Fatebringer", "Fiery Conch Warrior", "B.Erserker", "Captain Blackhat",
    "Galloping Tractor", "Gold Miner", "Conchie", "Crazy Conch", "Poseidon", "Deja Vu"
]

DICT_EMOJI = {
    "sad": {"icon": "😢", "path": "./sad.png"},
    "happy": {"icon": "😁", "path": "./happy.png"},
    "angry": {"icon": "😡", "path": "./angry.png"},
    "cool": {"icon": "😎", "path": "./cool.png"},
    "black": {"icon": "🖤", "path": "./black.png"}
}

# --- Core Functions ---

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

def save_to_sheet(data):
    """Saves the OCR data to a Google Sheet."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)
        data_sheet = client.open(SHEET_NAME).worksheet(DATA_WORKSHEET_NAME)
        
        header = ["Timestamp"] + LIST_CONCH
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp] + [f"{d['emoji']}" if (d := data.get(name)) else "" for name in LIST_CONCH]

        all_rows = sheet.get_all_values()
        if not all_rows:
            sheet.append_row(header)
            all_rows.append(header)
            
        existing_data = [row[1:-1] for row in all_rows[1:]]
        if row_data[1:] not in existing_data:
            sheet.append_row(row_data)
            logging.info("Data saved to Google Sheet.")
        else:
            row_num = existing_data.index(row_data[1:]) + 2
            logging.warning(f"Duplicate data found at row {row_num}. Not saving.")
        data_sheet.append_row([timestamp] + [f"{d['rate']} {d['emoji']}" if (d := data.get(name)) else f"" for name in LIST_CONCH])
        logging.info("Detailed data saved to data worksheet.")
    except Exception as e:
        logging.error(f"Error saving to Google Sheets: {e}")

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

def process_image_grid(img, reader):
    """Processes the grid of regions on the image for OCR and emoji detection."""
    ocr_data = {}
    img_height, img_width, _ = img.shape

    for row in range(ROWS):
        for col in range(COLS):
            x = START_X + col * (RECT_WIDTH + PADDING)
            y = START_Y + row * (RECT_HEIGHT + PADDING)
            
            if y + RECT_HEIGHT <= img_height and x + RECT_WIDTH <= img_width:
                region_img = img[y:y+RECT_HEIGHT, x:x+RECT_WIDTH]
                emoji = detect_emoji(region_img)
                
                # Preprocess the image for better OCR results
                preprocessed_region = preprocess_for_ocr(region_img)
                
                # debug: show preprocessed image
                # plt.imshow(preprocessed_region, cmap='gray')
                # plt.show()
                
                results = perform_ocr_on_region(reader, preprocessed_region)
                
                if len(results) >= 2:
                    name = find_best_match(results[0][1], LIST_CONCH)
                    if name:
                        ocr_data[name] = {'rate': results[1][1], 'emoji': emoji}
                elif len(results) == 1:
                    name = find_best_match(results[0][1], LIST_CONCH)
                    if name:
                        ocr_data[name] = {'rate': '0%', 'emoji': emoji}
                
                draw_ocr_results(img, results, x, y)
                cv2.rectangle(img, (x, y), (x + RECT_WIDTH, y + RECT_HEIGHT), GRID_COLOR, 2)
    
    return ocr_data

def main(image_path=None):
    """Main function to run the OCR process."""
    reader = easyocr.Reader(['en'])
    
    img_path = image_path if image_path else IMAGE_PATH
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Could not read image from {img_path}")
        return

    ocr_data = process_image_grid(img, reader)
    logging.info(f"OCR Data: {ocr_data}")
    
    if ocr_data:
        save_to_sheet(ocr_data)

    cv2.imwrite(OUTPUT_PATH, img)
    logging.info(f"Processed image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image to extract conch race data.")
    parser.add_argument('image_path', nargs='?', default=None, 
                        help='The path to the image file. If not provided, the path from config.ini will be used.')
    args = parser.parse_args()
    main(image_path=args.image_path)
