import easyocr
import cv2
import logging
import argparse
from config import (
    IMAGE_PATH, OUTPUT_PATH, WORKSHEET_NAME, DATA_WORKSHEET_NAME,
    ROWS, COLS, START_X, START_Y, RECT_WIDTH, RECT_HEIGHT, PADDING,
    GRID_COLOR, LIST_CONCH
)
from vision import (
    detect_emoji, perform_ocr_on_region, find_best_match, draw_ocr_results, preprocess_for_ocr
)
from model import load_model, predict_winner
from sheets import save_to_sheet

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                
                results = perform_ocr_on_region(reader, preprocessed_region)
                
                if len(results) >= 2:
                    name = find_best_match(results[0][1], LIST_CONCH)
                    if name:
                        ocr_data[name] = {'rate': results[1][1], 'emoji': emoji}
                
                draw_ocr_results(img, results, x, y)
                cv2.rectangle(img, (x, y), (x + RECT_WIDTH, y + RECT_HEIGHT), GRID_COLOR, 2)
    
    return ocr_data

def main():
    """Main function to run the OCR process."""
    parser = argparse.ArgumentParser(description="Conch Race OCR and Prediction")
    parser.add_argument("--image", type=str, help="Path to the image file to process.")
    args = parser.parse_args()

    image_path = args.image if args.image else IMAGE_PATH

    model, label_encoder, players = load_model()
    reader = easyocr.Reader(['en'])
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image from {image_path}")
        return

    ocr_data = process_image_grid(img, reader)
    logging.info(f"OCR Data: {ocr_data}")
    
    if ocr_data:
        # Save to the sheet with no emojis
        save_to_sheet(ocr_data, WORKSHEET_NAME, include_emoji=False)
        
        # Save to the data sheet with emojis and prediction
        prediction = None
        if model:
            prediction = predict_winner(model, label_encoder, players, ocr_data)
            logging.info(f"Predicted Winner: {prediction}")
        save_to_sheet(ocr_data, DATA_WORKSHEET_NAME, include_emoji=True, prediction=prediction)

    cv2.imwrite(OUTPUT_PATH, img)
    logging.info(f"Processed image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
