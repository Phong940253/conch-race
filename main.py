import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import cv2
import logging
import argparse
import matplotlib.pyplot as plt
from config import load_config
from vision import (
    detect_emoji, perform_ocr_on_region, find_best_match, draw_ocr_results, preprocess_for_ocr
)
from model import load_model, predict_winner
from sheets import save_to_sheet
from discord import send_discord_notification
import schedule
import time
from automation import click_refresh_button, capture_window, auto_bet
from utils import run_training

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image_grid(img, reader, debug=False):
    """Processes the grid of regions on the image for OCR and emoji detection."""
    from config import (
        ROWS, COLS, START_X, START_Y, RECT_WIDTH, RECT_HEIGHT, PADDING,
        GRID_COLOR, LIST_CONCH, DICT_EMOJI, EMOJI_THRESHOLD, SCORE_CUTOFF,
        BBOX_COLOR, TEXT_COLOR, NOISE_X1, NOISE_Y1, NOISE_X2, NOISE_Y2
    )
    ocr_data = {}
    conch_regions = {}
    img_height, img_width, _ = img.shape
    first_region_processed = False

    if debug:
        debug_img = img.copy()
        region_count = 0
        for row in range(ROWS):
            for col in range(COLS):
                if region_count >= 6:
                    break
                x = START_X + col * (RECT_WIDTH + PADDING)
                y = START_Y + row * (RECT_HEIGHT + PADDING)
                if y + RECT_HEIGHT <= img_height and x + RECT_WIDTH <= img_width:
                    cv2.rectangle(debug_img, (x, y), (x + RECT_WIDTH, y + RECT_HEIGHT), GRID_COLOR, 2)
                    region_count += 1
            if region_count >= 6:
                break
        
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("First 6 OCR Regions")
        plt.show()

    for row in range(ROWS):
        for col in range(COLS):
            x = START_X + col * (RECT_WIDTH + PADDING)
            y = START_Y + row * (RECT_HEIGHT + PADDING)
            
            if y + RECT_HEIGHT <= img_height and x + RECT_WIDTH <= img_width:
                region_img = img[y:y+RECT_HEIGHT, x:x+RECT_WIDTH]
                
                # Remove noise by drawing a white rectangle over the specified area
                cv2.rectangle(region_img, (NOISE_X1, NOISE_Y1), (NOISE_X2, NOISE_Y2), (255, 255, 255), -1)
                
                emoji = detect_emoji(region_img, DICT_EMOJI, EMOJI_THRESHOLD)
                
                # Preprocess the image for better OCR results
                preprocessed_region = preprocess_for_ocr(region_img)
                
                if debug and not first_region_processed:
                    plt.imshow(preprocessed_region, cmap='gray')
                    plt.title("Preprocessed First Region")
                    plt.show()
                    first_region_processed = True
                
                results = perform_ocr_on_region(reader, preprocessed_region)
                name = find_best_match(results[0][1], LIST_CONCH, SCORE_CUTOFF) if results else None
                
                if len(results) >= 2:
                    rate = results[1][1].replace('..', '.').replace(',', '.') if results[1][1] else '0%'
                    if name:
                        ocr_data[name] = {'rate': rate, 'emoji': emoji}
                        conch_regions[name] = (x, y, RECT_WIDTH, RECT_HEIGHT)
                elif len(results) == 1:
                    if name:
                        ocr_data[name] = {'rate': '0%', 'emoji': emoji}
                        conch_regions[name] = (x, y, RECT_WIDTH, RECT_HEIGHT)
                
                draw_ocr_results(img, results, x, y, BBOX_COLOR, TEXT_COLOR)
                cv2.rectangle(img, (x, y), (x + RECT_WIDTH, y + RECT_HEIGHT), GRID_COLOR, 2)
    
    return ocr_data, conch_regions

def run_ocr_process(debug=False, send_discord=False, model_type='pytorch'):
    """Runs the complete OCR and prediction process on a captured image."""
    from config import (
        OUTPUT_PATH, WORKSHEET_NAME, DATA_WORKSHEET_NAME,
        MODEL_PATH, CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH, WEBHOOK_URL
    )
    
    img = capture_window()
    if img is None:
        logging.error("Failed to capture window for OCR.")
        return None, None, {}, None

    model_path = "conch_race_lightgbm_model.pkl" if model_type == 'lightgbm' else MODEL_PATH
    model, label_encoder, players = load_model(model_path, model_type=model_type)
    reader = easyocr.Reader(['en'])

    ocr_data, conch_regions = process_image_grid(img, reader, debug=debug)
    # logging.info(f"OCR Data: {ocr_data}")
    
    prediction = None
    probabilities = None
    if model:
        prediction, probabilities = predict_winner(model, label_encoder, players, ocr_data, model_type=model_type)
        logging.info(f"Predicted Winner: {prediction}")
    
    duplicate_row = None
    if debug:
        logging.info("Debug mode is enabled. Skipping save to Google Sheets.")
    elif ocr_data:
        duplicate_row = save_to_sheet(ocr_data, WORKSHEET_NAME, CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH, include_rate=False, check_duplicates=True)
        save_to_sheet(ocr_data, DATA_WORKSHEET_NAME, CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH, include_rate=True, prediction=prediction, check_duplicates=False)
        
    if ocr_data and send_discord:
        send_discord_notification(ocr_data, prediction, probabilities, label_encoder, WEBHOOK_URL, debug=debug, duplicate_row=duplicate_row)

    cv2.imwrite(OUTPUT_PATH, img)
    logging.info(f"Processed image saved to {OUTPUT_PATH}")
    
    return prediction, probabilities, conch_regions, label_encoder

def scheduled_ocr_task(debug=False, send_discord=False):
    """Task for scheduled OCR runs, including clicking refresh."""
    logging.info("Running scheduled OCR task...")
    if click_refresh_button():
        time.sleep(5)
        prediction, probabilities, conch_regions, label_encoder = run_ocr_process(debug, send_discord)
        
        if not (prediction and conch_regions):
            logging.warning("No prediction or conch regions detected. Skipping auto-bet.")
            return

        if prediction in conch_regions:
            auto_bet(prediction, conch_regions)
        else:
            logging.warning(f"Predicted winner '{prediction}' is not in the current race.")
            
            if probabilities is not None and label_encoder is not None:
                # Create a dictionary of conch names to their probabilities
                prob_dict = {conch: prob for conch, prob in zip(label_encoder.classes_, probabilities[0])}
                
                # Filter for participants present in the current race
                available_conches = {conch: prob_dict.get(conch, 0) for conch in conch_regions.keys()}
                
                if available_conches:
                    # Find the best alternative participant
                    best_alternative = max(available_conches, key=available_conches.get)
                    logging.info(f"Betting on the best alternative: '{best_alternative}'")
                    auto_bet(best_alternative, conch_regions)
                else:
                    logging.error("No available conches found to place a bet on.")
            else:
                logging.error("Probabilities or label encoder not available to determine an alternative bet.")

def main():
    """Main function to run the OCR process."""
    parser = argparse.ArgumentParser(description="Conch Race OCR and Prediction")
    parser.add_argument("-c", "--config", type=str, default="config.ini", help="Path to the configuration file.")
    parser.add_argument("-i", "--image", type=str, help="Path to the image file to process.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode to visualize the first preprocessed image and skip saving to sheets.")
    parser.add_argument("-s", "--send-discord", action="store_true", help="Send a notification to Discord.")
    parser.add_argument("-dup", "--duplicate-check", action="store_true", help="Enable duplicate checking when saving to Google Sheets.")
    parser.add_argument("--schedule", action="store_true", help="Run in schedule mode.")
    parser.add_argument("--model-type", type=str, default="pytorch", choices=['pytorch', 'lightgbm'], help="Specify the model type to use.")
    args = parser.parse_args()

    load_config(args.config)
    

    if args.schedule:
        # Configure file logging for schedule mode
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file = 'conch-race.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_handler)
        
        logging.info("Running in schedule mode.")
        
        # Schedule OCR tasks
        for hour in [11, 18]:
            for minute in [4, 19, 39, 59]:
                schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(scheduled_ocr_task, debug=args.debug, send_discord=args.send_discord)
        for hour in [12, 19]:
            for minute in [19, 39]:
                schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(scheduled_ocr_task, debug=args.debug, send_discord=args.send_discord)

        # Schedule training tasks
        # for hour in [11, 18]:
        #     for minute in [2, 17, 37, 57]:
        #         schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(run_training)
        # for hour in [12, 19]:
        #     for minute in [17, 37]:
        #         schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(run_training)
        
        while True:
            schedule.run_pending()
            time.sleep(1)
    else:
        # Now that the config is loaded, we can import the variables
        from config import (
            IMAGE_PATH, OUTPUT_PATH, WORKSHEET_NAME, DATA_WORKSHEET_NAME,
            MODEL_PATH, DICT_EMOJI, EMOJI_THRESHOLD, SCORE_CUTOFF, BBOX_COLOR, TEXT_COLOR,
            CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH, WEBHOOK_URL
        )

        image_path = args.image if args.image else IMAGE_PATH

        model_path = "conch_race_lightgbm_model.pkl" if args.model_type == 'lightgbm' else MODEL_PATH
        model, label_encoder, players = load_model(model_path, model_type=args.model_type)
        reader = easyocr.Reader(['en'])
        
        if args.image:
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Could not read image from {image_path}")
                return
        else:
            img = capture_window()
            if img is None:
                logging.error("Could not capture the window.")
                return

        ocr_data, conch_regions = process_image_grid(img, reader, debug=args.debug)
        
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        prediction = None
        probabilities = None
        if model:
            prediction, probabilities = predict_winner(model, label_encoder, players, ocr_data, model_type=args.model_type)
            logging.info(f"Predicted Winner: {prediction}")
        
        duplicate_row = None
        if args.debug and not args.duplicate_check:
            logging.info("Debug mode is enabled. Skipping save to Google Sheets.")
        elif ocr_data:
            # Save to the sheet with emojis only, with duplicate checking
            duplicate_row = save_to_sheet(ocr_data, WORKSHEET_NAME, CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH, include_rate=False, check_duplicates=True)
            
            # Save to the data sheet with rates, emojis, and prediction, without duplicate checking
            if not args.duplicate_check:
                save_to_sheet(ocr_data, DATA_WORKSHEET_NAME, CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH, include_rate=True, prediction=prediction, check_duplicates=False)
            
        if ocr_data and args.send_discord:
            send_discord_notification(ocr_data, prediction, probabilities, label_encoder, WEBHOOK_URL, debug=args.debug, duplicate_row=duplicate_row)

        cv2.imwrite(OUTPUT_PATH, img)
        logging.info(f"Processed image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
