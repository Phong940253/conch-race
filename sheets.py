import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging
from config import CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH

def save_to_sheet(data, worksheet_name, include_emoji=False, prediction=None):
    """Saves the OCR data to a Google Sheet."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).worksheet(worksheet_name)
        
        header = ["Timestamp"] + LIST_CONCH
        if prediction:
            header.append("Predicted Winner")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp]
        for name in LIST_CONCH:
            conch_info = data.get(name)
            if conch_info:
                cell_value = f"{conch_info['rate']}"
                if include_emoji:
                    cell_value += f" {conch_info['emoji']}"
                row_data.append(cell_value)
            else:
                row_data.append("")
        
        if prediction:
            row_data.append(prediction)

        all_rows = sheet.get_all_values()
        if not all_rows:
            sheet.append_row(header)
            all_rows.append(header)
            
        existing_data = [row[1:] for row in all_rows[1:]]
        if row_data[1:] not in existing_data:
            sheet.append_row(row_data)
            logging.info(f"Data saved to '{worksheet_name}'.")
        else:
            row_num = existing_data.index(row_data[1:]) + 2
            logging.warning(f"Duplicate data in '{worksheet_name}' at row {row_num}. Not saving.")
    except Exception as e:
        logging.error(f"Error saving to Google Sheets: {e}")
