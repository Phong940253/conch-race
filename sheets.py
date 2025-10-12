import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging
from config import CREDENTIALS_PATH, SHEET_NAME, LIST_CONCH

def save_to_sheet(data, worksheet_name, include_rate=False, prediction=None, check_duplicates=True):
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
                cell_value = conch_info.get('emoji', '')
                if include_rate:
                    rate = conch_info.get('rate', '')
                    cell_value = f"{rate} {cell_value}".strip()
                row_data.append(cell_value)
            else:
                row_data.append("")
        
        if prediction:
            row_data.append("")
            row_data.append(prediction)

        if check_duplicates:
            all_rows = sheet.get_all_values()
            if not all_rows:
                sheet.append_row(header)
                all_rows.append(header)
                
            # ignore top timestamp and top 1st column
            existing_data = [row[1:-1] for row in all_rows[1:]]
            if row_data[1:] not in existing_data:
                sheet.append_row(row_data)
                logging.info(f"Data saved to '{worksheet_name}'.")
            else:
                row_num = existing_data.index(row_data[1:]) + 2
                logging.warning(f"Duplicate data in '{worksheet_name}' at row {row_num}. Not saving.")
        else:
            # If not checking for duplicates, just append the data.
            # Ensure header exists if sheet is empty.
            if not sheet.get_all_values():
                sheet.append_row(header)
            sheet.append_row(row_data)
            logging.info(f"Data saved to '{worksheet_name}'.")
    except Exception as e:
        logging.error(f"Error saving to Google Sheets: {e}")
