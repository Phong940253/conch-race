import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging

def save_to_sheet(data, worksheet_name, credentials_path, sheet_name, list_conch, include_rate=False, prediction=None, check_duplicates=True):
    """Saves the OCR data to a Google Sheet."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open(sheet_name).worksheet(worksheet_name)
        
        header = ["Timestamp"] + list_conch
        if prediction:
            header.append("Predicted Winner")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp]
        for name in list_conch:
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
                
            # The data to check for duplicates (emojis, ignoring timestamp)
            current_data_to_check = row_data[1:]

            # Iterate through existing rows to check for duplicates
            for i, row in enumerate(all_rows):  # Skip header row
                # Existing data to compare (emojis)
                existing_data_to_check = row[1:len(current_data_to_check) + 1]
                
                if current_data_to_check == existing_data_to_check:
                    logging.info(current_data_to_check)
                    # A duplicate is found
                    row_num = i + 2  # +1 for 1-based index, +1 for skipped header
                    winner_name = row[-1] if len(row) > len(current_data_to_check) + 1 else "Unknown"
                    logging.warning(f"Duplicate data in '{worksheet_name}' at row {row_num}. Winner: {winner_name}. Not saving.")
                    return winner_name

            # If no duplicate is found, append the new row
            sheet.append_row(row_data)
            logging.info(f"Data saved to '{worksheet_name}'.")
            return None
        else:
            # If not checking for duplicates, just append the data.
            # Ensure header exists if sheet is empty.
            if not sheet.get_all_values():
                sheet.append_row(header)
            sheet.append_row(row_data)
            logging.info(f"Data saved to '{worksheet_name}'.")
        return None
    except Exception as e:
        logging.error(f"Error saving to Google Sheets: {e}")
        return None
