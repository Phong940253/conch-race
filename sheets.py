import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging
import traceback

def calculate_match_score(current_data, existing_row):
    """
    current_data: list emoji hiện tại (bỏ timestamp)
    existing_row: list emoji trong sheet (bỏ timestamp)
    """
    score = 0
    for cur, exist in zip(current_data, existing_row):
        if cur and exist and cur == exist:
            score += 1
    return score

def find_first_empty_row_in_table(sheet, start_row=2, key_col=1):
    """
    start_row: dòng bắt đầu dữ liệu (sau header)
    key_col: cột bắt buộc có dữ liệu (ví dụ Timestamp = col 1)
    """
    col_values = sheet.col_values(key_col)
    return len(col_values) + 1


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
            best_score = 0
            best_matches = []

            for i, row in enumerate(all_rows[1:]):  # bỏ header
                existing_data = row[1:len(current_data_to_check) + 1]

                score = calculate_match_score(current_data_to_check, existing_data)

                if score > best_score:
                    best_score = score
                    best_matches = [{
                        "row_number": i + 2,  # +1 header, +1 index
                        "score": score,
                        "row_data": row,
                    }]
                elif score == best_score and score > 0:
                    best_matches.append({
                        "row_number": i + 2,
                        "score": score,
                        "row_data": row,
                    })
            
            
            # If no perfect match is found, append the new row
            if best_score < len(list_conch):
                row_index = find_first_empty_row_in_table(sheet)

                sheet.insert_row(
                    row_data,
                    row_index,
                    value_input_option="USER_ENTERED"
                )
                logging.info(f"Data saved to '{worksheet_name}'.")
                    
            if best_score > 0:
                logging.warning(
                    f"Found {len(best_matches)} best match(es) with score={best_score}"
                )

                for m in best_matches:
                    logging.info(
                        f"Match row {m['row_number']} | score={m['score']} | data={m['row_data']}"
                    )

                # ❌ Không append dòng mới
                # ✅ Trả về danh sách dòng trùng
                return best_matches

        else:
            # If not checking for duplicates, just append the data.
            # Ensure header exists if sheet is empty.
            if not sheet.get_all_values():
                sheet.append_row(header)
                
            row_index = find_first_empty_row_in_table(sheet)
            sheet.insert_row(
                row_data,
                row_index,
                value_input_option="USER_ENTERED"
            )
            logging.info(f"Data saved to '{worksheet_name}'.")
        return None
    except Exception as e:
        logging.error(traceback.format_exc())
        return None
