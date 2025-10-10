# Conch Race OCR

This project uses Optical Character Recognition (OCR) to extract data from screenshots of a "Conch Race" game. It identifies the names of the conches, their associated rates, and detects an emoji for each participant. The extracted data is then saved to a Google Sheet for tracking and analysis.

## Features

- **OCR Data Extraction**: Reads conch names and rates from a predefined grid in an image.
- **Emoji Detection**: Uses template matching to identify emojis associated with each conch.
- **Google Sheets Integration**: Saves the extracted race data to a specified Google Sheet, including a timestamp.
- **Configurable**: All major settings, including paths, OCR grid dimensions, and Google Sheets details, can be configured in the `config.ini` file.
- **Command-Line Interface**: The script can be run from the command line, with an option to provide the path to the image file directly.

## Setup

### 1. Prerequisites

- Python 3.12
- A Google Cloud Platform project with the Google Sheets API enabled.

### 2. Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository-url>
    cd conch-race
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

1.  **Google Sheets API Credentials:**
    - Follow the [Google Cloud documentation](https://developers.google.com/sheets/api/quickstart/python) to create a service account and download the JSON credentials file.
    - Rename the downloaded file to `credentials.json` and place it in the `conch-race` directory.
    - Share your target Google Sheet with the `client_email` found in your `credentials.json` file.

2.  **Configuration File (`config.ini`):**
    - Open `config.ini` and customize the settings as needed.

    ```ini
    [Paths]
    image_path = test.png          ; Default image to process
    output_path = test_ocr.png     ; Where to save the processed image with OCR boxes
    credentials_path = credentials.json ; Path to your Google API credentials

    [OCRGrid]
    ; Coordinates and dimensions for the OCR grid
    start_x = 516
    start_y = 218
    rect_width = 286
    rect_height = 193
    padding = 14
    rows = 2
    cols = 3

    [GoogleSheets]
    sheet_name = Coa               ; The name of your Google Sheet file
    worksheet_name = Race          ; The name of the worksheet for summary data
    data_worksheet_name = Race Data ; The name of the worksheet for detailed data

    [Settings]
    score_cutoff = 70              ; Confidence threshold for fuzzy name matching (0-100)
    emoji_threshold = 0.8          ; Confidence threshold for emoji detection (0.0-1.0)
    ```

## Usage

To run the script, execute `ocr.py` from your terminal.

**Process the default image specified in `config.ini`:**

```bash
python ocr.py
```

**Process a specific image by providing its path as an argument:**

```bash
python ocr.py /path/to/your/image.png
```

After execution, the script will:
1.  Read the data from the image.
2.  Log the extracted data to the console.
3.  Save a new image named `test_ocr.png` (or as configured) with the OCR detection boxes drawn on it.
4.  Append the new data to the configured Google Sheet.
