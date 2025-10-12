import configparser

# --- Configuration Loading ---
config = configparser.ConfigParser()
config.read('config.ini')

# --- Constants ---
# Paths
IMAGE_PATH = config.get('Paths', 'image_path')
OUTPUT_PATH = config.get('Paths', 'output_path')
CREDENTIALS_PATH = config.get('Paths', 'credentials_path')
MODEL_PATH = config.get('Paths', 'model_path')

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
