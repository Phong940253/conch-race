import configparser

# --- Global variables to be populated by load_config ---
IMAGE_PATH = None
OUTPUT_PATH = None
CREDENTIALS_PATH = None
MODEL_PATH = None
START_X = None
START_Y = None
RECT_WIDTH = None
RECT_HEIGHT = None
PADDING = None
ROWS = None
COLS = None
SHEET_NAME = None
WORKSHEET_NAME = None
DATA_WORKSHEET_NAME = None
SCORE_CUTOFF = None
EMOJI_THRESHOLD = None
WEBHOOK_URL = None
BBOX_COLOR = None
TEXT_COLOR = None
GRID_COLOR = None
LIST_CONCH = None
DICT_EMOJI = None
NOISE_X1 = None
NOISE_Y1 = None
NOISE_X2 = None
NOISE_Y2 = None

def load_config(config_file='config.ini'):
    """Loads configuration from the specified file and populates global variables."""
    global IMAGE_PATH, OUTPUT_PATH, CREDENTIALS_PATH, MODEL_PATH, START_X, START_Y, \
           RECT_WIDTH, RECT_HEIGHT, PADDING, ROWS, COLS, SHEET_NAME, WORKSHEET_NAME, \
           DATA_WORKSHEET_NAME, SCORE_CUTOFF, EMOJI_THRESHOLD, WEBHOOK_URL, \
           BBOX_COLOR, TEXT_COLOR, GRID_COLOR, LIST_CONCH, DICT_EMOJI, \
           NOISE_X1, NOISE_Y1, NOISE_X2, NOISE_Y2

    config = configparser.ConfigParser()
    config.read(config_file)

    # --- Assign values from config file ---
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

    # Discord
    WEBHOOK_URL = config.get('Discord', 'webhook_url')

    # Noise Removal
    NOISE_X1 = config.getint('NoiseRemoval', 'x1')
    NOISE_Y1 = config.getint('NoiseRemoval', 'y1')
    NOISE_X2 = config.getint('NoiseRemoval', 'x2')
    NOISE_Y2 = config.getint('NoiseRemoval', 'y2')

    # Drawing Colors (BGR format)
    BBOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 0, 0)  # Blue
    GRID_COLOR = (0, 0, 255)  # Red

    LIST_CONCH = [
        "Karl, the Fatebringer", "Fiery Conch Warrior", "B.Erserker", "Captain Blackhat",
        "Galloping Tractor", "Gold Miner", "Conchie", "Crazy Conch", "Poseidonn", "Deja Vu"
    ]

    # Dynamically build DICT_EMOJI from config
    DICT_EMOJI = {
        "sad": {"icon": "😢", "path": config.get('EmojiPaths', 'sad')},
        "happy": {"icon": "😁", "path": config.get('EmojiPaths', 'happy')},
        "angry": {"icon": "😡", "path": config.get('EmojiPaths', 'angry')},
        "cool": {"icon": "😎", "path": config.get('EmojiPaths', 'cool')},
        "black": {"icon": "🖤", "path": config.get('EmojiPaths', 'black')}
    }
