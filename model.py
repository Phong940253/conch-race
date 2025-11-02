import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import logging
import joblib
import numpy as np
from common import ConchPredictor, parse_rate_emoji

def load_model(model_path, model_type='pytorch'):
    """Loads the prediction model and its components."""
    try:
        if model_type == 'pytorch':
            checkpoint = torch.load(model_path, weights_only=False)
            model = ConchPredictor(checkpoint["input_dim"], checkpoint["num_classes"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            label_encoder = LabelEncoder()
            label_encoder.classes_ = checkpoint["label_encoder"]
            players = checkpoint["players"]
        elif model_type == 'lightgbm':
            data = joblib.load(model_path)
            model = data["model"]
            label_encoder = data["label_encoder"]
            players = data["players"]
        else:
            raise ValueError("Unsupported model type")
            
        logging.info(f"Model '{model_path}' ({model_type}) loaded successfully.")
        return model, label_encoder, players
    except FileNotFoundError:
        logging.error(f"Model file not found at '{model_path}'.")
        return None, None, None

def predict_winner(model, label_encoder, players, data, model_type='pytorch'):
    """Predicts the winner and probabilities based on the OCR data."""
    features = []
    logging.debug(f"Predicting with data: {data}")
    logging.debug(f"Using players: {players}")
    logging.debug(f"Label encoder classes: {label_encoder.classes_}")
    for p in players:
        conch_info = data.get(p)
        if conch_info:
            rate, emo = parse_rate_emoji(f"{conch_info['rate']}% {conch_info['emoji']}")
            logging.debug(f"Player: {p}, Rate: {rate}, Emoji: {emo}")
            features.extend([rate, emo])
        else:
            features.extend([0.0, 0.0])
    logging.debug(f"Extracted features: {features}")    
    
    if model_type == 'pytorch':
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
        winner = label_encoder.inverse_transform([pred])[0]
        return winner, probabilities
    elif model_type == 'lightgbm':
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        probabilities = model.predict(x)
        pred_index = np.argmax(probabilities, axis=1)[0]
        winner = label_encoder.inverse_transform([pred_index])[0]
        return winner, probabilities
    else:
        raise ValueError("Unsupported model type")
