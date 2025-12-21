# ===========================================
# 🐚 Conch Race - Common Utilities
# ===========================================

import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# --- Google Sheet info ---
SHEET_ID = "1M-cdCYevdk0ZZjbRRSutcN_4M5A3Hta-uQlCfW8wRbo"
SHEET_NAME = "Race Data"
SHEET = "Coa"

# --- Model Definition ---
class ConchPredictor(pl.LightningModule):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# --- Preprocessing function ---
emoji_map = {
    "😎":  2.0,  # confident
    "😁":  1.5,  # happy
    "😡": -1.0,  # angry
    "😢": -1.5,  # sad
    "🖤": -2.0,  # despair
}

def parse_rate_emoji(cell):
    # replace , with .
    cell = str(cell).replace(',', '.')
    if pd.isna(cell) or cell == '':
        return (0.0, 0.0)
    # Extract percentage
    rate_match = re.search(r"([\d\.]+)%", str(cell))
    rate = float(rate_match.group(1)) if rate_match else 0.0
    # Extract emoji
    emo_match = re.findall(r"([\U0001F600-\U0001F64F\U0001F300-\U0001FAFF])", str(cell))
    emoji_val = emoji_map.get(emo_match[0], 0.0) if emo_match else 0.0
    return (rate, emoji_val)
