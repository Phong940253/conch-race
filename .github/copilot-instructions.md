# Copilot instructions (conch-race)

## Big picture

- This is a **Windows desktop automation bot** for the “Conch Race” UI:
  1. capture the game window → 2) OCR a fixed grid of regions → 3) detect an emoji via template matching →
  2. predict a winner via a **LightGBM LambdaRank** model → 5) write to Google Sheets → 6) notify Discord → 7) optional auto-bet.
- The “production” entrypoint is `main.py`; training/prediction notebooks/scripts exist but are secondary.

## Key files & responsibilities

- `main.py`: CLI + schedule loop; orchestrates capture/OCR/predict/save/notify; installs crash hooks.
- `config.py` + `config.ini`: configuration is loaded into **module-level globals**. Call `load_config(...)` before importing/using config values.
- `automation.py`: UI automation (activate window, screenshot via `mss`, click images via `pyautogui`, Win32 mouse click).
- `vision.py`: OCR + fuzzy name matching + emoji detection (templates) + OCR drawing/preprocessing.
- `model.py`: loads `conch_race_ranker.pkl` and computes ranking features; returns `(winner, ranking)`.
- `sheets.py`: Google Sheets write + duplicate detection (emoji-only row match scoring).
- `discord.py`: Discord embeds for race results + a `DiscordWebhookLogHandler` that forwards logs to the panic webhook.

## How to run (local)

- Install: `pip install -r requirements.txt` (Python 3.12; Windows-only deps like `pywin32`).
- One-shot OCR:
  - `python main.py --config config.ini`
  - From file: `python main.py --config config.ini --image path/to.png`
- Schedule mode: `python main.py --config config.ini --schedule` (writes `conch-race.log`).
- Run scheduled task once: `python main.py --config config.ini --now`
- Discord notifications: add `--send-discord`.
- Debug mode: `--debug` shows first region preprocessing and avoids writing sheets unless `--duplicate-check` is used.
- Crash/panic verification:
  - `python main.py --config config.ini --test-panic`
  - `python main.py --config config.ini --test-crash`

## Project-specific conventions (important)

- **Config globals**: `config.py` populates globals (e.g. `START_X`, `ROWS`, `WEBHOOK_URL`, `PANIC_WEBHOOK_URL`). Don’t refactor these into dataclasses without updating all imports.
- **Window title matters**: automation targets `Crystal of Atlan  ` (note the trailing spaces). If capture/clicking fails, check the exact title.
- **Image templates are repo-root assets**: `refresh.png`, `support.png`, `increase.png`, `confirm1.png`, `confirm2.png`, and emoji templates (`sad.png`, `happy.png`, …). Keep filenames stable.
- **OCR grid is hard-coded by config**: `config.ini` defines a 2×3 grid with `start_x/start_y/rect_width/rect_height/padding`.
- **Noise removal**: each OCR region gets a white rectangle applied using `[NoiseRemoval]` in `config.ini`.
- **Duplicate detection semantics**: `sheets.save_to_sheet(..., check_duplicates=True)` compares emoji cells (ignoring timestamp) and returns best matches; `main.py` uses this to annotate Discord posts.

## ML model notes

- Runtime uses `conch_race_ranker.pkl` (joblib dict with keys: `model`, `players`, `features`).
- Ranking features are per-player rows: `rate`, emoji `(valence, arousal)`, plus rate-relative features (`rate-mean`, `rate/mean`, z-score). Keep these consistent if retraining.
- Training pipeline is in `training_light_next.py`; sheet-based prediction example is `predict_light.py`.

## Logging & Discord safety

- `main.py` configures root logging and forwards logs to the panic webhook via `DiscordWebhookLogHandler`.
- `discord.send_panic_notification` must never call `logging` (it prints to stderr on failure) to avoid recursion if the logger is what triggered the panic.
