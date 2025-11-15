import logging
import subprocess
import traceback

def run_training():
    """Runs the training script."""
    logging.info("Starting model training...")
    try:
        subprocess.run(["python", "\\\\vmware-host\\Shared Folders\\conch-race\\training.py"], check=True)
        logging.info("Model training finished.")
    except subprocess.CalledProcessError as e:
        logging.error(traceback.format_exc())
