import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure logger
logger = logging.getLogger("ad_cluster_backend")
logger.setLevel(logging.INFO)

# File handler (rotates logs up to 5 files, 5MB each)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Attach handlers
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
