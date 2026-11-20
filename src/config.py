"""Paths and constants shared by all scripts and the app."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Large files live outside git. Override with environment variables if needed
# (e.g. point FRF_CKPT_DIR to a Google Drive folder when working in Colab).
WEIGHTS_DIR = Path(os.environ.get("FRF_WEIGHTS_DIR", ROOT / "weights"))
CKPT_DIR = Path(os.environ.get("FRF_CKPT_DIR", ROOT / "work" / "checkpoints"))
FOOD101_DIR = Path(os.environ.get("FRF_FOOD101_DIR", ROOT / "work" / "food-101"))

CLASSES = [
    "pizza", "hamburger", "pancakes", "ice_cream", "french_fries",
    "omelette", "chocolate_cake", "lasagna", "tacos", "caesar_salad",
]
SEED = 42

# ImageNet statistics (the VGG16 backbone was pre-trained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Google Drive file ids of the large weight files (see README, "Weights").
# Share each file with "Anyone with the link" and paste its id here.
WEIGHT_FILES = {
    "vgg16_fp32.pt": "<DRIVE_FILE_ID>",
    "vgg16_int8.ts": "<DRIVE_FILE_ID>",
}
