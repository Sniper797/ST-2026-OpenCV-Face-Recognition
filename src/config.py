"""Shared paths and tunable constants."""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"

# Where 's' in recognize_live.py writes its screenshots. This folder is tracked
# by git — anything saved here is meant for the README. docs/images/ stays
# gitignored for throwaway output.
SHOTS_DIR = ROOT / "docs" / "screenshots"

CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

# Every training image must be this size — LBPH requires uniform dimensions.
FACE_SIZE = (200, 200)

# Detection tuning.
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# LBPH confidence is a DISTANCE: lower is a better match, 0.0 is perfect.
# Predictions at or above this are reported as "Unknown".
CONFIDENCE_THRESHOLD = 70.0

IMAGES_PER_PERSON = 30

# Minimum seconds between captures. Without this, holding SPACE or enabling
# auto-capture writes near-identical consecutive frames, so a 30-image dataset
# becomes 30 copies of one pose and LBPH overfits to that single angle.
CAPTURE_INTERVAL = 0.5
