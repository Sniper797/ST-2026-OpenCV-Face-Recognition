"""Core face logic: detection, normalization, and label decisions.

Kept free of webcam and CLI concerns so it can be unit tested.
"""
import json

import cv2

import config


def load_cascade():
    """Load the frontal-face Haar cascade.

    Raises RuntimeError with a pointed message if the XML is missing, which is
    what happens on OpenCV 5.x where the bundled cascades were removed.
    """
    cascade = cv2.CascadeClassifier(str(config.CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(
            f"Could not load cascade at {config.CASCADE_PATH}.\n"
            "On OpenCV 5.x the bundled cascades were removed. "
            "Install opencv-contrib-python==4.13.0.92."
        )
    return cascade


def detect_faces(gray, cascade=None):
    """Return a list of (x, y, w, h) boxes for faces in a grayscale image."""
    cascade = cascade or load_cascade()
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=config.SCALE_FACTOR,
        minNeighbors=config.MIN_NEIGHBORS,
        minSize=config.MIN_FACE_SIZE,
    )
    return [tuple(int(v) for v in b) for b in boxes]


def normalize_face(gray, box):
    """Crop a face box and standardize it for LBPH.

    Crop -> resize to FACE_SIZE -> histogram equalize. LBPH requires every image
    to share dimensions; equalization reduces sensitivity to lighting.
    """
    x, y, w, h = box
    crop = gray[y:y + h, x:x + w]
    resized = cv2.resize(crop, config.FACE_SIZE)
    return cv2.equalizeHist(resized)


def decide_label(label_id, confidence, labels):
    """Map an LBPH prediction to a display name.

    confidence is a DISTANCE: lower is better, 0.0 is a perfect match. Anything
    at or above the threshold is reported as Unknown.
    """
    if confidence >= config.CONFIDENCE_THRESHOLD:
        return "Unknown"
    return labels.get(str(label_id), "Unknown")


def save_labels(labels, path=None):
    path = path or config.LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labels(path=None):
    path = path or config.LABELS_PATH
    return json.loads(path.read_text(encoding="utf-8"))
