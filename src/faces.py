"""Core face logic: detection, normalization, and label decisions.

Kept free of webcam and CLI concerns so it can be unit tested.
"""
import functools
import json

import cv2

import config


@functools.lru_cache(maxsize=1)
def load_cascade():
    """Load the frontal-face Haar cascade.

    Cached: parsing the ~900KB XML costs more than a detection pass itself, and
    all four CLI scripts reach for it. Raises RuntimeError with a pointed
    message if the XML is missing, which is what happens on OpenCV 5.x where
    the bundled cascades were removed.
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

    Crop -> resize to FACE_SIZE -> histogram equalize. LBPH requires every
    image to share dimensions; equalization reduces sensitivity to lighting.

    Input must be grayscale. Boxes are clamped to the image bounds, so a box that
    hangs off an edge yields a smaller region that is then stretched to FACE_SIZE.
    detectMultiScale never returns such boxes, but adding a margin (x - 20, w + 40)
    can — which distorts the face's aspect ratio and degrades training data.

    A box entirely outside the image raises ValueError rather than returning an
    empty crop, which cv2.resize would reject with a less helpful message.
    """
    x, y, w, h = box
    height, width = gray.shape[:2]
    x, y = max(0, x), max(0, y)
    crop = gray[y:min(y + h, height), x:min(x + w, width)]
    if crop.size == 0:
        raise ValueError(f"Face box {box} lies outside the {width}x{height} image.")
    resized = cv2.resize(crop, config.FACE_SIZE)
    return cv2.equalizeHist(resized)


def to_gray(img):
    """Convert a BGR frame to grayscale, passing through already-gray input."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def decide_label(label_id, confidence, labels):
    """Map an LBPH prediction to a display name.

    confidence is a DISTANCE: lower is better, 0.0 is a perfect match. Anything
    at or above the threshold is reported as Unknown.
    """
    if confidence >= config.CONFIDENCE_THRESHOLD:
        return "Unknown"
    return labels.get(str(label_id), "Unknown")


def create_recognizer():
    """Create an untrained LBPH recognizer.

    Raises RuntimeError if cv2.face is absent, which happens when plain
    opencv-python is installed instead of the contrib build.
    """
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is missing. Install opencv-contrib-python==4.13.0.92 "
            "(plain opencv-python does not include the LBPH recognizer)."
        )
    return cv2.face.LBPHFaceRecognizer_create()


def load_recognizer(model_path=None):
    """Create an LBPH recognizer and load a trained model from disk."""
    model_path = model_path or config.MODEL_PATH
    if not model_path.exists():
        raise RuntimeError(f"No trained model at {model_path}. Run train_model.py first.")
    recognizer = create_recognizer()
    recognizer.read(str(model_path))
    return recognizer


def save_labels(labels, path=None):
    path = path or config.LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labels(path=None):
    path = path or config.LABELS_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"No label map at {path}. Run train_model.py first.") from None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Label map at {path} is corrupt ({e}). Re-run train_model.py.") from None
