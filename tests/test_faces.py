"""Unit tests for the face core."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
import faces

# This fixture lives outside the repo and contains photos of real people, so it is
# deliberately not committed. Tests that need it skip cleanly when it is absent.
GROUP_PHOTO = config.ROOT.parent / "opencv-ex" / "Face.jpg"


def test_load_cascade_returns_usable_classifier():
    cascade = faces.load_cascade()
    assert not cascade.empty()


@pytest.mark.skipif(not GROUP_PHOTO.exists(), reason="local fixture Face.jpg not present")
def test_detects_all_four_faces_in_group_photo():
    # Face.jpg contains exactly 4 people; verified on this machine.
    img = cv2.imread(str(GROUP_PHOTO))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert len(faces.detect_faces(gray)) == 4


def test_flat_image_yields_no_detections():
    # A featureless image is the honest negative control. Note that a photo of a
    # cat DOES produce a false positive with this cascade, so it is unsuitable here.
    flat = np.full((300, 300), 128, np.uint8)
    assert len(faces.detect_faces(flat)) == 0


def test_normalize_face_returns_configured_size():
    gray = np.full((480, 640), 100, np.uint8)
    out = faces.normalize_face(gray, (100, 50, 220, 220))
    assert out.shape == config.FACE_SIZE


def test_normalize_face_equalizes_contrast():
    # A low-contrast patch should be stretched toward the full range.
    gray = np.zeros((300, 300), np.uint8)
    gray[50:250, 50:250] = np.random.RandomState(0).randint(100, 140, (200, 200))
    out = faces.normalize_face(gray, (50, 50, 200, 200))
    assert out.max() - out.min() > 200
