# Face Detection & Recognition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect faces in images and live webcam video, and identify who they are by name.

**Architecture:** A testable core module (`faces.py`) holds all pure logic — cascade loading, detection, face normalization, and the confidence-to-name decision. Four thin CLI scripts wrap that core for the four user-facing operations. Tests cover the core; webcam loops are verified manually.

**Tech Stack:** Python 3.10, `opencv-contrib-python==4.13.0.92` (contrib build required for `cv2.face` LBPH), NumPy, pytest.

**Deviation from spec:** `docs/DESIGN.md` listed logic living directly in each script. During planning it became clear that puts the interesting logic inside webcam loops where it cannot be tested. Extracting `src/faces.py` keeps every script thin and makes the core unit-testable. Same behavior, better boundaries.

---

## Verified ground truth

These were measured on this machine before planning. Tests below depend on them.

| Fact | Value |
|---|---|
| `opencv-ex/Face.jpg` | 4 people, all 4 detected at `scaleFactor=1.1, minNeighbors=5` |
| `opencv-ex/Cat.jpg` with human cascade | **1 false positive** — do not use as a negative control |
| Solid-gray / black / noise image | 0 detections — reliable negative control |
| Webcam | Index 0, 640×480, working |

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/config.py` | Paths and tunable constants only. No logic. |
| `src/faces.py` | Core: cascade load, detect, normalize, label decision. Fully testable. |
| `src/detect_image.py` | CLI: box faces in a still image |
| `src/capture_dataset.py` | CLI: webcam → `dataset/<name>/*.jpg` |
| `src/train_model.py` | CLI: `dataset/` → LBPH model + label map |
| `src/recognize_live.py` | CLI: webcam → detect + identify |
| `tests/test_faces.py` | Unit tests for the core |
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Excludes `dataset/`, `models/` |
| `README.md` | Full step-by-step + OpenCV 5 incident writeup |

---

## Task 1: Environment and scaffolding

**Files:**
- Create: `requirements.txt`, `.gitignore`, `dataset/.gitkeep`, `models/.gitkeep`

- [ ] **Step 1: Swap to the contrib build**

`cv2.face` ships only in `opencv-contrib-python`, and it conflicts with `opencv-python`.
Uninstall one, install the other:

```bash
PY="C:/ProgramData/anaconda3/envs/opencv/python.exe"
"$PY" -m pip uninstall -y opencv-python
"$PY" -m pip install "opencv-contrib-python==4.13.0.92" pytest
```

- [ ] **Step 2: Verify `cv2.face` now exists**

```bash
"$PY" -c "import cv2; print(cv2.__version__); print(hasattr(cv2,'face')); print(hasattr(cv2,'CascadeClassifier'))"
```

Expected: `4.13.0` / `True` / `True`. If `cv2.face` is False, both packages are still
installed — uninstall both, then install only contrib.

- [ ] **Step 3: Write `requirements.txt`**

```
opencv-contrib-python==4.13.0.92
numpy>=2.0
pytest>=8.0
```

- [ ] **Step 4: Write `.gitignore`**

```
dataset/*
!dataset/.gitkeep
models/*
!models/.gitkeep
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore dataset/.gitkeep models/.gitkeep
git commit -m "chore: pin dependencies and scaffold project layout"
```

---

## Task 2: `config.py`

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write it**

```python
"""Shared paths and tunable constants."""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"

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
```

- [ ] **Step 2: Verify it imports and resolves the cascade**

```bash
"$PY" -c "import sys; sys.path.insert(0,'src'); import config; print(config.CASCADE_PATH.exists())"
```

Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/config.py && git commit -m "feat: add shared configuration"
```

---

## Task 3: Core module — detection and normalization

**Files:**
- Create: `tests/test_faces.py`, `src/faces.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the face core."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
import faces


def test_load_cascade_returns_usable_classifier():
    cascade = faces.load_cascade()
    assert not cascade.empty()


def test_detects_all_four_faces_in_group_photo():
    # Face.jpg contains exactly 4 people; verified on this machine.
    img = cv2.imread(str(config.ROOT.parent / "opencv-ex" / "Face.jpg"))
    assert img is not None, "test fixture Face.jpg not found"
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
"$PY" -m pytest tests/test_faces.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'faces'`

- [ ] **Step 3: Implement `src/faces.py`**

```python
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
"$PY" -m pytest tests/test_faces.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/faces.py tests/test_faces.py
git commit -m "feat: add core face detection and normalization with tests"
```

---

## Task 4: The confidence-threshold decision

This is the single most bug-prone piece — the distance/percentage inversion.
It gets its own tests.

**Files:**
- Modify: `tests/test_faces.py`

- [ ] **Step 1: Add the failing tests**

```python
def test_low_distance_resolves_to_the_person():
    labels = {"0": "Mohammed", "1": "Ali"}
    assert faces.decide_label(0, 12.0, labels) == "Mohammed"


def test_high_distance_resolves_to_unknown():
    # 95 exceeds the threshold: a poor match must NOT be given a name.
    labels = {"0": "Mohammed", "1": "Ali"}
    assert faces.decide_label(0, 95.0, labels) == "Unknown"


def test_threshold_boundary_is_exclusive():
    labels = {"0": "Mohammed"}
    assert faces.decide_label(0, config.CONFIDENCE_THRESHOLD, labels) == "Unknown"
    assert faces.decide_label(0, config.CONFIDENCE_THRESHOLD - 0.1, labels) == "Mohammed"


def test_unseen_label_id_is_unknown():
    assert faces.decide_label(7, 5.0, {"0": "Mohammed"}) == "Unknown"
```

- [ ] **Step 2: Run them**

```bash
"$PY" -m pytest tests/test_faces.py -v
```

Expected: 9 passed (the implementation from Task 3 already satisfies these — these
tests exist to lock the behavior against future edits that invert the comparison).

- [ ] **Step 3: Commit**

```bash
git add tests/test_faces.py
git commit -m "test: lock in confidence-as-distance semantics"
```

---

## Task 5: `detect_image.py`

**Files:**
- Create: `src/detect_image.py`

- [ ] **Step 1: Write it**

```python
"""Detect faces in a still image and draw boxes.

Usage:  python src/detect_image.py <image> [-o output.jpg]
"""
import argparse
import sys
from pathlib import Path

import cv2

import config
import faces


def main():
    parser = argparse.ArgumentParser(description="Detect faces in an image.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("-o", "--output", help="save result instead of displaying")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image at {args.image}", file=sys.stderr)
        return 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = faces.detect_faces(gray)
    print(f"Found {len(boxes)} face(s)")

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if args.output:
        cv2.imwrite(args.output, img)
        print(f"Saved to {args.output}")
    else:
        cv2.imshow("Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # flushes the close event on Windows
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify against the known fixture**

```bash
"$PY" src/detect_image.py ../opencv-ex/Face.jpg -o docs/images/detect_result.jpg
```

Expected: `Found 4 face(s)`

- [ ] **Step 3: Verify the failure path**

```bash
"$PY" src/detect_image.py nope.jpg
```

Expected: `Error: could not read image at nope.jpg`, exit code 1.

- [ ] **Step 4: Commit**

```bash
git add src/detect_image.py docs/images/detect_result.jpg
git commit -m "feat: add still-image face detection script"
```

---

## Task 6: `capture_dataset.py`

**Files:**
- Create: `src/capture_dataset.py`

- [ ] **Step 1: Write it**

```python
"""Capture face images from the webcam to build a training dataset.

Usage:  python src/capture_dataset.py <person-name> [-n 30]
Press SPACE to capture a frame, or 'a' to auto-capture, 'q' to quit.
"""
import argparse
import sys

import cv2

import config
import faces


def main():
    parser = argparse.ArgumentParser(description="Capture face images.")
    parser.add_argument("name", help="name of the person")
    parser.add_argument("-n", "--count", type=int, default=config.IMAGES_PER_PERSON)
    parser.add_argument("-c", "--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = config.DATASET_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cascade = faces.load_cascade()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    print(f"Capturing {args.count} images for '{args.name}'.")
    print("SPACE = capture, 'a' = auto-capture, 'q' = quit")

    saved, auto = 0, False
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            print("Error: failed to read frame", file=sys.stderr)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = faces.detect_faces(gray, cascade)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"{saved}/{args.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture - SPACE/a/q", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            auto = True
        if (key == ord(" ") or auto) and len(boxes) == 1:
            path = out_dir / f"{saved:03d}.jpg"
            cv2.imwrite(str(path), faces.normalize_face(gray, boxes[0]))
            saved += 1
            print(f"  saved {path.name}")
        elif (key == ord(" ") or auto) and len(boxes) != 1:
            # Ambiguous frame: skip rather than poison the dataset.
            print(f"  skipped - need exactly 1 face, saw {len(boxes)}")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"Saved {saved} images to {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

Note the deliberate guard: frames containing zero or multiple faces are never saved.
A dataset labeled "Mohammed" that contains someone else's face silently ruins training.

- [ ] **Step 2: Capture two people**

```bash
"$PY" src/capture_dataset.py Mohammed -n 30
"$PY" src/capture_dataset.py Person2 -n 30
```

A second person is required — LBPH cannot discriminate with only one class.
If no second person is available, use a photo of a different face held up to the camera.

- [ ] **Step 3: Verify the dataset**

```bash
ls dataset/Mohammed | wc -l
```

Expected: 30

- [ ] **Step 4: Commit (script only — dataset is gitignored)**

```bash
git add src/capture_dataset.py
git commit -m "feat: add webcam dataset capture script"
```

---

## Task 7: `train_model.py`

**Files:**
- Create: `src/train_model.py`

- [ ] **Step 1: Write it**

```python
"""Train an LBPH recognizer over dataset/ and save the model + label map.

Usage:  python src/train_model.py
"""
import sys

import cv2
import numpy as np

import config
import faces


def main():
    if not hasattr(cv2, "face"):
        print("Error: cv2.face is missing. Install opencv-contrib-python==4.13.0.92",
              file=sys.stderr)
        return 1

    people = sorted(d for d in config.DATASET_DIR.iterdir() if d.is_dir())
    if not people:
        print(f"Error: no people found in {config.DATASET_DIR}.\n"
              "Expected layout: dataset/<name>/*.jpg — run capture_dataset.py first.",
              file=sys.stderr)
        return 1
    if len(people) < 2:
        print(f"Warning: only {len(people)} person found. LBPH needs at least 2 "
              "to tell people apart; it will match everyone to the single class.",
              file=sys.stderr)

    images, ids, labels = [], [], {}
    for label_id, person in enumerate(people):
        labels[str(label_id)] = person.name
        count = 0
        for img_path in sorted(person.glob("*.jpg")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  skipping unreadable {img_path}", file=sys.stderr)
                continue
            if img.shape != config.FACE_SIZE:
                img = cv2.resize(img, config.FACE_SIZE)
            images.append(img)
            ids.append(label_id)
            count += 1
        print(f"  {person.name}: {count} images (id={label_id})")

    if not images:
        print("Error: no readable images found.", file=sys.stderr)
        return 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(ids))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(config.MODEL_PATH))
    faces.save_labels(labels)

    print(f"Trained on {len(images)} images across {len(people)} people.")
    print(f"Model:  {config.MODEL_PATH}")
    print(f"Labels: {config.LABELS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Train**

```bash
"$PY" src/train_model.py
```

Expected: per-person counts, then `Trained on 60 images across 2 people.`

- [ ] **Step 3: Verify the artifacts**

```bash
ls -la models/ && cat models/labels.json
```

Expected: `lbph_model.yml` present and non-trivial in size; `labels.json` has 2 entries.

- [ ] **Step 4: Commit**

```bash
git add src/train_model.py
git commit -m "feat: add LBPH training script"
```

---

## Task 8: `recognize_live.py`

**Files:**
- Create: `src/recognize_live.py`

- [ ] **Step 1: Write it**

```python
"""Recognize faces in a live webcam feed.

Usage:  python src/recognize_live.py     ('q' to quit)
"""
import argparse
import sys

import cv2

import config
import faces


def main():
    parser = argparse.ArgumentParser(description="Live face recognition.")
    parser.add_argument("-c", "--camera", type=int, default=0)
    args = parser.parse_args()

    if not hasattr(cv2, "face"):
        print("Error: cv2.face is missing. Install opencv-contrib-python==4.13.0.92",
              file=sys.stderr)
        return 1
    if not config.MODEL_PATH.exists():
        print(f"Error: no model at {config.MODEL_PATH}. Run train_model.py first.",
              file=sys.stderr)
        return 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(config.MODEL_PATH))
    labels = faces.load_labels()
    cascade = faces.load_cascade()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    print("Running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for box in faces.detect_faces(gray, cascade):
            label_id, distance = recognizer.predict(faces.normalize_face(gray, box))
            name = faces.decide_label(label_id, distance, labels)

            x, y, w, h = box
            colour = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, f"{name} ({distance:.0f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        cv2.imshow("Recognition - 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify recognition works**

```bash
"$PY" src/recognize_live.py
```

Expected: your face gets a green box with your name and a distance below 70.

- [ ] **Step 3: Verify the negative case**

Point the camera at someone not in the dataset (or a photo of a stranger).

Expected: a red `Unknown` box. **If an unenrolled face gets a name, the threshold is
too permissive** — lower `CONFIDENCE_THRESHOLD` in `config.py` and retest. This check
matters: without it a recognizer that names everyone still passes the positive test.

- [ ] **Step 4: Commit**

```bash
git add src/recognize_live.py
git commit -m "feat: add live face recognition script"
```

---

## Task 9: README

**Files:**
- Create: `README.md` (replacing the stub)

- [ ] **Step 1: Write the README**

It must contain, in order:

1. **What this project does** — detection vs recognition, stated as two distinct jobs.
2. **Setup** — clone, `pip install -r requirements.txt`, note that contrib is required
   for `cv2.face` and conflicts with plain `opencv-python`.
3. **Usage** — the exact four commands, in dependency order, with example output.
4. **How it works** — Haar cascades in two sentences; LBPH in two sentences;
   the normalization pipeline and why uniform size is mandatory.
5. **The OpenCV 5.0 problem we hit and how we solved it** — see Task 10.
6. **Limitations, honestly stated:**
   - Haar cascades produce false positives — a photo of a cat registers one "face"
     with the human cascade, measured on this project's own `Cat.jpg`.
   - LBPH is lighting-sensitive; train under conditions similar to use.
   - Frontal faces only; profiles are missed.
   - LBPH needs ≥2 people to discriminate.
7. **Screenshots** from `docs/images/`.

- [ ] **Step 2: Commit**

```bash
git add README.md docs/images/
git commit -m "docs: add full setup, usage, and limitations"
```

---

## Task 10: The incident writeup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the section**

Content, factual and already verified:

> **Problem:** The notebook cell `pip install opencv-python` was unpinned, so it
> installed **opencv-python 5.0.0.93**. OpenCV 5.0 removed `cv2.CascadeClassifier`
> entirely and shipped an empty `cv2/data/` directory with none of the bundled
> Haar cascade XML files. Every cascade-based script broke with:
>
> `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`
>
> **Why it was confusing:** the notebook's saved output still showed a successful run
> (`Faces: 1`, `Eyes: 2`) because that output was cached from an earlier run under
> OpenCV 4.x. The file looked healthy while the environment beneath it had changed.
>
> **Diagnosis:** confirmed by introspection — `cv2.CascadeClassifier` was absent from
> `cv2`, `cv2.objdetect`, and `cv2.legacy`, and `cv2/data/` contained only
> `__init__.py`. The package's install timestamp matched the day the break appeared.
>
> **Solution:** pin to the 4.x line — `opencv-contrib-python==4.13.0.92`. This restores
> `CascadeClassifier` and all 17 bundled cascades. **Every dependency in this project is
> pinned as a result.** An unpinned install is a time bomb: it works until upstream
> ships a major version, then breaks code you never touched.

- [ ] **Step 2: Commit**

```bash
git add README.md && git commit -m "docs: document the OpenCV 5.0 breakage and fix"
```

---

## Task 11: Final verification and single push

- [ ] **Step 1: Full test run**

```bash
"$PY" -m pytest tests/ -v
```

Expected: 9 passed.

- [ ] **Step 2: Confirm no private data is staged**

```bash
git status --porcelain && git ls-files | grep -c "^dataset/" 
```

Expected: `dataset/` contains only `.gitkeep`. **Do not push if face images appear here.**

- [ ] **Step 3: Review the full diff before it becomes public**

```bash
git log --oneline origin/main..HEAD
git diff origin/main..HEAD --stat
```

- [ ] **Step 4: Push once**

```bash
git push origin main
```

Authentication may prompt — there is no credential helper configured and `gh` is not
installed, so this step may need to be run by the repo owner.

---

## Self-review

**Spec coverage:** detection (T5), recognition (T6-T8), the contrib swap (T1),
confidence-as-distance (T4), normalization (T3), privacy/gitignore (T1, T11),
error handling (T5-T8), testing (T3, T4), incident writeup (T10). All covered.

**Placeholders:** none — every code step contains complete, runnable code.

**Type consistency:** `detect_faces` returns `list[tuple[int,int,int,int]]` and is
consumed as such in T5, T6, T8. `normalize_face(gray, box)` and
`decide_label(label_id, confidence, labels)` keep identical signatures across
definition and all call sites. `labels` is `dict[str, str]` throughout — string keys,
because JSON has no integer keys, which is why `decide_label` looks up `str(label_id)`.
