# Design — OpenCV Face Detection & Recognition

**Date:** 2026-07-20
**Task:** SmartMethods ST-2026, Task 1 — *Make a project using OpenCV* → Face Recognition

## Goal

Detect human faces in still images and in a live webcam feed, and identify *who* a
detected face belongs to by name.

Two distinct capabilities, deliberately kept separate:

| Capability | Question answered | Technique |
|---|---|---|
| Detection | *Where* is a face? | Haar cascade |
| Recognition | *Who* is this? | LBPH (Local Binary Patterns Histograms) |

## Architecture

Four single-purpose scripts, communicating through files on disk:

```
capture_dataset.py  →  dataset/<name>/*.jpg
                              │
                              ▼
train_model.py      →  models/lbph_model.yml + models/labels.json
                              │
                              ▼
recognize_live.py   →  webcam → box + name + confidence

detect_image.py     →  image file → boxed faces        (independent)
```

`detect_image.py` depends on nothing but the cascade, so it runs correctly before any
model has been trained. The repo is therefore never in a half-broken state.

### Files

| Path | Responsibility | Depends on |
|---|---|---|
| `src/config.py` | Paths, cascade location, face size, threshold | — |
| `src/detect_image.py` | Haar detection on a still image | config |
| `src/capture_dataset.py` | Capture N aligned face crops from webcam | config |
| `src/train_model.py` | Train LBPH, write model + label map | config, dataset |
| `src/recognize_live.py` | Live detect + identify | config, model |

Each is runnable standalone from the command line and holds one clear job.

## Key technical decisions

### 1. `opencv-contrib-python`, not `opencv-python`

LBPH lives in `cv2.face`, which ships **only** in `opencv-contrib-python`. The two
packages install into the same `cv2` namespace and conflict, so exactly one may be
installed. We pin `opencv-contrib-python==4.13.0.92` — the same version we pinned
during the incident below, but the superset build. Detection is unaffected.

### 2. LBPH confidence is a *distance*, not a percentage

`recognizer.predict()` returns `(label, confidence)` where **confidence is a distance:
lower means a better match, and 0.0 is a perfect match.** Reading it as "percent
confident" inverts the logic and makes the system confidently wrong.

Rule: `confidence < 70` → report the name; otherwise report `Unknown`.

### 3. Faces are normalized before training

LBPH requires every training image to be the **same dimensions** or training fails.
Each captured face is therefore: cropped to the detection box → converted to grayscale
→ resized to 200×200 → histogram-equalized (which reduces sensitivity to lighting).
The identical transform is applied at recognition time.

## Privacy

`dataset/` is **gitignored**. The repository is public, and git history is permanent —
committing face photos then deleting them in a later commit does not remove them.
No biometric data leaves the local machine. The repo ships `capture_dataset.py` plus
instructions so that anyone cloning generates their own dataset.

## Error handling

| Condition | Behavior |
|---|---|
| Image file missing / unreadable | Clear message naming the path, exit non-zero |
| Webcam unavailable | Report the camera index tried, exit non-zero |
| Cascade XML not found | Report resolved path, hint at the OpenCV 5 issue |
| `cv2.face` missing | Explicitly instruct installing `opencv-contrib-python` |
| Training set empty | Report expected `dataset/<name>/` layout |
| Fewer than 2 people | Warn — LBPH needs ≥2 classes to discriminate |

Scripts exit non-zero on failure. `exit()` is never used (it terminates a Jupyter
kernel rather than the cell); `sys.exit(1)` is used instead.

## Testing

- `detect_image.py` against a known face photo — expect ≥1 box on the face.
- Negative control: a non-face image — expect 0 detections.
- `train_model.py` on a 2-person dataset — expect a written model and a 2-entry label map.
- `recognize_live.py` — the enrolled person resolves to their name; an unenrolled
  person resolves to `Unknown` (verifies the threshold actually discriminates).

## Appendix — the OpenCV 5.0 incident

Documented in full in the README. Summary: an unpinned `pip install opencv-python`
pulled 5.0.0.93, which removed `cv2.CascadeClassifier` and emptied the bundled
`cv2/data/` cascade directory, breaking all Haar cascade code with
`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.
Resolved by pinning to the 4.x line. This is why every dependency here is pinned.
