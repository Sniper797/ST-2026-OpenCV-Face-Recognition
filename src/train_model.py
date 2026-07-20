"""Train an LBPH recognizer over dataset/ and save the model + label map.

Usage:  python src/train_model.py
"""
import sys

import cv2
import numpy as np

import config
import faces


def main():
    try:
        recognizer = faces.create_recognizer()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not config.DATASET_DIR.exists():
        print(f"Error: {config.DATASET_DIR} does not exist. Run capture_dataset.py first.",
              file=sys.stderr)
        return 1

    people = sorted(d for d in config.DATASET_DIR.iterdir() if d.is_dir())
    if not people:
        print(f"Error: no people found in {config.DATASET_DIR}.\n"
              "Expected layout: dataset/<name>/*.jpg - run capture_dataset.py first.",
              file=sys.stderr)
        return 1
    if len(people) < 2:
        print(f"Warning: only {len(people)} person found. LBPH needs at least 2 to tell "
              "people apart; it will match every face to the single class.",
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
