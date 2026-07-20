"""Capture face images from the webcam to build a training dataset.

Usage:  python src/capture_dataset.py <person-name> [-n 30]
Press SPACE to capture a frame, 'a' to auto-capture, 'q' to quit.
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

        gray = faces.to_gray(frame)
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
        if key == ord(" ") or auto:
            if len(boxes) == 1:
                path = out_dir / f"{saved:03d}.jpg"
                cv2.imwrite(str(path), faces.normalize_face(gray, boxes[0]))
                saved += 1
                print(f"  saved {path.name}")
            else:
                # Ambiguous frame: skip rather than poison the dataset with the
                # wrong person's face or a frame containing nobody.
                print(f"  skipped - need exactly 1 face, saw {len(boxes)}")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"Saved {saved} images to {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
