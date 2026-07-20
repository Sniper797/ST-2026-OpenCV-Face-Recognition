"""Capture face images from the webcam to build a training dataset.

Usage:  python src/capture_dataset.py <person-name> [-n 30] [-c 0]
Press SPACE to capture a frame, 'a' to toggle auto-capture, 's' to save a
screenshot of this window, 'q' to quit.
"""
import argparse
import re
import sys
import time

import cv2

import config
import faces


def main():
    parser = argparse.ArgumentParser(description="Capture face images.")
    parser.add_argument("name", help="name of the person")
    parser.add_argument("-n", "--count", type=int, default=config.IMAGES_PER_PERSON)
    parser.add_argument("-c", "--camera", type=int, default=0,
                        help="camera index (default 0)")
    args = parser.parse_args()

    # Keep the name a single path segment: '../models/pwned' would otherwise
    # write outside dataset/.
    if not re.fullmatch(r"[\w -]+", args.name):
        print(f"Error: '{args.name}' is not a valid name. Use letters, digits, "
              "spaces, hyphens or underscores.", file=sys.stderr)
        return 1

    try:
        cascade = faces.load_cascade()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    # Created only once the camera works, so a failed run leaves no empty
    # directory for train_model.py to mistake for a person.
    out_dir = config.DATASET_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing {args.count} images for '{args.name}'.")
    print("SPACE = capture, 'a' = toggle auto-capture, 's' = screenshot, 'q' = quit")

    saved, auto, last_save = 0, False, 0.0
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read frame - camera may have been disconnected",
                      file=sys.stderr)
                break

            gray = faces.to_gray(frame)
            boxes = faces.detect_faces(gray, cascade)
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"{saved}/{args.count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture - SPACE/a/s/q", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
                print(f"  auto-capture {'on' if auto else 'off'}")
            if key == ord("s"):
                # Documents the capture process itself: the detection box and
                # the progress counter, not the cropped face being written.
                # Deliberately does not fall through to a dataset capture.
                shot = faces.save_shot(frame, "capture")
                if shot:
                    print(f"  screenshot {shot.name}")
                else:
                    print("Error: could not write screenshot", file=sys.stderr)

            if key == ord(" ") or auto:
                # Consecutive frames are near-identical, so an unthrottled
                # burst fills the dataset with copies of a single pose.
                if time.monotonic() - last_save < config.CAPTURE_INTERVAL:
                    continue
                if len(boxes) == 1:
                    path = out_dir / f"{saved:03d}.jpg"
                    if not cv2.imwrite(str(path), faces.normalize_face(gray, boxes[0])):
                        print(f"Error: could not write {path}", file=sys.stderr)
                        break
                    saved += 1
                    last_save = time.monotonic()
                    print(f"  saved {path.name}")
                else:
                    # Ambiguous frame: skip rather than poison the dataset with
                    # the wrong person's face or a frame containing nobody.
                    last_save = time.monotonic()   # also throttles this message
                    print(f"  skipped - need exactly 1 face, saw {len(boxes)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"Saved {saved} images to {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
