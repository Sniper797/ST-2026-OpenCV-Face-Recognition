"""Recognize faces in a live webcam feed.

Usage:  python src/recognize_live.py     ('s' to save a screenshot, 'q' to quit)
"""
import argparse
import sys

import cv2

import faces


def main():
    parser = argparse.ArgumentParser(description="Live face recognition.")
    parser.add_argument("-c", "--camera", type=int, default=0)
    args = parser.parse_args()

    try:
        recognizer = faces.load_recognizer()
        labels = faces.load_labels()
        cascade = faces.load_cascade()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    print("Running. Press 's' to save a screenshot, 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read frame - camera may have been disconnected",
                      file=sys.stderr)
                return 1

            gray = faces.to_gray(frame)
            for box in faces.detect_faces(gray, cascade):
                label_id, distance = recognizer.predict(faces.normalize_face(gray, box))
                name = faces.decide_label(label_id, distance, labels)

                x, y, w, h = box
                colour = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(frame, f"{name} ({distance:.0f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            cv2.imshow("Recognition - 's' saves, 'q' quits", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                # frame carries the boxes and labels drawn above, which is
                # exactly what the README needs.
                path = faces.save_shot(frame, "recognition")
                if path:
                    print(f"  saved {path.name}")
                else:
                    print("Error: could not write screenshot", file=sys.stderr)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
