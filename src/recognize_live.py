"""Recognize faces in a live webcam feed.

Usage:  python src/recognize_live.py     ('q' to quit)
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
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

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

        gray = faces.to_gray(frame)
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
