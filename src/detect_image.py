"""Detect faces in a still image and draw boxes.

Usage:  python src/detect_image.py <image> [-o output.jpg]
"""
import argparse
import sys
from pathlib import Path

import cv2

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

    try:
        cascade = faces.load_cascade()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    boxes = faces.detect_faces(faces.to_gray(img), cascade)
    print(f"Found {len(boxes)} face(s)")

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = cv2.imwrite(str(out_path), img)
        except cv2.error as exc:
            print(f"Error: could not write {out_path} ({exc})", file=sys.stderr)
            return 1
        if not written:
            print(f"Error: could not write {out_path}", file=sys.stderr)
            return 1
        print(f"Saved to {out_path}")
    else:
        cv2.imshow("Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # flushes the close event on Windows
    return 0


if __name__ == "__main__":
    sys.exit(main())
