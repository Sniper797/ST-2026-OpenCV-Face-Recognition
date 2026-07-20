"""Detect faces in a still image and draw boxes.

Usage:  python src/detect_image.py <image> [-o output.jpg]
"""
import argparse
import sys

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

    boxes = faces.detect_faces(faces.to_gray(img))
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
