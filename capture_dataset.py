# capture_dataset.py
"""
Capture images from webcam and save to data/train/<name> for training.
Usage:
 python capture_dataset.py --name JohnDoe --count 30 --out data/train
Press SPACE to capture an image, ESC to exit.
"""

import cv2
import os
import argparse
import time

def main(args):
    name = args.name
    count = args.count
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    dest = os.path.join(out_root, name)
    os.makedirs(dest, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press SPACE to capture frame, ESC to exit.")
    saved = 0
    while saved < count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        cv2.putText(frame, f"{name}: {saved}/{count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture (SPACE to save)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == 32:  # SPACE
            fname = os.path.join(dest, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            saved += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {saved} images for {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--out', type=str, default='data/train')
    args = parser.parse_args()
    main(args)
