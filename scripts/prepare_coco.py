"""Build the 250-image COCO evaluation subset (512x512).

    python scripts/prepare_coco.py --src /path/to/coco/val2017 --dst dataset/COCO/resized_images --n 250 --seed 42
"""
import argparse
import os
import random
from pathlib import Path

import cv2

EXTS = (".jpg", ".jpeg", ".png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["center_crop", "stretch"], default="center_crop")
    a = p.parse_args()

    files = sorted(f for f in os.listdir(a.src) if f.lower().endswith(EXTS))
    chosen = sorted(random.Random(a.seed).sample(files, a.n))
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        img = cv2.imread(os.path.join(a.src, name))
        if a.mode == "center_crop":
            h, w = img.shape[:2]
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            img = img[y0:y0 + s, x0:x0 + s]
        img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / (Path(name).stem + ".png")), img)
    (dst.parent / "coco_subset.txt").write_text("\n".join(chosen) + "\n")
    print(f"wrote {len(chosen)} images to {dst} (file list: {dst.parent / 'coco_subset.txt'})")


if __name__ == "__main__":
    main()
