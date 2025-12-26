import argparse
from pathlib import Path
import numpy as np
import cv2


def make_palette(K: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    pal = np.zeros((K + 1, 3), dtype=np.uint8)
    pal[0] = [0, 0, 0]
    pal[1:] = rng.integers(0, 255, size=(K, 3), dtype=np.uint8)
    return pal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_dir", required=True, type=str)
    ap.add_argument("--rgb_dir", required=True, type=str)
    ap.add_argument("--num_classes", required=True, type=int)
    ap.add_argument("--seed", default=123, type=int)
    args = ap.parse_args()

    ids_dir = Path(args.ids_dir)
    rgb_dir = Path(args.rgb_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    pal = make_palette(args.num_classes, args.seed)

    for p in sorted(ids_dir.glob("*_mask.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        rgb = pal[m]  # HxWx3
        out = rgb_dir / p.name.replace("_mask.png", "_cond.png")
        cv2.imwrite(str(out), rgb)

    print(f"[OK] Converted masks -> {rgb_dir}")


if __name__ == "__main__":
    main()
