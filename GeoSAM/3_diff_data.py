import argparse
import json
import os
from pathlib import Path
import shutil


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_real", required=True, type=str)
    ap.add_argument("--cond_rgb", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--prompt", default="aerial photo", type=str)
    args = ap.parse_args()

    images_real = Path(args.images_real)
    cond_rgb = Path(args.cond_rgb)
    out_dir = Path(args.out_dir)

    images_out = out_dir / "images"
    cond_out = out_dir / "conditioning_images"
    images_out.mkdir(parents=True, exist_ok=True)
    cond_out.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.jsonl"
    lines = []

    # We match by stem: 0001.jpg <-> 0001_cond.png
    for img_path in sorted(images_real.glob("*.jpg")):
        stem = img_path.stem
        cond_path = cond_rgb / f"{stem}_cond.png"
        if not cond_path.exists():
            print(f"[WARN] missing conditioning for {img_path.name}: {cond_path}")
            continue

        img_dst = images_out / img_path.name
        cond_dst = cond_out / cond_path.name

        link_or_copy(img_path, img_dst)
        link_or_copy(cond_path, cond_dst)

        # Relative paths inside dataset folder (important)
        lines.append({
            "file_name": f"images/{img_dst.name}",
            "conditioning_image": f"conditioning_images/{cond_dst.name}",
            "text": args.prompt
        })

    with metadata_path.open("w", encoding="utf-8") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")

    print(f"[OK] diffusers dataset at: {out_dir}")
    print(f"[OK] metadata rows: {len(lines)}")


if __name__ == "__main__":
    main()
