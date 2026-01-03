import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_coco(coco_path: Path):
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    imgs = {Path(im["file_name"]).name: im["id"] for im in coco["images"]}
    anns = coco["annotations"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    return imgs, anns, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, type=str)
    ap.add_argument("--dataset_dir", required=True, type=str, help="data/diffusers_train")
    ap.add_argument("--min_count", default=200, type=int)
    ap.add_argument("--rare_repeat", default=8, type=int)
    ap.add_argument("--ignore_category_ids", default="0", type=str)
    ap.add_argument("--prompt", default="aerial photo", type=str)
    args = ap.parse_args()

    ignore_ids = {int(x) for x in args.ignore_category_ids.split(",") if x.strip()}

    coco_path = Path(args.coco)
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    cond_dir = dataset_dir / "conditioning_images"

    imgs_by_name, anns, cats = load_coco(coco_path)

    # Count instances per class
    class_counts = defaultdict(int)
    imgid_to_cats = defaultdict(set)

    for a in anns:
        cid = a["category_id"]
        if cid in ignore_ids:
            continue
        class_counts[cid] += 1
        imgid_to_cats[a["image_id"]].add(cid)

    rare_cats = {cid for cid, cnt in class_counts.items() if cnt < args.min_count}
    print("[INFO] Rare classes:", [cats[cid] for cid in rare_cats if cid in cats])

    lines = []
    kept = 0
    oversampled = 0

    for img_path in sorted(images_dir.glob("*.jpg")):
        name = img_path.name
        if name not in imgs_by_name:
            continue

        img_id = imgs_by_name[name]
        present = imgid_to_cats.get(img_id, set())
        has_rare = any(c in rare_cats for c in present)

        cond_path = cond_dir / f"{img_path.stem}_cond.png"
        if not cond_path.exists():
            print(f"[WARN] missing conditioning for {name}")
            continue

        repeat = args.rare_repeat if has_rare else 1
        if has_rare:
            oversampled += 1

        for _ in range(repeat):
            lines.append({
                "image": f"images/{img_path.name}",
                "conditioning_image": f"conditioning_images/{cond_path.name}",
                "text": args.prompt,
            })

        kept += 1

    metadata_path = dataset_dir / "metadata.jsonl"
    metadata_path.write_text(
        "\n".join(json.dumps(r) for r in lines),
        encoding="utf-8"
    )

    print(f"[OK] metadata.jsonl rewritten")
    print(f"[OK] images used: {kept}")
    print(f"[OK] images with rare classes: {oversampled}")
    print(f"[OK] total training rows: {len(lines)}")


if __name__ == "__main__":
    main()
