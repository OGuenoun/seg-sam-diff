import argparse
import json
from pathlib import Path
from collections import defaultdict
import copy


def load_coco(coco_path: Path):
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    imgs_by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_imgid = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_imgid[ann["image_id"]].append(ann)
    return coco, imgs_by_name, anns_by_imgid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, type=str)
    ap.add_argument("--syn_images_dir", required=True, type=str)
    ap.add_argument("--syn_map", required=True, type=str)
    ap.add_argument("--out_coco", required=True, type=str)
    args = ap.parse_args()

    coco_path = Path(args.coco)
    syn_images_dir = Path(args.syn_images_dir)
    syn_map = json.loads(Path(args.syn_map).read_text(encoding="utf-8"))

    coco, imgs_by_name, anns_by_imgid = load_coco(coco_path)

    # new COCO with same categories
    coco_syn = {
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(coco["categories"]),
    }

    max_img_id = max(im["id"] for im in coco["images"]) if coco["images"] else 0
    max_ann_id = max(a["id"] for a in coco["annotations"]) if coco["annotations"] else 0
    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    for real_name, syn_names in syn_map.items():
        if real_name not in imgs_by_name:
            print(f"[WARN] real image not in COCO images list: {real_name}")
            continue

        real_im = imgs_by_name[real_name]
        real_imgid = real_im["id"]
        real_anns = anns_by_imgid.get(real_imgid, [])

        for syn_name in syn_names:
            syn_path = syn_images_dir / syn_name
            if not syn_path.exists():
                print(f"[WARN] synthetic image missing on disk: {syn_path}")
                continue

            syn_imgid = next_img_id
            next_img_id += 1

            coco_syn["images"].append({
                "id": syn_imgid,
                "file_name": syn_name,
                "width": real_im["width"],
                "height": real_im["height"],
            })

            for ann in real_anns:
                a = copy.deepcopy(ann)
                a["id"] = next_ann_id
                next_ann_id += 1
                a["image_id"] = syn_imgid
                coco_syn["annotations"].append(a)

    out_path = Path(args.out_coco)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco_syn, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    print(f"[OK] syn images: {len(coco_syn['images'])}, syn anns: {len(coco_syn['annotations'])}")


if __name__ == "__main__":
    main()
