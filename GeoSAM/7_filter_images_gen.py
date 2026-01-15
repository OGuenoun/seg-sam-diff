# build_augmented_coco_from_synmap.py
import os, json, math, shutil
from typing import Dict, List, Tuple, Set
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------
# Hist metric (Chi-square)
# -------------------------
def open_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def rgb_hist(img: np.ndarray, bins: int = 32, mask: np.ndarray | None = None) -> np.ndarray:
    # img: HxWx3 uint8, mask: HxW bool
    if mask is None:
        pixels = img.reshape(-1, 3)
    else:
        pixels = img[mask]
        if pixels.size == 0:
            return np.zeros(3 * bins, dtype=np.float32)

    hists = []
    for c in range(3):
        h, _ = np.histogram(pixels[:, c], bins=bins, range=(0, 256))
        h = h.astype(np.float32)
        h /= (h.sum() + 1e-12)
        hists.append(h)
    return np.concatenate(hists, axis=0)

def chi2_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    eps = 1e-12
    return float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps)))

# -------------------------
# COCO helpers
# -------------------------
def load_coco(coco_path: str) -> dict:
    with open(coco_path, "r") as f:
        return json.load(f)

def coco_indexes(coco: dict):
    # file_name -> image dict
    img_by_fname = {im["file_name"]: im for im in coco["images"]}
    # image_id -> list anns
    anns_by_imgid: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_imgid.setdefault(ann["image_id"], []).append(ann)
    return img_by_fname, anns_by_imgid

def compute_category_counts(coco: dict) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        counts[cid] = counts.get(cid, 0) + 1
    return counts

def rare_category_ids_from_counts(counts: Dict[int, int], min_occurrence: int) -> Set[int]:
    # "rare" = strictly less than threshold
    return {cid for cid, c in counts.items() if c < min_occurrence}

def rare_bbox_union_mask(H: int, W: int, anns: List[dict], rare_ids: Set[int]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for ann in anns:
        if ann.get("category_id") not in rare_ids:
            continue
        x, y, w, h = ann["bbox"]  # COCO [x,y,w,h]
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(W, int(math.ceil(x + w)))
        y2 = min(H, int(math.ceil(y + h)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask

def image_contains_rare(anns: List[dict], rare_ids: Set[int]) -> bool:
    return any(ann["category_id"] in rare_ids for ann in anns)

def next_id(existing: List[int]) -> int:
    return (max(existing) + 1) if existing else 1

# -------------------------
# Main pipeline
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--coco_in", required=True, help="Original COCO annotations json")
    ap.add_argument("--syn_map", required=True, help="syn_map.json (orig_fname -> [syn_fnames])")

    ap.add_argument("--orig_img_dir", required=True, help="Directory containing original images (file_name paths resolve here)")
    ap.add_argument("--syn_img_dir", required=True, help="Directory containing synthetic images")

    ap.add_argument("--out_img_dir", required=True, help="Output folder where selected synthetic images will be copied")
    ap.add_argument("--coco_out", required=True, help="Output COCO json (original + selected synthetic)")

    ap.add_argument("--min_occurrence", type=int, default=50, help="Category is rare if count < min_occurrence")
    ap.add_argument("--keep_k", type=int, default=4, help="Keep K best synthetic images per original")
    ap.add_argument("--bins", type=int, default=32, help="Histogram bins per channel")

    args = ap.parse_args()

    os.makedirs(args.out_img_dir, exist_ok=True)

    coco = load_coco(args.coco_in)
    img_by_fname, anns_by_imgid = coco_indexes(coco)

    # 1) Detect rare classes from COCO
    cat_counts = compute_category_counts(coco)
    rare_ids = rare_category_ids_from_counts(cat_counts, args.min_occurrence)
    print(f"Rare category_ids (count < {args.min_occurrence}): {sorted(list(rare_ids))}")
    if not rare_ids:
        print("No rare classes found with this threshold. Nothing to do.")
        # still save a copy of original json
        with open(args.coco_out, "w") as f:
            json.dump(coco, f, indent=2)
        return

    # 2) Load syn_map
    with open(args.syn_map, "r") as f:
        syn_map: Dict[str, List[str]] = json.load(f)

    # Prepare new COCO as a deep-ish copy (we will append)
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": list(coco["images"]),
        "annotations": list(coco["annotations"]),
    }

    existing_image_ids = [im["id"] for im in new_coco["images"]]
    existing_ann_ids = [ann["id"] for ann in new_coco["annotations"]]
    next_image_id = next_id(existing_image_ids)
    next_ann_id = next_id(existing_ann_ids)

    # For safety: map original image_id -> original image dict
    img_by_id = {im["id"]: im for im in coco["images"]}

    added_images = 0
    added_anns = 0

    # 3) For each original in syn_map, filter best K by hist over rare-class bboxes
    for orig_fname, syn_list in tqdm(syn_map.items(), desc="Filtering & building COCO"):
        if orig_fname not in img_by_fname:
            # syn_map might have basenames that don't match COCO file_name
            # If so, you must align naming; here we hard fail to avoid silent bugs.
            raise KeyError(f"Original file_name not found in COCO images: {orig_fname}")

        orig_img_entry = img_by_fname[orig_fname]
        orig_imgid = orig_img_entry["id"]
        orig_anns = anns_by_imgid.get(orig_imgid, [])

        # Only process originals that actually contain rare classes (by annotations)
        if not image_contains_rare(orig_anns, rare_ids):
            continue

        orig_path = os.path.join(args.orig_img_dir, orig_fname)
        if not os.path.exists(orig_path):
            raise FileNotFoundError(f"Original image not found: {orig_path}")

        orig_img = open_rgb(orig_path)
        H, W = orig_img.shape[:2]

        roi_mask = rare_bbox_union_mask(H, W, orig_anns, rare_ids)
        if roi_mask.sum() == 0:
            # This shouldn't happen if image_contains_rare is True, but keep safe:
            continue

        h_orig = rgb_hist(orig_img, bins=args.bins, mask=roi_mask)

        scored: List[Tuple[float, str]] = []
        for syn_fname in syn_list:
            syn_path = os.path.join(args.syn_img_dir, syn_fname)
            if not os.path.exists(syn_path):
                raise FileNotFoundError(f"Synthetic image not found: {syn_path}")

            syn_img = open_rgb(syn_path)
            # If size differs, we still compute hist on resized copy for fair comparison
            if syn_img.shape[:2] != (H, W):
                syn_img_resized = np.array(Image.fromarray(syn_img).resize((W, H), Image.BILINEAR), dtype=np.uint8)
            else:
                syn_img_resized = syn_img

            h_syn = rgb_hist(syn_img_resized, bins=args.bins, mask=roi_mask)
            d = chi2_distance(h_orig, h_syn)
            scored.append((d, syn_fname))

        scored.sort(key=lambda x: x[0])
        chosen = [name for _, name in scored[: args.keep_k]]

        # 4) Copy chosen synthetic images + append them into COCO (duplicate anns)
        for syn_fname in chosen:
            src = os.path.join(args.syn_img_dir, syn_fname)
            dst = os.path.join(args.out_img_dir, syn_fname)
            shutil.copy2(src, dst)

            # read size to store correct width/height
            syn_im = Image.open(dst)
            syn_w, syn_h = syn_im.size

            # Create new image entry
            new_img = {
                "id": next_image_id,
                "file_name": os.path.join(os.path.basename(args.out_img_dir), syn_fname).replace("\\", "/"),
                "width": syn_w,
                "height": syn_h,
            }
            # keep optional fields if present in original image entry (license/date_captured/etc.)
            for k in ["license", "flickr_url", "coco_url", "date_captured"]:
                if k in orig_img_entry:
                    new_img[k] = orig_img_entry[k]

            new_coco["images"].append(new_img)

            # Duplicate annotations from original image, BUT keep only those belonging to the original image.
            # We include ALL original anns (not only rare ones), as you requested "including all original annotations".
            # If synthetic size differs from original, scale bbox/area accordingly.
            sx = syn_w / W
            sy = syn_h / H

            for ann in orig_anns:
                x, y, w, h = ann["bbox"]
                new_bbox = [x * sx, y * sy, w * sx, h * sy]

                new_ann = dict(ann)  # shallow copy
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                new_ann["bbox"] = new_bbox

                # update area if present
                if "area" in new_ann and new_ann["area"] is not None:
                    new_ann["area"] = float(new_bbox[2] * new_bbox[3])

                # segmentation scaling is non-trivial; if your dataset uses segmentation,
                # easiest safe path is to drop it for synthetic images unless you also scale polygons.
                if "segmentation" in new_ann:
                    # If segmentation is RLE, scaling is non-trivial too.
                    # Remove to avoid wrong labels.
                    new_ann.pop("segmentation", None)

                new_coco["annotations"].append(new_ann)
                next_ann_id += 1
                added_anns += 1

            next_image_id += 1
            added_images += 1

    # 5) Save new COCO
    with open(args.coco_out, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"\nDone.")
    print(f"Copied synthetic images to: {args.out_img_dir}")
    print(f"Added synthetic images: {added_images}")
    print(f"Added synthetic annotations: {added_anns}")
    print(f"Saved new COCO json: {args.coco_out}")

if __name__ == "__main__":
    main()
