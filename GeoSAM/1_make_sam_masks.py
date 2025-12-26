import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from samgeo import SamGeo2


def load_coco(coco_path: Path, ignore_category_ids=None):
    if ignore_category_ids is None:
        ignore_category_ids = set()

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in ignore_category_ids:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    # Map category_id -> contiguous 1..K (mask values); 0 reserved for background
    cat_ids = sorted([c["id"] for c in coco["categories"] if c["id"] not in ignore_category_ids])
    catid_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    return coco, imgs, anns_by_img, catid_to_idx


def xywh_to_xyxy(b):
    x, y, w, h = b
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def shrink_box(x1, y1, x2, y2, frac, W, H):
    w = x2 - x1
    h = y2 - y1
    dx = int(round(frac * w))
    dy = int(round(frac * h))
    return clamp_box(x1 + dx, y1 + dy, x2 - dx, y2 - dy, W, H)


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    if mask.sum() == 0:
        return (mask > 0).astype(np.uint8)

    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = mask | flood_inv
    return (filled > 0).astype(np.uint8)


def postprocess(mask01: np.ndarray, k: int = 5) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = fill_holes(m)
    return m


def samgeo_predict_best_mask(sam: SamGeo2, box_xyxy: np.ndarray) -> np.ndarray:
    # Try boxes= then box= for compatibility
    try:
        out = sam.predict(
            boxes=box_xyxy.astype(np.float32),
            normalize_coords=True,
            multimask_output=True,
            dtype="uint8",
            return_results=True,
        )
    except TypeError:
        out = sam.predict(
            box=box_xyxy.astype(np.float32),
            normalize_coords=True,
            multimask_output=True,
            dtype="uint8",
            return_results=True,
        )

    if isinstance(out, dict):
        masks = out.get("masks", None)
        scores = out.get("scores", None)
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        masks, scores = out[0], out[1]
    else:
        masks, scores = out, None

    if masks is None:
        raise RuntimeError("SamGeo2 did not return masks")

    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, :, :]

    if scores is None:
        best = 0
    else:
        scores = np.asarray(scores).reshape(-1)
        best = int(np.argmax(scores))

    return (masks[best] > 0).astype(np.uint8)


def bbox_cover(mask01: np.ndarray, x1, y1, x2, y2) -> float:
    crop = mask01[y1:y2, x1:x2]
    area = max(1, (y2 - y1) * (x2 - x1))
    return float(crop.sum()) / float(area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, type=str)
    ap.add_argument("--images_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--sam_model_id", default="sam2-hiera-large", type=str)

    ap.add_argument("--ignore_category_ids", default="0", type=str,
                    help="Comma-separated COCO category_ids to ignore (e.g. '0').")

    ap.add_argument("--min_bbox_cover", default=0.20, type=float,
                    help="Accept SAM mask only if it covers >= this fraction of the bbox.")
    ap.add_argument("--shrink_frac", default=0.12, type=float,
                    help="Shrink bbox for SAM prompt to reduce clutter (0-0.3).")
    ap.add_argument("--morph_kernel", default=5, type=int,
                    help="Kernel size for morphological closing.")
    args = ap.parse_args()

    ignore_ids = set()
    if args.ignore_category_ids.strip():
        ignore_ids = set(int(x) for x in args.ignore_category_ids.split(",") if x.strip() != "")

    coco_path = Path(args.coco)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, imgs, anns_by_img, catid_to_idx = load_coco(coco_path, ignore_category_ids=ignore_ids)

    sam = SamGeo2(model_id=args.sam_model_id, automatic=False)

    used_sam = 0
    used_rect = 0

    for image_id, im in imgs.items():
        fn = im["file_name"]
        img_path = images_dir / fn
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:

            continue
        H, W = bgr.shape[:2]

        sam.set_image(str(img_path))

        anns = anns_by_img.get(image_id, [])
        # draw small first so they survive
        anns = sorted(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])

        cond = np.zeros((H, W), dtype=np.uint8)

        for ann in anns:
            cid = ann["category_id"]
            if cid in ignore_ids:
                continue
            cls = catid_to_idx[cid]

            x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

            sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, args.shrink_frac, W, H)
            box = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

            use_sam = False
            cover = 0.0

            try:
                m = samgeo_predict_best_mask(sam, box)
                m = postprocess(m, k=args.morph_kernel)
                cover = bbox_cover(m, x1, y1, x2, y2)
                use_sam = (cover >= args.min_bbox_cover)
            except Exception as e:

                use_sam = False

            if use_sam:
                used_sam += 1
                cond[m == 1] = cls

            else:
                used_rect += 1
                cond[y1:y2, x1:x2] = cls


        out_path = out_dir / (Path(fn).stem + "_mask.png")
        cv2.imwrite(str(out_path), cond)


if __name__ == "__main__":
    main()
