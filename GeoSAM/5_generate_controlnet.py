import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image


# -------------------------
# COCO helpers
# -------------------------
def load_coco(coco_path: Path):
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}
    anns = coco["annotations"]
    cats = {c["id"]: c for c in coco["categories"]}
    return imgs, anns, cats


def compute_class_counts(annotations, ignore_ids):
    counts = defaultdict(int)
    for a in annotations:
        cid = a["category_id"]
        if cid in ignore_ids:
            continue
        counts[cid] += 1
    return counts


def images_with_rare_classes(imgs_by_id, annotations, rare_cat_ids, ignore_ids):
    keep = set()
    for a in annotations:
        cid = a["category_id"]
        if cid in ignore_ids:
            continue
        if cid in rare_cat_ids:
            keep.add(imgs_by_id[a["image_id"]]["file_name"])
    return keep


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # Models / paths
    ap.add_argument("--controlnet_dir", required=True, type=str)
    ap.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5", type=str)
    ap.add_argument("--images_real", required=True, type=str)
    ap.add_argument("--cond_rgb", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--coco", required=True, type=str)

    # Filtering
    ap.add_argument("--min_count", default=200, type=int, help="Class is rare if instances < min_count")
    ap.add_argument("--ignore_category_ids", default="0", type=str)

    # Generation params (low-VRAM safe)
    ap.add_argument("--num_per_image", default=1, type=int)
    ap.add_argument("--prompt", default="aerial photo", type=str)
    ap.add_argument("--negative_prompt", default="blurry, low quality, distorted, artifacts, text, watermark", type=str)
    ap.add_argument("--steps", default=20, type=int)
    ap.add_argument("--guidance_scale", default=4.5, type=float)
    ap.add_argument("--controlnet_scale", default=1.0, type=float)
    ap.add_argument("--gen_res", default=256, type=int)
    ap.add_argument("--seed0", default=0, type=int)

    args = ap.parse_args()

    ignore_ids = {int(x) for x in args.ignore_category_ids.split(",") if x.strip()}

    # -------------------------
    # COCO analysis
    # -------------------------
    imgs_by_id, anns, cats = load_coco(Path(args.coco))
    class_counts = compute_class_counts(anns, ignore_ids)

    rare_cat_ids = {cid for cid, cnt in class_counts.items() if cnt < args.min_count}
    if not rare_cat_ids:
        print("[INFO] No rare classes found. Nothing to generate.")
        return

    rare_names = [cats[cid]["name"] for cid in sorted(rare_cat_ids) if cid in cats]
    print(f"[INFO] Rare classes (<{args.min_count} instances): {rare_names}")

    keep_files = images_with_rare_classes(imgs_by_id, anns, rare_cat_ids, ignore_ids)
    print(f"[INFO] Images selected for generation: {len(keep_files)}")

    # -------------------------
    # Load ControlNet pipeline
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # -------------------------
    # Generation loop
    # -------------------------
    images_real = Path(args.images_real)
    cond_rgb = Path(args.cond_rgb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_map = {}

    for img_path in sorted(images_real.glob("*.jpg")):
        if img_path.name not in keep_files:
            continue

        stem = img_path.stem
        cond_path = cond_rgb / f"{stem}_cond.png"
        if not cond_path.exists():
            print(f"[WARN] Missing conditioning for {img_path.name}")
            continue

        cond_img = load_image(str(cond_path)).convert("RGB")
        cond_img = cond_img.resize((args.gen_res, args.gen_res), Image.NEAREST)

        W, H = Image.open(img_path).size
        outs = []

        for i in range(args.num_per_image):
            seed = args.seed0 + i
            g = torch.Generator(device=device).manual_seed(seed)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                img = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    image=cond_img,
                    generator=g,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.controlnet_scale,
                ).images[0]

            if img.size != (W, H):
                img = img.resize((W, H), Image.BICUBIC)

            name = f"{stem}_seed{seed}.jpg"
            img.save(out_dir / name, quality=95)
            outs.append(name)

        syn_map[img_path.name] = outs
        print(f"[OK] {img_path.name} → {len(outs)} synthetic")

        if device == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "syn_map.json").write_text(json.dumps(syn_map, indent=2))
    print(f"[DONE] Generated synthetic images for {len(syn_map)} real images")


if __name__ == "__main__":
    main()
