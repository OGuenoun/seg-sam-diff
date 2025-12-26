import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlnet_dir", required=True, type=str)
    ap.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5", type=str)
    ap.add_argument("--images_real", required=True, type=str)
    ap.add_argument("--cond_rgb", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--num_per_image", default=2, type=int)
    ap.add_argument("--prompt", default="aerial photo", type=str)
    ap.add_argument("--seed0", default=0, type=int)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    images_real = Path(args.images_real)
    cond_rgb = Path(args.cond_rgb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_map = {}  # real_file_name -> list of syn_file_names

    for img_path in sorted(images_real.glob("*.jpg")):
        stem = img_path.stem
        cond_path = cond_rgb / f"{stem}_cond.png"
        if not cond_path.exists():
            print(f"[WARN] missing conditioning for {img_path.name}")
            continue

        cond_img = load_image(str(cond_path))  # PIL RGB
        # To help keep label alignment: generate at same size
        W, H = Image.open(img_path).size

        outs = []
        for i in range(args.num_per_image):
            seed = args.seed0 + i
            g = torch.Generator(device=device).manual_seed(seed)

            result = pipe(
                prompt=args.prompt,
                image=cond_img,
                generator=g,
                num_inference_steps=30,
            ).images[0]

            # Resize back exactly to original (keeps bbox pixel coords valid)
            if result.size != (W, H):
                result = result.resize((W, H), resample=Image.BICUBIC)

            syn_name = f"{stem}_seed{seed}.jpg"
            result.save(out_dir / syn_name, quality=95)
            outs.append(syn_name)

        syn_map[img_path.name] = outs
        print(f"[OK] {img_path.name} -> {len(outs)} synthetic")

    map_path = out_dir / "syn_map.json"
    map_path.write_text(json.dumps(syn_map, indent=2), encoding="utf-8")
    print(f"[OK] wrote {map_path}")


if __name__ == "__main__":
    main()
