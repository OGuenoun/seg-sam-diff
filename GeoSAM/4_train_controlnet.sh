#!/usr/bin/env bash
set -e

# Example paths (edit as needed)
TRAIN_DATA_DIR="data/diffusers_train"
OUTPUT_DIR="models/controlnet/cadot_seg"
BASE_MODEL="runwayml/stable-diffusion-v1-5"

# You need accelerate configured:
# accelerate config

accelerate launch diffusers/examples/controlnet/train_controlnet.py \
  --pretrained_model_name_or_path="$BASE_MODEL" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --image_column="image" \
  --conditioning_image_column="conditioning_image" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=8 \
  --learning_rate=1e-5 \
  --max_train_steps=20000 \
  --checkpointing_steps=2000 \
  --validation_steps=2000 \
  --output_dir="$OUTPUT_DIR" \
  --report_to="none"
