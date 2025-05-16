#!/bin/bash
# Example script to train the 4B model

# Run the training script with specific parameters
python train_script.py \
  --model-name "Qwen/Qwen3-4B" \
  --run-name "rust_rl_4b_run1" \
  --num-generations 4 \
  --learning-rate 5e-6 \
  --gradient-accumulation-steps 8 \
  --wandb-tags rust 4b \
  "$@"