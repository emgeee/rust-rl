#!/bin/bash
# Example script to train the 1.5B model

# Run the training script with specific parameters
python train_script.py \
  --model-name "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --run-name "rust_rl_1_5b_run1" \
  --num-generations 4 \
  --learning-rate 5e-6 \
  --wandb-tags rust 1_5b \
  "$@"