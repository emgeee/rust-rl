#!/bin/bash
# Helper script to train the 1.5B model

# Navigate to project root if running from scripts directory
if [[ $(basename "$PWD") == "scripts" ]]; then
    cd ..
fi

# Generate run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="rust_rl_Qwen2_5_Coder_1_5B_${TIMESTAMP}"

# Run training
python train_script.py \
    --model-name "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    --run-name "${RUN_NAME}" \
    --dataset-name "mgreen/rust-ft" \
    --dataset-split "train" \
    --output-dir "outputs" \
    --num-epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --save-every 100 \
    --num-generations 4 \
    --use-peft \
    --lora-r 16 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --precision "bf16" \
    "$@"

echo "Training completed for ${RUN_NAME}"
