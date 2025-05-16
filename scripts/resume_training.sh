#!/bin/bash
# Helper script to resume training from a checkpoint

# Check if checkpoint is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint-path> [additional-args...]"
    echo "Example: $0 outputs/rust_rl_Qwen2_5_Coder_1_5B_20250516/checkpoint-500"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift

# Navigate to project root if running from scripts directory
if [[ $(basename "$PWD") == "scripts" ]]; then
    cd ..
fi

# Generate run name indicating it's a resumed run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Extract the original run name from the checkpoint path if possible
ORIG_RUN_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))
if [[ $ORIG_RUN_NAME == rust_rl_* ]]; then
    RUN_NAME="${ORIG_RUN_NAME}_resumed_${TIMESTAMP}"
else
    RUN_NAME="resumed_training_${TIMESTAMP}"
fi

# Run training with checkpoint
python train_script.py \
    --run-name "${RUN_NAME}" \
    --resume-from-checkpoint "${CHECKPOINT_PATH}" \
    --dataset-name "mgreen/rust-ft" \
    --dataset-split "train" \
    --output-dir "outputs" \
    --save-every 100 \
    --num-generations 4 \
    "$@"

echo "Resumed training completed for ${RUN_NAME}"