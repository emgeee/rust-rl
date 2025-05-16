#!/bin/bash
# Script to resume training from a checkpoint

# Check if checkpoint path is provided
if [ "$#" -lt 1 ]; then
    echo "Error: Checkpoint path is required"
    echo "Usage: $0 <checkpoint_path> [additional_args...]"
    exit 1
fi

CHECKPOINT_PATH=$1
shift  # Remove the first argument (checkpoint path)

# Look for the parent output directory (usually containing the run name)
PARENT_DIR=$(dirname "$CHECKPOINT_PATH")
echo "Parent directory: $PARENT_DIR"

# Extract WandB ID from various possible sources
WANDB_ID=""

# Try looking for wandb files in the parent directory
WANDB_DIR=$(find "$PARENT_DIR" -type d -name "wandb" 2>/dev/null | head -n 1)
if [ -n "$WANDB_DIR" ]; then
    echo "Found WandB directory: $WANDB_DIR"
    # Look for run files
    RUN_ID_FILE=$(find "$WANDB_DIR" -name "*.wandb" -o -name "run-*.json" 2>/dev/null | head -n 1)
    if [ -n "$RUN_ID_FILE" ]; then
        # Extract ID from filename (e.g., run-20230101_123456-abc123.wandb -> abc123)
        BASENAME=$(basename "$RUN_ID_FILE")
        if [[ "$BASENAME" =~ -([a-z0-9]+)\. ]]; then
            WANDB_ID="${BASH_REMATCH[1]}"
        fi
    fi
fi

# If we still don't have an ID, look for other files
if [ -z "$WANDB_ID" ]; then
    # Try looking for the wandb-summary.json file
    WANDB_SUMMARY=$(find "$PARENT_DIR" -name "wandb-summary.json" 2>/dev/null | head -n 1)
    if [ -n "$WANDB_SUMMARY" ]; then
        WANDB_ID=$(grep -o '"_wandb_id": "[^"]*"' "$WANDB_SUMMARY" | cut -d'"' -f4)
    fi
fi

# Set up wandb argument if we found an ID
if [ -n "$WANDB_ID" ]; then
    echo "Found WandB ID: $WANDB_ID"
    WANDB_ARG="--resume-wandb-id $WANDB_ID"
else
    echo "No WandB ID found. Will create a new WandB run."
    WANDB_ARG=""
fi

# Run the training script with the checkpoint
python train_script.py \
  --resume-from-checkpoint "$CHECKPOINT_PATH" \
  $WANDB_ARG \
  "$@"