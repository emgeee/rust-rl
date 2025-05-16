#!/bin/bash
# General evaluation script that accepts a model name parameter

# Check if model name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model-name> [additional-args...]"
    echo "Example: $0 Qwen3-4B --sample-size 100"
    exit 1
fi

MODEL_NAME="$1"
shift

# Navigate to project root if running from scripts directory
if [[ $(basename "$PWD") == "scripts" ]]; then
    cd ..
fi

# Set Oxen key from environment if not already set
if [[ -z "${OXEN_KEY}" ]]; then
    echo "Warning: OXEN_KEY environment variable not set. You may need to authenticate manually."
fi

# Generate run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CLEAN_MODEL_NAME=$(echo "${MODEL_NAME}" | tr '/' '_')
RUN_NAME="eval_${CLEAN_MODEL_NAME}_${TIMESTAMP}"

# Run evaluation
python evaluate_script.py \
    --model-name "${MODEL_NAME}" \
    --run-name "${RUN_NAME}" \
    --oxen-repo "mgreen/rust-rl" \
    --predictions-path "results/{model_name}/predictions_code_and_tests.parquet" \
    --results-path "results/{model_name}/results_code_and_tests.parquet" \
    --results-plot "results/{model_name}/results_plot.png" \
    --run-build \
    --run-clippy \
    --run-test \
    --save-every 50 \
    --commit-results \
    "$@"

echo "Evaluation complete for ${RUN_NAME}"