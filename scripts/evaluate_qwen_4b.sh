#!/bin/bash
# Helper script to evaluate Qwen3-4B model

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
RUN_NAME="eval_Qwen3_4B_${TIMESTAMP}"

# Run evaluation
python evaluate_script.py \
    --model-name "Qwen3-4B" \
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