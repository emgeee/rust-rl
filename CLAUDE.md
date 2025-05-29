# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Generative Reinforcement Learning from Policy Optimization (GRPO) to finetune the Qwen2.5-Coder-1.5B-Instruct model to generate better Rust code. The reinforcement learning approach uses actual Rust tooling (cargo build, clippy, test) as reward signals to improve code generation.

## Repository Structure

- **Main Scripts**:
  - `run_evaluation.py`: Run complete evaluation pipeline (inference + evaluation + visualization)
  - `process_data.py`: Marimo notebook for data processing and analysis
  - `train_script.py`: Training script for GRPO fine-tuning
  - `viz.py`: Marimo notebook for visualizing training results

- **Library Code**:
  - `src/rust_rl`: Main package containing modular components:
    - `dataset`: Dataset handling utilities
    - `evaluation`: Code evaluation, visualization, and orchestration utilities
      - `orchestration.py`: High-level pipeline orchestration and server management
      - `inference_runner.py`: Multi-model inference execution
      - `eval_runner.py`: Code evaluation using Rust tooling
      - `evaluator.py`: Core evaluation functions
      - `multi_model_visualize.py`: Comparison visualization across models
    - `experiment_utils`: Experiment and logging utilities
    - `prompts`: System prompts for Rust code generation
    - `reward_functions`: Reward functions for evaluating Rust code quality

- **Data and Outputs**:
  - `qwen3-rust-finetune/`: Directory containing datasets and outputs
  - `multi_model_eval_config.yaml`: Configuration for multi-model evaluation

## Environment Setup

```bash
# Install Python dependencies in development mode
pip install -e .

# Ensure Rust toolchain is installed for evaluation
rustup default stable
```

## Common Commands

### Start Marimo Server

```bash
./start_marimo.sh
```

This starts the Marimo server with a headless configuration on port 2700 with the token password "test1234".

### Evaluation Pipeline

1. **Run Complete Evaluation Pipeline**:
   ```bash
   # Run full pipeline (inference + evaluation + visualization)
   python run_evaluation.py --all
   
   # Run specific models only
   python run_evaluation.py --all --models "claude-3-5-sonnet-20241022" "qwen-qwen2.5-coder-7b-instruct"
   
   # Run individual stages
   python run_evaluation.py --inference-only
   python run_evaluation.py --eval-only
   python run_evaluation.py --viz-only
   ```

**Note**: The evaluation script will check if vLLM server is running when vLLM models are selected and will exit with an error if the server is not reachable.

**vLLM Server Management**
```bash
# Start server with specific model using vLLM directly
python -m vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-Coder-7B-Instruct" --host 0.0.0.0 --port 8000
```

### Training Pipeline

1. **Process Data**: Use the process_data.py Marimo notebook to prepare datasets
2. **Train Model**: Use train_script.py to finetune the model with GRPO
3. **Evaluate Model**: Use run_evaluation.py to test the trained model
4. **Visualize Performance**: Use viz.py to visualize training metrics

## Implementation Details

### GRPO Training

The project implements GRPO (Generative Reinforcement Learning from Policy Optimization) with:
- Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- Reward functions based on Rust compiler/tooling feedback
- Evaluation through actual compilation and testing

### Reward Functions

The training uses multiple reward signals to evaluate code quality:
1. Cargo build success (compilation)
2. Cargo clippy success (linting)
3. Cargo test success (testing)
4. Non-empty code check
5. Test block count check
6. Tests have assertions check

### Dependencies

The project relies on several key libraries:
- transformers and trl for model handling and training
- peft for efficient fine-tuning
- wandb for experiment tracking
- marimo for interactive notebooks