# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Generative Reinforcement Learning from Policy Optimization (GRPO) to finetune the Qwen2.5-Coder-1.5B-Instruct model to generate better Rust code. The reinforcement learning approach uses actual Rust tooling (cargo build, clippy, test) as reward signals to improve code generation.

## Repository Structure

- **Main Scripts**:
  - `start_vllm_server.py`: Unified vLLM server script (traditional or dynamic mode)
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

**Option 1: Dynamic Server (Recommended)**
```bash
# Run full pipeline with automatic model loading
python run_evaluation.py --dynamic-server --all

# Run specific models with dynamic loading  
python run_evaluation.py --dynamic-server --all --models "claude-3-5-sonnet-20241022" "qwen-qwen2.5-coder-7b-instruct"
```

**Option 2: Traditional Server**
1. **Start Inference Server** (if using vLLM models):
   ```bash
   python start_vllm_server.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"
   ```

2. **Run Complete Evaluation Pipeline**:
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

**vLLM Server Management**
```bash
# Start dynamic server
python start_vllm_server.py --dynamic

# Start traditional server with specific model
python start_vllm_server.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Interactive monitoring mode (dynamic only)
python start_vllm_server.py --dynamic --interactive

# Check status
python start_vllm_server.py --status

# List available models
python start_vllm_server.py --list

# Stop server
python start_vllm_server.py --stop
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