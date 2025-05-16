# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Generative Reinforcement Learning from Policy Optimization (GRPO) to finetune the Qwen2.5-Coder-1.5B-Instruct model to generate better Rust code. The reinforcement learning approach uses actual Rust tooling (cargo build, clippy, test) as reward signals to improve code generation.

## Repository Structure

- **Main Scripts**:
  - `train.py`: Marimo notebook for training models using GRPO
  - `eval.py`: Marimo notebook for evaluating generated Rust code
  - `inference.py`: Marimo notebook for running inference with trained models
  - `process_data.py`: Marimo notebook for data processing and analysis
  - `viz.py`: Marimo notebook for visualizing training results

- **Data and Outputs**:
  - `qwen3-rust-finetune/`: Directory containing datasets and outputs
  - `outputs/`: Training outputs

## Environment Setup

```bash
# Install Python dependencies using uv
uv pip install -e .

# Ensure Rust toolchain is installed for evaluation
rustup default stable
```

## Common Commands

### Start Marimo Server

```bash
./start_marimo.sh
```

This starts the Marimo server with a headless configuration on port 2700 with the token password "test1234".

### Training Pipeline

1. **Process Data**: Use the process_data.py Marimo notebook to prepare datasets
2. **Train Model**: Use train.py to finetune the model with GRPO
3. **Generate Predictions**: Use inference.py to generate Rust code with the model
4. **Evaluate Results**: Use eval.py to test generated code with Rust tools
5. **Visualize Performance**: Use viz.py to visualize training metrics

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
- oxenai and wandb for experiment tracking
- marimo for interactive notebooks