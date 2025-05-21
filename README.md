# Rust RL

This project uses Generative Reinforcement Learning from Policy Optimization (GRPO) to finetune Qwen3 to write better Rust code.

This project is inspired by reinforcement learning approaches to improve code generation models.

## Project Structure

The project is organized as follows:

```
rust-rl/
├── src/
│   └── rust_rl/
│       ├── __init__.py
│       ├── dataset/        # Dataset handling utilities
│       ├── evaluation/     # Code evaluation utilities
│       ├── experiment_utils/ # Experiment and logging utilities
│       ├── prompts/        # System prompts for Rust code generation
│       └── reward_functions/ # Reward functions for evaluating Rust code
├── scripts/              # Helper scripts for common operations
├── train_script.py       # Standalone training script
├── evaluate_script.py    # Standalone evaluation script 
├── train.py              # Marimo notebook for training
├── eval.py               # Marimo notebook for evaluation
├── inference.py          # Marimo notebook for inference
├── process_data.py       # Marimo notebook for data processing
└── viz.py                # Marimo notebook for visualization
```

## Installation

```bash
# Install the package in development mode
pip install -e .
```

## Usage

### Start Marimo Server

```bash
./start_marimo.sh
```

### Training Pipeline

1. Process data using `process_data.py`
2. Train the model using either:
   - `train.py` (Marimo notebook)
   - `train_script.py` (Standalone script)
   - Helper scripts in the `scripts/` directory:
     - `scripts/train_1_5b.sh` (for the 1.5B model)
     - `scripts/train_4b.sh` (for the 4B model)
     - `scripts/resume_training.sh` (to resume from a checkpoint)
3. Generate predictions using `inference.py`
4. Evaluate results using either:
   - `eval.py` (Marimo notebook)
   - `evaluate_script.py` (Standalone script)
   - Helper scripts in the `scripts/` directory:
     - `scripts/evaluate_qwen_1_5b.sh` (for the 1.5B model)
     - `scripts/evaluate_qwen_4b.sh` (for the 4B model)
     - `scripts/evaluate_model.sh <model-name>` (for any model)
5. Visualize performance using `viz.py`

### Standalone Scripts

For training and evaluation outside of Marimo notebooks:

```bash
# Training
python train_script.py --model-name "Qwen/Qwen2.5-Coder-1.5B-Instruct" --output-dir "outputs"

# Evaluation
python evaluate_script.py --model-name "Qwen3-1.5B" --output-dir "outputs"

# Or use the helper scripts
./scripts/train_1_5b.sh
./scripts/evaluate_qwen_1_5b.sh
./scripts/evaluate_model.sh "CustomModel-1B" --sample-size 100
```

## License

This project is open source and inspired by approaches to training language models with reinforcement learning for code generation.