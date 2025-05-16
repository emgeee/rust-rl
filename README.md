# Rust RL

This project uses Generative Reinforcement Learning from Policy Optimization (GRPO) to finetune Qwen3 to write better Rust code.

Inspiration and much code taken from https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo

## Project Structure

The project is organized as follows:

```
rust-rl/
├── src/
│   └── rust_rl/
│       ├── __init__.py
│       ├── dataset/        # Dataset handling utilities
│       ├── oxen_utils/     # Oxen experiment and logging utilities
│       ├── prompts/        # System prompts for Rust code generation
│       └── reward_functions/ # Reward functions for evaluating Rust code
├── train.py               # Marimo notebook for training
├── eval.py                # Marimo notebook for evaluation
├── inference.py           # Marimo notebook for inference
├── process_data.py        # Marimo notebook for data processing
└── viz.py                 # Marimo notebook for visualization
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
2. Train the model using `train.py`
3. Generate predictions using `inference.py`
4. Evaluate results using `eval.py`
5. Visualize performance using `viz.py`

## License

This project is inspired by and contains code from the Oxen.ai blog post on training a Rust coder with reinforcement learning.