# Rust RL

This project uses Generative Reinforcement Learning from Policy Optimization (GRPO) to finetune the Qwen2.5-Coder-1.5B-Instruct model to generate better Rust code. The reinforcement learning approach uses actual Rust tooling (cargo build, clippy, test) as reward signals to improve code generation.

The project also includes a comprehensive multi-model evaluation system for comparing AI models on their ability to generate high-quality Rust code.

## Project Structure

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
    - `models`: Model providers for API, vLLM, and HuggingFace models
    - `prompts`: System prompts for Rust code generation
    - `reward_functions`: Reward functions for evaluating Rust code quality

- **Data and Outputs**:
  - `qwen3-rust-finetune/`: Directory containing datasets and outputs
  - `multi_model_eval_config.yaml`: Configuration for multi-model evaluation

## Installation

```bash
# Install Python dependencies in development mode
pip install -e .

# Ensure Rust toolchain is installed for evaluation
rustup default stable
```

## Usage

### Docker Deployment

**Run vLLM Server with Docker:**
```bash
# Run in background
docker run -d \
  --name rust-rl-vllm \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  emgeee/rust-rl:latest

# Run interactively (see logs)
docker run -it \
  --name rust-rl-vllm \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  emgeee/rust-rl:latest

# Stop the container
docker stop rust-rl-vllm
docker rm rust-rl-vllm
```

**Or use Docker Compose:**
```bash
# Build and start the server
docker-compose up --build -d

# Stop the server
docker-compose down
```

### Start Marimo Server

```bash
./start_marimo.sh
```

This starts the Marimo server with a headless configuration on port 2700 with the token password "test1234".

### Multi-Model Evaluation Pipeline

**Dynamic Server Mode (Recommended)**
```bash
# Run full pipeline with automatic model loading
python run_evaluation.py --dynamic-server --all

# Run specific models with dynamic loading
python run_evaluation.py --dynamic-server --all --models "claude-3-5-sonnet-20241022" "qwen-qwen2.5-coder-7b-instruct"
```

**Traditional Server Mode**
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

### Multi-Model Evaluation Features

- **Multi-Provider Support**: API models (Claude, OpenAI, XAI, Google), vLLM-served models, and local HuggingFace models
- **Dynamic Server Support**: Automatic model loading on demand without manual server management
- **Comprehensive Evaluation**: Uses actual Rust toolchain (`cargo build`, `cargo clippy`, `cargo test`)
- **Rich Visualizations**: Comparison charts, heatmaps, and performance breakdowns
- **Resumable Pipeline**: Skip completed stages, force re-runs when needed
- **Progress Tracking**: Real-time progress bars and intermediate saves
- **Flexible Configuration**: YAML-based configuration for models and parameters

### Configuration

Edit `multi_model_eval_config.yaml` to configure:
- **Models**: API models, vLLM models, local models
- **Generation Parameters**: temperature, top_p, max_tokens
- **Dataset Path**: Path to evaluation dataset
- **Output Directory**: Where to save results
- **Evaluation Tools**: Which Rust tools to use

### Dependencies

The project relies on several key libraries:
- transformers and trl for model handling and training
- peft for efficient fine-tuning
- wandb for experiment tracking
- marimo for interactive notebooks

## License

This project is open source and inspired by approaches to training language models with reinforcement learning for code generation.
