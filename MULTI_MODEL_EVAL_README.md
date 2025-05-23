# Multi-Model Rust Code Evaluation System

A comprehensive system for evaluating multiple AI models on their ability to generate high-quality Rust code.

## 🚀 Features

- **Multi-Provider Support**: API models (Claude, OpenAI, XAI, Google), vLLM-served models, and local HuggingFace models
- **Dynamic Server Support**: Automatic model loading on demand without manual server management
- **Comprehensive Evaluation**: Uses actual Rust toolchain (`cargo build`, `cargo clippy`, `cargo test`)
- **Rich Visualizations**: Comparison charts, heatmaps, and performance breakdowns
- **Resumable Pipeline**: Skip completed stages, force re-runs when needed
- **Progress Tracking**: Real-time progress bars and intermediate saves
- **Flexible Configuration**: YAML-based configuration for models and parameters

## 📋 Requirements

Install dependencies:
```bash
pip install -e .
```

Additional requirements for API models:
- `ANTHROPIC_API_KEY` environment variable for Claude
- `OPENAI_API_KEY` environment variable for OpenAI models
- `XAI_API_KEY` environment variable for Grok
- `GOOGLE_API_KEY` environment variable for Gemini models

For vLLM models, you can either:
- Use dynamic server mode (recommended): Models are automatically loaded on demand
- Use traditional mode: Start vLLM server manually with specific model

## ⚙️ Configuration

Edit `multi_model_eval_config.yaml` to configure:

- **Models**: API models, vLLM models, local models
- **Generation Parameters**: temperature, top_p, max_tokens
- **Dataset Path**: Path to evaluation dataset
- **Output Directory**: Where to save results
- **Evaluation Tools**: Which Rust tools to use

## 🏃 Usage

### Complete Pipeline

**Dynamic Server Mode (Recommended)**
```bash
# Run full pipeline with automatic model loading
python run_evaluation.py --dynamic-server --all

# Run specific models with dynamic loading
python run_evaluation.py --dynamic-server --all --models "claude-3-5-sonnet-20241022" "qwen-qwen2.5-coder-7b-instruct"

# Force re-run everything
python run_evaluation.py --dynamic-server --all --force
```

**Traditional Mode**
```bash
# Start vLLM server first (for vLLM models)
python start_vllm_server.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Run inference + evaluation + visualization for all models
python run_evaluation.py --all

# Run for specific models only
python run_evaluation.py --all --models "claude-3-5-sonnet-20241022" "gpt-4o"
```

### Individual Stages
```bash
# Run only inference
python run_evaluation.py --inference-only

# Run only evaluation (requires existing predictions)
python run_evaluation.py --eval-only

# Run only visualization (requires existing results)
python run_evaluation.py --viz-only
```

### Validation
```bash
# Dry run to validate configuration
python run_evaluation.py --dry-run --all
```

## 📊 Output Structure

```
qwen3-rust-finetune/outputs/
├── claude-3-5-sonnet-20241022/
│   ├── predictions.parquet    # Generated code responses
│   ├── results.parquet       # Evaluation results
│   └── api_calls.jsonl       # API call logs (API models only)
├── gpt-4o/
│   ├── predictions.parquet
│   ├── results.parquet
│   └── api_calls.jsonl
├── qwen-qwen2.5-coder-7b-instruct/
│   ├── predictions.parquet
│   └── results.parquet
├── ...
└── comparison_charts/
    ├── overall_success_rates.png
    ├── tool_specific_performance.png
    ├── success_rate_heatmap.png
    ├── performance_breakdown.png
    └── model_comparison_summary.csv
```

**Directory Naming**: Model names are converted to directory-safe format (lowercase, "/" and "_" replaced with "-")

## 🔧 Architecture

### Components

1. **Model Providers** (`src/rust_rl/models/`)
   - `api_models.py`: Claude, OpenAI, XAI, Google via APIs
   - `vllm_models.py`: Local models via vLLM server
   - `hf_models.py`: Direct HuggingFace transformers

2. **Evaluation Pipeline** (`src/rust_rl/evaluation/`)
   - `orchestration.py`: High-level pipeline orchestration and server management
   - `inference_runner.py`: Multi-model inference execution
   - `eval_runner.py`: Code evaluation using Rust toolchain
   - `evaluator.py`: Core evaluation functions
   - `multi_model_visualize.py`: Comparison visualization across models
   - `dynamic_model_server.py`: Dynamic vLLM server management

3. **Configuration** (`src/rust_rl/evaluation/config.py`)
   - YAML-based configuration management
   - Unified configuration for all model types and parameters

### Evaluation Process

1. **Inference**: Generate Rust code for each prompt using configured models
2. **Evaluation**: Test generated code with `cargo build`, `cargo clippy`, `cargo test`
3. **Visualization**: Create comparison charts and summary statistics

## 🎯 Example Workflow

```bash
# 1. Validate configuration
python run_evaluation.py --dry-run --all

# 2. Run inference for API models first (dynamic mode)
python run_evaluation.py --dynamic-server --inference-only --models "claude-3-5-sonnet-20241022" "gpt-4o"

# 3. Run evaluation on completed predictions
python run_evaluation.py --eval-only

# 4. Add vLLM model and complete pipeline
python run_evaluation.py --dynamic-server --all --models "Qwen/Qwen2.5-Coder-7B-Instruct"

# 5. Generate final visualizations
python run_evaluation.py --viz-only
```

## 📈 Metrics

The system evaluates models on:

- **Build Success**: Code compiles with `cargo build`
- **Clippy Success**: Code passes linting with `cargo clippy`
- **Test Success**: Unit tests pass with `cargo test`
- **Overall Success**: All tools pass

Results include individual tool success rates and overall performance comparisons.

## 🔍 Troubleshooting

- **Missing API Keys**: Set required environment variables
- **vLLM Connection**: Ensure vLLM server is running and accessible
- **Memory Issues**: Reduce batch sizes or use smaller models
- **Import Errors**: Ensure all dependencies are installed with `pip install -e .`

## 🤝 Extending

To add new model providers:

1. Inherit from `ModelProvider` base class
2. Implement `generate()` and `is_available()` methods
3. Add provider to `ModelFactory.create_model()`
4. Update configuration schema

## 📝 Notes

- Results are saved periodically to enable resuming
- Failed generations are logged and marked as errors
- Visualization charts are saved as high-DPI PNG files
- Summary statistics are available in CSV format
