# Multi-Model Rust Code Evaluation System

A comprehensive system for evaluating multiple AI models on their ability to generate high-quality Rust code.

## 🚀 Features

- **Multi-Provider Support**: API models (Claude, ChatGPT, Grok), vLLM-served models, and local HuggingFace models
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
- `OPENAI_API_KEY` environment variable for ChatGPT
- `XAI_API_KEY` environment variable for Grok

For vLLM models, ensure vLLM server is running on the configured URL.

## ⚙️ Configuration

Edit `multi_model_eval_config.yaml` to configure:

- **Models**: API models, vLLM models, local models
- **Generation Parameters**: temperature, top_p, max_tokens
- **Dataset Path**: Path to evaluation dataset 
- **Output Directory**: Where to save results
- **Evaluation Tools**: Which Rust tools to use

## 🏃 Usage

### Complete Pipeline
```bash
# Run inference + evaluation + visualization for all models
python multi_model_eval.py --config multi_model_eval_config.yaml --all

# Run for specific models only
python multi_model_eval.py --config multi_model_eval_config.yaml --all --models claude chatgpt

# Force re-run everything
python multi_model_eval.py --config multi_model_eval_config.yaml --all --force
```

### Individual Stages
```bash
# Run only inference
python multi_model_eval.py --config multi_model_eval_config.yaml --inference

# Run only evaluation (requires existing predictions)
python multi_model_eval.py --config multi_model_eval_config.yaml --evaluate

# Run only visualization (requires existing results)
python multi_model_eval.py --config multi_model_eval_config.yaml --visualize
```

### Validation
```bash
# Dry run to validate configuration
python multi_model_eval.py --config multi_model_eval_config.yaml --dry-run --all
```

## 📊 Output Structure

```
qwen3-rust-finetune/outputs/
├── claude/
│   ├── predictions.parquet    # Generated code responses
│   └── results.parquet       # Evaluation results
├── chatgpt/
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

## 🔧 Architecture

### Components

1. **Model Providers** (`src/rust_rl/models/`)
   - `APIModelProvider`: Claude, ChatGPT, Grok via APIs
   - `VLLMModelProvider`: Local models via vLLM server
   - `HuggingFaceModelProvider`: Direct HuggingFace transformers

2. **Evaluation Pipeline** (`src/rust_rl/evaluation/`)
   - `InferenceRunner`: Manages model inference
   - `EvaluationRunner`: Runs Rust toolchain evaluation
   - `MultiModelVisualizer`: Creates comparison charts

3. **Configuration** (`src/rust_rl/evaluation/config.py`)
   - YAML-based configuration management
   - Model and parameter specifications

### Evaluation Process

1. **Inference**: Generate Rust code for each prompt using configured models
2. **Evaluation**: Test generated code with `cargo build`, `cargo clippy`, `cargo test`
3. **Visualization**: Create comparison charts and summary statistics

## 🎯 Example Workflow

```bash
# 1. Validate configuration
python multi_model_eval.py --dry-run --all

# 2. Run inference for API models first
python multi_model_eval.py --inference --models claude chatgpt

# 3. Run evaluation on completed predictions
python multi_model_eval.py --evaluate

# 4. Add local model and complete pipeline
python multi_model_eval.py --all --models qwen-coder-1.5b

# 5. Generate final visualizations
python multi_model_eval.py --visualize
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