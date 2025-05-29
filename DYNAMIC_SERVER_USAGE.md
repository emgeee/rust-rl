# vLLM Server Usage

Instructions for using vLLM server for model inference.

## Key Features

- **Automatic Model Loading**: Models are loaded automatically when requested by inference code
- **No Pre-configuration**: No need to specify which model to load at startup
- **Sequential Processing**: Only one model loads at a time, with automatic switching between models
- **Queue Management**: Multiple requests for the same model are efficiently handled
- **Configuration-based**: Uses existing model configuration from `multi_model_eval_config.yaml`

## Usage

### Start vLLM Server Manually

Start the vLLM server directly with the model you want to use:

```bash
# Start vLLM server with specific model
python -m vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-Coder-7B-Instruct" --host 0.0.0.0 --port 8000

# With additional options
python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096
```

### Run Evaluation

Once the server is running, run the evaluation:

```bash
# Run full evaluation pipeline
python run_evaluation.py --all

# Run specific models
python run_evaluation.py --all --models "qwen-qwen2.5-coder-7b-instruct"
```

### Interactive Mode Commands

When running with `--interactive`, you can use these commands:

- `status` - Show server status
- `load <model>` - Load a specific model  
- `stop` - Stop current model
- `list` - List available models
- `quit` - Exit interactive mode

## How It Works

1. **Model Request**: When inference code requests a model, it checks if that model is currently loaded
2. **Automatic Loading**: If a different model is needed, the server automatically:
   - Stops the current vLLM server (if running)
   - Starts a new vLLM server with the requested model
   - Waits for the new server to be ready
3. **Request Processing**: Once loaded, the inference request proceeds normally
4. **Queue Management**: Multiple requests for the same model are handled efficiently

## Comparison with Traditional Approach

### Before (Traditional)
```bash
# Start server with specific model
python start_vllm_server.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Run evaluation (only works with pre-loaded model)
python run_evaluation.py --all --models "qwen-qwen2.5-coder-7b-instruct"

# To use different model, must restart server manually
python start_vllm_server.py --stop
python start_vllm_server.py --model "deepseek-ai/deepseek-coder-6.7b-instruct"
```

### After (Dynamic)
```bash
# Run evaluation with any/all models - they load automatically
python run_evaluation.py --dynamic-server --all
```

## Configuration

The dynamic server uses the same configuration file (`multi_model_eval_config.yaml`) as the traditional approach. All models listed under `models.vllm` are available for dynamic loading.

## Performance Notes

- **Model Loading Time**: Each model switch requires stopping/starting the vLLM server (typically 30-60 seconds)
- **Memory Efficiency**: Only one model is loaded in memory at a time
- **Sequential Processing**: Models are processed one at a time, which is more memory-efficient but slower than parallel processing

## Error Handling

The dynamic server includes robust error handling:
- Invalid model requests are rejected with clear error messages
- Failed model loads are retried automatically
- Server crashes are detected and handled gracefully
- Queue management prevents request conflicts