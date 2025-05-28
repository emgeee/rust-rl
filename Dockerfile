# Use the official vLLM image as base
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Install additional Python dependencies needed for our server
RUN pip install --no-cache-dir \
    aiohttp>=3.8.0 \
    pyyaml>=6.0

# Copy source code and config
COPY src/ ./src/
COPY start_vllm_server.py ./
COPY multi_model_eval_config.yaml ./

# Expose vLLM server port
EXPOSE 8000

# Set default command to start the vLLM server
CMD ["python", "start_vllm_server.py"]