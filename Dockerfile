# Minimal vLLM server image with aggressive optimization
FROM vllm/vllm-openai:latest

# Set environment variables for flexible CUDA/CPU operation
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_CPU_KVCACHE_SPACE=4
ENV VLLM_ATTENTION_BACKEND=XFORMERS

# Install only required dependencies for the server
RUN pip install --no-cache-dir \
    pyyaml==6.0.2 \
    requests==2.31.0

# Aggressive cleanup to reduce image size
RUN pip cache purge && \
    find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -name '*.pyo' -delete && \
    find /usr/local -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -name 'tests' -type d -exec rm -rf {} + 2>/dev/null || true && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/* \
           /root/.cache \
           /root/.local \
           /usr/share/doc \
           /usr/share/man \
           /usr/share/locale \
           /usr/share/info \
           /var/cache/debconf/*

# Set working directory
WORKDIR /app

# Copy only essential server files
COPY start_vllm_server.py ./
COPY multi_model_eval_config.yaml ./
COPY src/rust_rl/__init__.py ./src/rust_rl/__init__.py
COPY src/rust_rl/evaluation/config.py ./src/rust_rl/evaluation/config.py
COPY src/rust_rl/evaluation/dynamic_model_server.py ./src/rust_rl/evaluation/dynamic_model_server.py
COPY src/rust_rl/evaluation/__init__.py ./src/rust_rl/evaluation/__init__.py

# Create minimal non-root user
RUN useradd -M -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod 755 /app
USER appuser

# Expose vLLM server port
EXPOSE 8000

# Lightweight health check that works for both CPU and GPU modes
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

# Create startup script that detects CUDA availability
RUN echo '#!/bin/bash\n\
if nvidia-smi &> /dev/null && [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "" ]; then\n\
    echo "CUDA detected, using GPU backend"\n\
    exec python start_vllm_server.py "$@"\n\
else\n\
    echo "No CUDA detected or disabled, using CPU backend"\n\
    exec python start_vllm_server.py --device cpu "$@"\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Start the vLLM server with automatic device detection
CMD ["/app/start.sh"]