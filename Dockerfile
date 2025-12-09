# Start from your GPU-ready base
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + pip + common tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates bash && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*


# install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY . /app

# install your package + python client
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --ignore-installed -e . && \
    pip install --no-cache-dir --ignore-installed --upgrade ollama

# streamlit settings (used in app mode)
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# add an entrypoint script
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    '' \
    '# start ollama in background (shared for both modes)' \
    'ollama serve &> /tmp/ollama.log &' \
    '' \
    'MODE="${1:-app}"' \
    '' \
    'if [ "$MODE" = "app" ]; then' \
    '  echo "Starting extractinator (Streamlit)..."' \
    '  exec launch-extractinator' \
    'elif [ "$MODE" = "shell" ]; then' \
    '  echo "Dropping into shell with llm_extractinator installed..."' \
    '  exec bash' \
    'else' \
    '  echo "Unknown mode: $MODE"' \
    '  echo "Use: app | shell"' \
    '  exit 1' \
    'fi' \
    > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8501
EXPOSE 11434

# Health check to ensure Ollama is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:11434/api/tags || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["app"]
