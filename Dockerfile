FROM python:3.11-slim

# system deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates bash && \
    rm -rf /var/lib/apt/lists/*

# install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY . /app

# install your package + python client
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir --upgrade ollama

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

ENTRYPOINT ["/entrypoint.sh"]
CMD ["app"]
