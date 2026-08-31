# Start from your GPU-ready base
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + pip + common tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates bash zstd && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*


# install ollama
#
# Deliberately not pinned to a fixed version: new model architectures need a new
# Ollama, and this image exists to run new models.
#
# But a RUN layer is cached by its command *text*, which never changes on its
# own — so without a varying value here, a rebuild silently keeps whichever
# Ollama was current when this layer was first built. That is exactly how a
# newly released model ends up unsupported in an image you just rebuilt, and it
# is worse here because this layer sits above `COPY . /app`: a source change
# invalidates everything below it but never this.
#
# build.sh resolves the newest release and passes it in, so the layer rebuilds
# when — and only when — the version actually moves. Leave it empty and the
# install script picks the latest itself; set it to an older version to roll
# back or to reproduce an earlier image.
ARG OLLAMA_VERSION=
# Downloaded first rather than piped straight into sh: a pipeline's exit status
# is the last command's, so `curl … | sh` succeeds even when curl fails and
# hands sh an empty script. Printing the version at the end makes the build log
# say which Ollama actually went in.
RUN curl -fsSL https://ollama.com/install.sh -o /tmp/install-ollama.sh \
    && OLLAMA_VERSION="${OLLAMA_VERSION}" sh /tmp/install-ollama.sh \
    && rm /tmp/install-ollama.sh \
    && ollama --version

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
