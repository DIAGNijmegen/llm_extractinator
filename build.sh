#!/usr/bin/env bash
# Build the image with the newest Ollama.
#
#   ./build.sh                          -> llm-extractinator:latest
#   ./build.sh myname:tag               -> a name of your choosing
#   OLLAMA_VERSION=0.32.0 ./build.sh    -> pin, to roll back or reproduce
#
# Why this exists rather than a plain `docker build`: the Ollama install is a
# RUN layer, and Docker caches those by command text alone. A plain rebuild
# therefore keeps whatever Ollama the layer was first built with, however old.
# Resolving the version here and passing it in makes the layer key change when
# the version does — so you get the newest Ollama, and rebuilds stay fast when
# nothing has moved.
set -euo pipefail

IMAGE="${1:-llm_extractinator:latest}"
LATEST_URL="https://api.github.com/repos/ollama/ollama/releases/latest"

version="${OLLAMA_VERSION:-}"
if [ -z "$version" ]; then
    version="$(curl -fsSL --max-time 15 "$LATEST_URL" 2>/dev/null \
        | sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -1)" || true
fi

if [ -z "$version" ]; then
    echo "Could not reach GitHub to resolve the latest Ollama version."
    echo "Building without a version, so the install script picks its own latest."
    echo "Note: if this layer is already cached, that means the Ollama in the"
    echo "image will NOT change. Use --no-cache if you need it refreshed."
    exec docker build -t "$IMAGE" .
fi

echo "Building $IMAGE with Ollama $version"
exec docker build --build-arg "OLLAMA_VERSION=$version" -t "$IMAGE" .
