# Installation

There are two ways to install LLM Extractinator. **Docker is the recommended route** — it's the least fiddly and the closest thing to "it just works." A local install is the alternative if you want to develop against the package or can't use containers.

!!! tip "Recommended: Docker"
    The [Docker image](docker.md) bundles Python, Ollama, and the Studio into one GPU-ready container. The only thing you install is Docker itself — no Python environment to manage, no separate Ollama setup. If you just want to *use* the tool, start there:

    ```bash
    mkdir -p data examples tasks output ollama_models
    docker run --rm --gpus all \
      -p 127.0.0.1:8501:8501 -p 11434:11434 \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/examples:/app/examples \
      -v $(pwd)/tasks:/app/tasks \
      -v $(pwd)/output:/app/output \
      -v $(pwd)/ollama_models:/root/.ollama \
      lmmasters/llm_extractinator:latest
    ```

    This opens the Studio at [http://127.0.0.1:8501](http://127.0.0.1:8501) (drop `--gpus all` for CPU-only). See the [Docker guide](docker.md) for the full walkthrough, including Windows/PowerShell.

The rest of this page covers the **local install**.

---

## Local install

### Requirements

- **Python 3.10+** (3.11 recommended)
- A running **[Ollama](https://ollama.com)** instance (the local LLM backend)
- *(Optional)* Conda or `venv` to keep things isolated

### 1. Create an environment

```bash
conda create -n llm_extractinator python=3.11
conda activate llm_extractinator
```

(Or `python -m venv venv && source venv/bin/activate`.)

---

### 2. Install Ollama

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows / macOS:** download the installer from [ollama.com/download](https://ollama.com/download).

Ollama runs a small background service. Make sure it's running before you extract — on desktop, launching the Ollama app is enough; on a headless Linux box, `ollama serve` starts it.

!!! note "You don't need to pull models yourself"
    When you run a task, LLM Extractinator asks Ollama for the model you named and pulls it automatically on first use. (The exception is when you point at an *externally managed* server with `--ollama_host` — then the model must already be present there.)

---

### 3. Install the package

From PyPI:

```bash
pip install llm_extractinator
```

Or from source (handy if you want to hack on it):

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator.git
cd llm_extractinator
pip install -e .
```

Installing adds three commands to your environment:

| Command | What it does |
|---|---|
| `launch-extractinator` | Opens the Studio (Streamlit app) |
| `build-parser` | Opens just the Output Schema Builder |
| `extractinate` | Runs a task from the terminal |

---

### 4. Verify

Check the CLI is available:

```bash
extractinate --help
```

Then head to the **[Quickstart](quickstart.md)** for a complete first run, or launch the Studio to explore:

```bash
launch-extractinator
```

Running without a GPU? See [CPU-only hardware](cpu-only.md).
