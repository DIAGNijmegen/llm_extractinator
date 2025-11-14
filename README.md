# LLM Extractinator

![Overview of the LLM Data Extractor](docs/images/doofenshmirtz.jpg)

> ⚠️ **Prototype**: this project is under active development. Interfaces, task formats, and directory names may change. Always inspect and validate the generated output before using it downstream.

LLM Extractinator lets you turn unstructured text (reports, notes, CSV text columns) into **structured, Pydantic-validated data** using local LLMs (via **Ollama**). It comes with:

- a **Studio** (Streamlit app) for no-code configuration
- a **CLI** (`extractinate`) for repeatable runs
- a **parser builder** (`build-parser`) to generate Pydantic models
- **task JSON files** to describe what to extract and from where

📚 **Docs**: https://diagnijmegen.github.io/llm_extractinator/

---

## 1. Prerequisites

This project expects a local LLM endpoint, currently **Ollama**.

### Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows / macOS:** download from https://ollama.com/download

Make sure the Ollama service is running before you try to extract.

---

## 2. Installation

We recommend using a fresh environment:

```bash
conda create -n llm_extractinator python=3.11
conda activate llm_extractinator
```

Install from PyPI:

```bash
pip install llm_extractinator
```

Or install from source:

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator.git
cd llm_extractinator
pip install -e .
```

---

## 3. Quick start

### A. Run with Docker (recommended for GPU systems)

You can run LLM Extractinator entirely via **Docker**, which includes Python, Ollama, and the Streamlit app in one container.

Make sure [Docker](https://docs.docker.com/get-docker/) is installed.  
For GPU acceleration, also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Create local directories that will be mounted inside the container:

```bash
mkdir -p data examples tasks output
```

**Windows / PowerShell:**
```powershell
# Remove `--gpus all` if you don't have a GPU
docker run --rm --gpus all `
  -p 127.0.0.1:8501:8501 `
  -p 11434:11434 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/examples:/app/examples `
  -v ${PWD}/tasks:/app/tasks `
  -v ${PWD}/output:/app/output `
  lmmasters/llm_extractinator:latest
```

**Linux / macOS:**
```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8501:8501 \
  -p 11434:11434 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/examples:/app/examples \
  -v $(pwd)/tasks:/app/tasks \
  -v $(pwd)/output:/app/output \
  lmmasters/llm_extractinator:latest
```

This launches the **Streamlit Studio** on [http://127.0.0.1:8501](http://127.0.0.1:8501).  
To open an interactive shell instead of the app, append `shell` to the command.

See [docs/docker.md](docs/docker.md) for full details.

---

### B. Studio (local install)

![Overview of the Studio](docs/images/GUI.gif)

Run the interactive UI:

```bash
launch-extractinator
```

This starts the Streamlit-based Studio (usually at http://localhost:8501) where you can:

- create/select datasets
- design a parser/output model
- create task JSON files
- run tasks and view logs

Anything you configure here can also be run from the CLI.

---

### C. CLI

Run a task by ID:

```bash
extractinate --task_id 1 --model_name "phi4"
```

Common options:

- `--task_dir tasks/` — where your task JSON files live
- `--data_dir data/` — where your CSV/JSON input lives
- `--output_dir output/` — where to write extracted results
- `--run_name my_first_run` — for easier tracking

See `docs/cli.md` for the full reference.

---

### D. Python

You can also call it from Python:

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="phi4",
    output_dir="output/",
)
```

---

## 4. Tasks

Tasks describe **what** to extract and **from where**. By convention they live in `tasks/` and are named:

```text
Task001_products.json
Task002_reports.json
...
```

A minimal task might look like:

```json
{
  "Description": "Extract product data from CSV",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_parser.py"
}
```

- `Data_Path`: relative to your data directory
- `Input_Field`: column/key that contains the text
- `Parser_Format`: Python file in `tasks/parsers/` that defines the Pydantic model

You can author these in Studio or by hand.

---

## 5. Parser builder

If you don’t want to write Pydantic models by hand:

```bash
build-parser
```

Export the generated model to:

```text
tasks/parsers/<name>.py
```

Then reference that filename in the task JSON.

---

## 6. Documentation

See the `docs/` directory (or the published site) for:

- data preparation
- parser UI
- CLI flags and examples
- Studio walkthrough
- manual/advanced running

---

## 7. Contributing

Pull requests are welcome. Please keep the CLI and the docs in sync (especially task naming and required fields).

---

## 8. Citation

If you use this tool, please cite:

https://doi.org/10.1093/jamiaopen/ooaf109
