# Installation

## Requirements

- Python 3.11 (recommended)
- A running [Ollama](https://ollama.com) instance
- (Optional) Conda or venv for isolation

Instead of installing locally, you can also run the entire application in a Docker container (see the [Docker guide](docker.md)).

## 1. Create environment

```bash
conda create -n llm_extractinator python=3.11
conda activate llm_extractinator
```

(You can use `python -m venv venv` instead.)

## 2. Install Ollama

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows / macOS:** download from [ollama.com](https://ollama.com/download)

Make sure the Ollama service is running before you try to extract.

## 3. Install the package

```bash
pip install llm_extractinator
```

Or from source:

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator.git
cd llm_extractinator
pip install -e .
```

## Next

- go to **Preparing Data** to see how your CSV/JSON should look
- or run `launch-extractinator` to explore the Studio
