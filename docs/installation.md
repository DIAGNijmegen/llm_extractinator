# Installation

## Requirements

- Python 3.11 (recommended)
- A running [Ollama](https://ollama.com) instance
- (Optional) Conda or venv for isolation

## 1. Create environment

```bash
conda create -n llm_extractinator python=3.11
conda activate llm_extractinator
```

(You can use `python -m venv venv` instead.)

## 2. Install the package

```bash
pip install llm_extractinator
```

Or from source:

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator.git
cd llm_extractinator
pip install -e .
```

## 3. Make sure Ollama is running

Start the Ollama service first; the extractor will call it when you run a task. If Ollama is not running or the model name is wrong, extraction will fail.

## Next

- go to **Preparing Data** to see how your CSV/JSON should look
- or run `launch-extractinator` to explore the Studio
