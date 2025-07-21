# 🚀 Installation Guide

Follow these steps to install and use the tool:

---

## 1. **Install Ollama**

### On **Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### On **Windows** or **macOS**

Download the installer from:  
[https://ollama.com/download](https://ollama.com/download)

---

## 2. **Install the Package**

Create a fresh conda environment:

```bash
conda create -n llm_extractinator python=3.11
conda activate llm_extractinator
```

Install the package via pip:

```bash
pip install llm_extractinator
```

Or from source:

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator.git
cd llm_extractinator
pip install -e .
```

To be able to run the latest models available, make sure to update the `ollama` package to the latest version:

```bash
pip install --upgrade ollama langchain-ollama
```
