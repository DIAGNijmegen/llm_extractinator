# LLM Extractinator

![Overview of the LLM Data Extractor](docs/images/doofenshmirtz.jpg)

> ⚠️ This tool is a prototype in active development and may change significantly. Always verify results!

LLM Extractinator enables efficient extraction of structured data from unstructured text using large language models (LLMs). It supports configurable task definitions, CLI or Python usage, a point‑and‑click GUI Studio, and flexible data input/output formats.

📘 **Full documentation**: [https://DIAGNijmegen.github.io/llm\_extractinator/](https://DIAGNijmegen.github.io/llm_extractinator/)

---

## 🔧 Installation

### 1. Install **Ollama**

#### On **Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On **Windows** or **macOS**

Download the installer from:
[https://ollama.com/download](https://ollama.com/download)

---

### 2. Install the Package

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

> **Tip:** to be able to run the latest models, update the Ollama client regularly:
>
> ```bash
> pip install --upgrade ollama langchain-ollama
> ```

---

## 🖥️ Interactive Studio GUI 

Starting with **v0.5**, Extractinator ships with a Streamlit‑based Studio for designing, running and monitoring extraction tasks with zero code:

![Studio screenshot](docs/images/GUI.gif)

🚀 To run:

```bash
launch-extractinator  # opens http://localhost:8501 in your browser
```

Features

|                            |                                                                  |
| -------------------------- | ---------------------------------------------------------------- |
| 🗂️ Project Manager        | Create / select datasets, parsers and tasks with file previews   |
| 🔧 Parser Builder          | Visual Pydantic schema designer (nested models supported)        |
| 🚀 One‑click Runs          | Configure model, sampling & advanced flags, then watch live logs |
| 🛠️ Task JSON Wizard       | Step‑by‑step helper to generate valid `TaskXXX.json` files       |
| 🆘 Help bubbles everywhere | Inline docs so you never lose context                            |

The Studio is fully optional: anything you configure here can still be executed from the CLI or Python API.

---

## 🚀 Quick Usage

### GUI

```bash
launch-extractinator  # recommended for new users
```

### CLI

```bash
extractinate --task_id 001 --model_name "phi4"
```

### Python

```python
from llm_extractinator import extractinate

extractinate(task_id=1, model_name="phi4")
```

---

## 📁 Task Files

Each task is defined by a JSON file stored in `tasks/`.

Filename format:

```bash
TaskXXX_name.json
```

Example:

```json
{
  "Description": "Extract product data from text.",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_parser.py"
}
```

`Parser_Format` points to a `.py` file in `tasks/parsers/` that implements a Pydantic `OutputParser` model used to structure the LLM output.

---

## 🛠️ Visual Schema Builder (optional)

If you prefer a graphical approach to designing parsers, run:

```bash
build-parser
```

This starts the same builder embedded in the Studio, letting you assemble nested Pydantic models visually. Save the resulting `.py` file in `tasks/parsers/` and reference it via `Parser_Format`.

👉 Read the [parser docs](https://DIAGNijmegen.github.io/llm_extractinator/parser) for full details.

---

## 📄 Citation

If you use this tool, please cite: [https://doi.org/10.5281/zenodo.15089764](https://doi.org/10.5281/zenodo.15089764)

---

## 🤝 Contributing

We welcome pull requests! See the [contributing guide](https://DIAGNijmegen.github.io/llm_extractinator/contributing/) for details.
