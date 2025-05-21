# LLM Extractinator

![Overview of the LLM Data Extractor](docs/images/doofenshmirtz.jpg)

> ⚠️ This tool is a prototype in active development and may change significantly. Always verify results!

LLM Extractinator enables efficient extraction of structured data from unstructured text using large language models (LLMs). It supports configurable task definitions, CLI or Python usage, and flexible data input/output formats.

📘 **Full documentation**: [https://DIAGNijmegen.github.io/llm_extractinator/](https://DIAGNijmegen.github.io/llm_extractinator/)  

---

## 🔧 Installation

Install from PyPI:

```bash
pip install llm_extractinator
```

Or for development:

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator.git
cd llm_extractinator
pip install -e .
```

---

## 🚀 Quick Usage

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

Each task is defined using a JSON file stored in the `tasks/` directory.

Filename format:

```bash
TaskXXX_name.json
```

Example contents:

```json
{
  "Description": "Extract product data from text.",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_parser.py"
}
```

`Parser_Format` refers to a `.py` file in `tasks/parsers/` that defines a Pydantic `OutputParser` class used to structure the LLM output.

---

## 🛠️ Visual Schema Builder (Optional)

You can visually design the output schema using:

```bash
build-parser
```

This launches a web UI to create a Pydantic `OutputParser` model, which defines the structure of the extracted data. Additional models can be added and nested for complex formats.

The resulting `.py` file should be saved in:

```bash
tasks/parsers/
```

And referenced in your task JSON under the `Parser_Format` key.

👉 See [parser docs](https://DIAGNijmegen.github.io/llm_extractinator/parser) for full usage.

---

## 📄 Citation

If you use this tool, please cite:
[10.5281/zenodo.15089764](https://doi.org/10.5281/zenodo.15089764)

---

## 🤝 Contributing

We welcome contributions! See the full [contributing guide](https://<your_username>.github.io/llm_extractinator/contributing/) in the docs.
