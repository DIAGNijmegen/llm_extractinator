# LLM Extractinator

![Overview of the LLM Data Extractor](images/doofenshmirtz.jpg)

> ⚠️ This tool is a prototype in active development and may change significantly. Always verify results!

LLM Extractinator enables efficient extraction of structured data from unstructured text using large language models (LLMs). It supports configurable task definitions, CLI or Python usage, and flexible data input/output formats.

📘 **Full documentation**: [https://<your_username>.github.io/llm_extractinator/](https://<your_username>.github.io/llm_extractinator/)  
_(replace with actual GitHub Pages link after deployment)_

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
extractinate --task_id 001 --model_name "mistral-nemo"
```

### Python

```python
from llm_extractinator import extractinate

extractinate(task_id=1, model_name="mistral-nemo")
```

---

## 📄 Citation

If you use this tool, please cite:
[10.5281/zenodo.15089764](https://doi.org/10.5281/zenodo.15089764)

---

## 🤝 Contributing

We welcome contributions! See the full [contributing guide](https://<your_username>.github.io/llm_extractinator/contributing/) in the docs.
