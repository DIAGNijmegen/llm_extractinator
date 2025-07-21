
# 📊 LLM Extractinator Studio

A simple, point-and-click interface to **design, test, and run extraction tasks**—no coding required. Built with Streamlit, the Studio lets you go from dataset to results in minutes.

---

## 🚀 How to Launch

Once installed, open your terminal and run:

```bash
launch-extractinator
```

This starts the app locally and opens it in your browser.

> 💡 By default, it runs on port `8501`. Use `--port` to choose a different one:
> 
> ```bash
> launch-extractinator --port 8502
> ```

---

## 🧭 What You Can Do

The Studio is organized into three main tabs:

---

## 1️⃣ Quick‑start

Run an existing Task JSON with a few clicks.

- **Select Task** – Dropdown shows all saved tasks in `tasks/`
- **Preview** – Expand to review the JSON
- **✅ Use this Task** – Loads the task and highlights the ▶️ Run tab

> 📝 Tip: Save and version-control tasks to share or rerun later.

---

## 2️⃣ Build Task

Create a new task in 3 guided steps:

### 🪄 Step-by-step Wizard

| Step             | What it Does                                           |
|------------------|--------------------------------------------------------|
| **1. Files**     | Pick your **dataset**, **parser**, and optional **examples** |
| **2. Description** | Add a title and choose which column has the input text |
| **3. Review & Save** | Preview the full JSON and save it to `tasks/`         |

**✨ Features:**

- ✅ Supports CSV and JSON datasets
- ✅ Drag and drop or browse files from your project
- ✅ Auto-preview for tables and code

---

## 🔧 Parser Builder

Still need to build your `OutputParser`? Click **🛠️ Open Parser Builder** in the sidebar.

- Design a Pydantic schema visually
- Set types and nesting
- Export as a ready-to-use `OutputParser` file

Once saved, place it in `tasks/parsers/` and reference it in your Task JSON.
For more details, see the [Parser documentation](parser.md).

---

## 3️⃣ Run

The final step—run your task and see live output.

### Key Features

- **Switch Tasks** easily without leaving the tab
- **Choose Model** – Any model supported by Ollama or your local setup
- **Enable Reasoning Mode** – Toggle special model settings if needed

### Advanced Settings (Optional)

Split into two subtabs:  
- **General** – run name, repetitions, seed, verbosity
- **Sampling & Limits** – temperature, top-k, context length, examples

### Run & Monitor

- Click **🚀 Run** to start
- Logs stream live in real-time
- Success or failure reported at the end

---

## 📁 Project Structure Overview

```
project-root/
├── data/            # Datasets (.csv/.json)
├── examples/        # Few-shot examples (optional)
├── tasks/
│   ├── Task001.json # Task configs
│   └── parsers/
│       └── your_parser.py  # OutputParser files
```

---

## 💡 Tips

- **Dark Mode** – Enable via ☰ → Settings → Theme
- **Reset Session** – Use the sidebar button to clear everything

---

## 🙌 Feedback & Contributions

Spotted a bug or want to help improve the Studio?  
Open an issue or PR on GitHub—code lives in `llm_extractinator/gui.py`.

We welcome all contributions! 🎉
