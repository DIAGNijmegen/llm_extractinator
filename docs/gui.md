# 📊 Extractinator Studio GUI

A point‑and‑click interface for **designing, testing and running extraction tasks** without writing a single line of code. Built with Streamlit, the Studio wraps the entire Extractinator workflow—dataset selection, parser creation, task definition and execution—into one coherent web app.

> **URL:** [http://localhost:8501](http://localhost:8501) (opened automatically by the launcher)

---

## 🚀 Launching the GUI

```bash
# From any activated environment where llm‑extractinator is installed
launch-extractinator
```

The command starts a local Streamlit server and opens a browser tab. You can safely keep the terminal open; all interactions happen in the web UI.

### Command‑line flags (optional)

| Flag           | Default | Description                               |                                     |
| -------------- | ------- | ----------------------------------------- | ----------------------------------- |
| `--port`       | `8501`  | Change the port if 8501 is busy.          |                                     |
| `--wide`       | off     | Force *wide* layout on ultrawide screens. |                                     |
| `--theme dark` | `light` | auto                                      | Override Streamlit theme detection. |

---

## 🗺️ UI Layout at a Glance

```bash
┌───────────────────────────────┐
│ Sidebar                       │  ← navigation & global actions
├───────────────────────────────┤
│   Tabs                        │  ← Quick‑start | Build Task | Run
│ ───────────────────────────── │
│   Active page content         │
└───────────────────────────────┘
```

### Sidebar

* **Studio / Builder switch** – jump between the main Studio and the stand‑alone *Parser Builder*.
* **Reset Session** – clears cached file paths and widget state.
* **Working directory** – shows where the app is reading and writing files.

---

## 1️⃣ Quick‑start Tab

Load an **existing Task JSON** and run it immediately.

1. **Select Task** – dropdown of all `tasks/TaskXXX*.json` files.
2. *Preview* – collapsible JSON view for a sanity check.
3. **✅ Use this Task** – sets the session and highlights the ▶️ Run tab.

> **Tip:** Store sample Tasks in *version control* so colleagues can reproduce your results.

---

## 2️⃣ Build Task Tab

A three‑step wizard that guarantees a valid Task file.

| Step              | Purpose                                                 | Key widgets                         |
| ----------------- | ------------------------------------------------------- | ----------------------------------- |
| 1 ‑ Files         | Point to **dataset**, **parser**, optional **examples** | file picker / uploader with preview |
| 2 ‑ Description   | Human‑readable summary & chosen **text column**         | textarea, dropdown                  |
| 3 ‑ Review & Save | Inspect assembled JSON → **💾 Save Task**               | JSON viewer, save button            |

### File pickers

* Works for CSV, JSON, or Python files.
* Supports drag‑and‑drop upload **and** browsing existing project files.
* Each picker has its own help bubble and renders a tiny preview (first 5 rows for tables, syntax‑highlight for code).

---

## 3️⃣ Run Tab

Where the magic (extraction) happens.

### Task selection

Switch between any saved Task without leaving the tab—the config widgets update instantly.

### Model settings

* **Model name** – anything supported by Ollama or your local library.
* **Reasoning model?** – toggles the `--reasoning_model` CLI flag.

### ⚙️ Advanced flags

Split into *General* and *Sampling & limits* subtabs. Ticking a checkbox instantly enables the associated input (no more greyed‑out frustration!).

* **General** – run name, repetitions, RNG seed, verbosity, overwrite.
* **Sampling** – temperature, top‑k/nucleus, context window, example count.

### Launch & live logs

* **🚀 Run** generates the exact CLI command.
* A Streamlit spinner wraps the process; stdout/stderr are streamed line‑by‑line into a live code block for real‑time monitoring.
* Success or failure is reported when the subprocess exits.

---

## 🔧 Parser Builder (embedded or stand‑alone)

Click **🛠️ Open Parser Builder** in the sidebar—or run `build-parser` from the CLI—to open a full‑screen visual Pydantic schema designer. Build nested models, set field types, and export a ready‑to‑import `OutputParser` Python file.

> Once exported, place the .py file in `tasks/parsers/` and reference it inside your Task JSON’s `Parser_Format`.

For more details, see the [Parser documentation](parser.md).

---

## 📂 Project Structure Cheatsheet

```bash
.
├── data/            # raw datasets (.csv/.json)
├── examples/        # few‑shot example files (optional)
├── tasks/
│   ├── Task001.json # each task config lives here
│   └── parsers/
│       └── product_parser.py

```

---

## 💡 Tips & Tricks

* **Dark mode** – toggle in Streamlit settings ( ☰ → Settings → Theme ).
* **Fast reload** – GUI watches for file changes; hit *Reset Session* if widgets get out of sync.
* **Remote servers** – run on a headless box with `launch-extractinator --server.address 0.0.0.0` and SSH tunnel port 8501.

---

## 📣 Feedback & Contributions

Found a bug, have an idea, or built a killer feature? Open an issue or PR on GitHub. The GUI lives in `llm_extractinator/studio/`; contributions welcome! 🎉
