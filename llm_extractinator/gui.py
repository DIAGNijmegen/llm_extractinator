"""
LLM Extractinator Studio – streamlined wizard edition (v2-c)
-----------------------------------------------------------

• **Quick-start**: unchanged – jump straight to the Runner if a Task exists.
• **Cleaner Model Settings**:
  – Grouped into tabs *General* vs *Sampling & limits* inside the expander.
  – Help-bubbles on every control for instant guidance.
  – Temperature / top-p now use sliders for friendlier interaction.
• **API modernisation**: swapped `st.experimental_rerun()` ➜ `st.rerun()`.

-------------------------------- DO NOT DELETE ABOVE --------------------------------
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from schema_builder import render_schema_builder  # type: ignore
except ImportError:  # pragma: no cover
    render_schema_builder = lambda **_: st.info("`schema_builder` missing – install or remove this call.")  # type: ignore[arg-type]

# ──────────────────── Global paths ───────────────────────────────
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
EX_DIR = BASE_DIR / "examples"
TASK_DIR = BASE_DIR / "tasks"
PAR_DIR = TASK_DIR / "parsers"

# Ensure folders exist
for _d in (DATA_DIR, EX_DIR, TASK_DIR, PAR_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────── Streamlit config ───────────────────────────
st.set_page_config("LLM Extractinator Studio", "🧩", layout="wide")
PAGE = st.session_state.get("page", "wizard")  # wizard | builder

# Sidebar – page switcher
with st.sidebar:
    st.title("🧩 Studio")
    if PAGE == "wizard":
        if st.button("🛠️  Open Schema / Parser Builder"):
            st.session_state["page"] = "builder"
            st.rerun()
    else:
        if st.button("← Back to Wizard"):
            st.session_state["page"] = "wizard"
            st.rerun()
    st.markdown("---")
    st.caption(f"📁 Working directory: `{BASE_DIR}`")

# ──────────────────── Page: Schema / Parser Builder ──────────────
if PAGE == "builder":
    st.header("🛠️  Schema / Parser Builder")
    render_schema_builder(embed=True)
    st.stop()

# ──────────────────── Helpers ────────────────────────────────────


def preview_file(path: Path, n_rows: int = 5) -> None:
    """Quick tabular/code peek at CSV, JSON or Python files."""
    if not path.exists():
        return
    try:
        match path.suffix.lower():
            case ".csv":
                df = pd.read_csv(path)
                st.dataframe(df.head(n_rows), use_container_width=True)
            case ".json":
                df = pd.read_json(path, lines=False)
                st.dataframe(df.head(n_rows), use_container_width=True)
            case ".py":
                st.code(path.read_text(), language="python")
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not preview file → {exc}")


def code_block(cmd_list: list[str]) -> None:
    st.code(" ".join(map(str, cmd_list)), language="bash")


def pick_or_upload(
    label: str, dir_path: Path, suffixes: tuple[str, ...]
) -> Path | None:
    st.markdown(f"**{label}**")
    mode = st.radio(
        "",
        ["Use existing file", "Upload new file"],
        horizontal=True,
        key=f"{label}_mode",
    )
    if mode == "Use existing file":
        existing = [f.name for f in dir_path.iterdir() if f.suffix.lower() in suffixes]
        if not existing:
            st.info("No files found in this folder.")
            return None
        choice = st.selectbox("Choose file:", existing, key=f"{label}_select")
        path = dir_path / choice
        preview_file(path)
        return path
    upload = st.file_uploader(
        "Drag a file here:",
        type=[s.strip(".") for s in suffixes],
        key=f"{label}_uploader",
    )
    if upload is None:
        return None
    path = dir_path / upload.name
    path.write_bytes(upload.getbuffer())
    st.success(f"Saved → `{path.relative_to(BASE_DIR)}`")
    preview_file(path)
    return path


def next_free_task_id() -> str:
    used = {
        int(m.group(1))
        for m in (re.match(r"Task(\d{3})", p.name) for p in TASK_DIR.glob("Task*.json"))
        if m
    }
    for i in range(1000):
        if i not in used:
            return f"{i:03d}"
    raise RuntimeError("All 1000 Task IDs are taken!")


# ──────────────────── 0 • Quick-start ────────────────────────────
st.title("🧩 LLM Extractinator Studio · Wizard")

st.header("0️⃣  Quick-start with an existing Task (optional)")
all_tasks = sorted(TASK_DIR.glob("Task*.json"))
if all_tasks:
    task_names = [p.name for p in all_tasks]
    existing_choice = st.selectbox("Select a Task to run/edit:", ["—"] + task_names)
    if existing_choice != "—":
        existing_path = TASK_DIR / existing_choice
        st.success(f"Loaded `{existing_choice}` → ready to go!")
        st.json(json.loads(existing_path.read_text()), expanded=False)
        if st.button("🚀 Run this Task now"):
            st.session_state.update(
                {
                    "task_choice": existing_choice,
                    "task_ready": True,
                    "skip_wizard": True,
                }
            )
            st.rerun()
else:
    st.info("No Task files found yet – use the wizard below to create one.")

st.divider()

# ──────────────────── Setup for wizard/skipping ------------------
skip_wizard = bool(st.session_state.get("skip_wizard"))
steps = ["Files", "Description", "Task JSON", "Run"]
status = {s: True for s in steps} if skip_wizard else {s: False for s in steps}

# ──────────────────── Wizard steps 1-3 (hidden if skipped) ——
if not skip_wizard:
    # STEP 1 • Files
    st.header("1️⃣  Select or upload your files 🗂️")
    data_path = pick_or_upload("Dataset (.csv / .json)", DATA_DIR, (".csv", ".json"))
    examples_path = pick_or_upload("Examples (optional .json)", EX_DIR, (".json",))
    parser_path = pick_or_upload("Parser (optional .py)", PAR_DIR, (".py",))

    input_field = st.text_input(
        "Name of text column in the dataset",
        value="",
        placeholder="e.g. passage",
        disabled=data_path is None,
        help="Column containing the text passages you want to extract info from.",
    )

    if data_path and input_field.strip():
        status["Files"] = True
        st.session_state.update(
            {
                "data_path": data_path,
                "examples_path": examples_path,
                "parser_path": parser_path,
                "input_field": input_field.strip(),
            }
        )
        st.success("✓ Files step complete!")

    st.divider()

    # STEP 2 • Description
    if status["Files"]:
        st.header("2️⃣  Describe the task ✍️")
        desc = st.text_area(
            "Task description (Markdown supported)",
            value=st.session_state.get("task_description", ""),
            help="Explain in plain language what you want the model to extract.",
        )
        task_id = st.text_input(
            "3-digit Task ID",
            value=st.session_state.get("task_id", next_free_task_id()),
            max_chars=3,
            help="Unique numeric id (000-999) for the Task file.",
        )
        if desc.strip() and task_id.isdigit() and int(task_id) < 1000:
            st.session_state.update(
                {"task_description": desc.strip(), "task_id": f"{int(task_id):03d}"}
            )
            status["Description"] = True
            st.success("✓ Description step complete!")

    st.divider()

    # STEP 3 • Task JSON
    if status["Description"]:
        st.header("3️⃣  Review & save Task file 💾")
        task_obj: dict[str, str] = {
            "Description": st.session_state["task_description"],
            "Data_Path": Path(st.session_state["data_path"]).name,
            "Input_Field": st.session_state["input_field"],
        }
        if st.session_state.get("examples_path"):
            task_obj["Example_Path"] = Path(st.session_state["examples_path"]).name
        if st.session_state.get("parser_path"):
            task_obj["Parser_Format"] = Path(st.session_state["parser_path"]).name

        st.json(task_obj, expanded=False)
        task_json = TASK_DIR / f"Task{st.session_state['task_id']}.json"
        needs_write = (
            not task_json.exists() or json.loads(task_json.read_text()) != task_obj
        )
        if st.button("💾 Create / update Task file", disabled=not needs_write):
            task_json.write_text(json.dumps(task_obj, indent=4))
            st.toast(f"Saved → {task_json.relative_to(BASE_DIR)}")
            st.session_state.update({"task_ready": True, "task_choice": task_json.name})
            status["Task JSON"] = True
        elif not needs_write:
            status["Task JSON"] = True
    st.divider()

# ──────────────────── STEP 4 • Run (always available once task_ready) ——
if st.session_state.get("task_ready"):
    st.header("4️⃣  Run the Extractinator 🚀")
    task_files = sorted(TASK_DIR.glob("Task*.json"))
    default_idx = next(
        (
            i
            for i, p in enumerate(task_files)
            if p.name == st.session_state.get("task_choice")
        ),
        0,
    )
    task_choice = st.selectbox(
        "Choose a Task file",
        [p.name for p in task_files],
        index=default_idx,
        key="task_choice",
        help="Task JSON controls what to extract and how to parse the results.",
    )

    # ——— Model Settings form ———
    with st.form("runner_form"):
        st.subheader("🧠 Model Settings")
        model_name = st.text_input(
            "Model name",
            value="phi4",
            help="Model id recognised by the backend (HuggingFace name or local path)",
        )
        reasoning = st.toggle(
            "Reasoning model?",
            value=False,
            help="Adds the --reasoning_model flag for chain-of-thought capable variants.",
        )

        with st.expander("⚙️ Advanced flags"):
            general_tab, sampling_tab = st.tabs(["General", "Sampling & limits"])

            # — General tab
            with general_tab:
                col1, col2 = st.columns(2)
                run_name = col1.text_input(
                    "--run_name",
                    value="run",
                    help="Folder name for outputs (inside results directory)",
                )
                n_runs = col2.number_input(
                    "--n_runs",
                    min_value=1,
                    step=1,
                    value=1,
                    help="How many times to repeat the extraction job.",
                )
                col3, col4 = st.columns(2)
                verbose = col3.checkbox(
                    "--verbose", help="Print per-example details to console"
                )
                overwrite = col3.checkbox(
                    "--overwrite", help="Overwrite previous run with same name"
                )
                seed_on = col4.checkbox(
                    "Set --seed?", help="Fix random seed for reproducibility"
                )
                seed = col4.number_input(
                    "--seed",
                    min_value=0,
                    step=1,
                    value=0,
                    disabled=not seed_on,
                    help="Integer RNG seed (only used if checkbox above is ticked)",
                )

            # — Sampling & limits tab
            with sampling_tab:
                col5, col6 = st.columns(2)
                temperature = col5.slider(
                    "--temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Higher values ⇒ more diverse output (0 = deterministic)",
                )
                num_predict = col6.number_input(
                    "--num_predict",
                    min_value=1,
                    value=512,
                    step=1,
                    help="Max new tokens to generate (model-specific upper bound)",
                )
                col7, col8 = st.columns(2)
                topk_on = col7.checkbox(
                    "Set --top_k?", help="Activate top-k sampling filter"
                )
                top_k = col7.number_input(
                    "--top_k",
                    min_value=1,
                    value=40,
                    step=1,
                    disabled=not topk_on,
                    help="Restrict sampling to k most-probable tokens",
                )
                topp_on = col8.checkbox(
                    "Set --top_p?", help="Activate nucleus sampling filter"
                )
                top_p = col8.slider(
                    "--top_p",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    disabled=not topp_on,
                    help="Cumulative probability threshold for nucleus sampling",
                )
                max_ctx = sampling_tab.text_input(
                    "--max_context_len",
                    value="max",
                    help='Override model\'s default context window ("max") = leave unchanged',
                )
                num_examples = sampling_tab.number_input(
                    "--num_examples",
                    min_value=0,
                    value=0,
                    step=1,
                    help="If >0: few-shot prompt with that many dataset examples",
                )

        launch = st.form_submit_button("🚀 Run")

    # ——— Launch handler ———
    if launch:
        cmd = [
            "extractinate",
            "--task_id",
            re.match(r"Task(\d{3})", task_choice).group(1),
            "--model_name",
            model_name,
        ]
        if reasoning:
            cmd.append("--reasoning_model")
        if run_name != "run":
            cmd += ["--run_name", run_name]
        if n_runs != 1:
            cmd += ["--n_runs", str(n_runs)]
        if verbose:
            cmd.append("--verbose")
        if overwrite:
            cmd.append("--overwrite")
        if seed_on:
            cmd += ["--seed", str(seed)]
        if temperature:
            cmd += ["--temperature", str(temperature)]
        if topk_on:
            cmd += ["--top_k", str(top_k)]
        if topp_on:
            cmd += ["--top_p", str(top_p)]
        if num_predict != 512:
            cmd += ["--num_predict", str(num_predict)]
        if max_ctx != "max":
            cmd += ["--max_context_len", max_ctx]
        if num_examples:
            cmd += ["--num_examples", str(num_examples)]

        st.markdown("##### Final command")
        code_block(cmd)

        with st.spinner("Running llm_extractinator…"):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            output_box = st.empty()
            output_lines = []

            for line in process.stdout:
                output_lines.append(line)
                output_box.code("".join(output_lines), language="bash")
                time.sleep(0.05)

            process.stdout.close()
            return_code = process.wait()

        if return_code == 0:
            st.success("llm_extractinator finished successfully ✅")
        else:
            st.error("llm_extractinator failed ❌")

        status["Run"] = True

# ──────────────────── Sidebar progress tracker ───────────────────
with st.sidebar:
    st.header("Progress")
    for _s in steps:
        st.markdown(f"{'✅' if status[_s] else '⬜️'}  {_s}")
    st.markdown("---")
    st.caption("Built with ❤️  &  Streamlit • v2-c")
    st.caption("2025 Luc Builtjes • Apache-2.0 License")
