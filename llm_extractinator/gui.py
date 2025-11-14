from __future__ import annotations

"""
LLM Extractinator Studio
---------------------------------------------------------
A streamlined GUI for creating, managing, and running information extraction tasks using LLM Extractinator.
"""

import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from schema_builder import render_schema_builder  # type: ignore
except ImportError:  # pragma: no cover
    render_schema_builder = lambda **_: st.info(
        "`schema_builder` missing – install or remove this call."
    )  # type: ignore[arg-type]

# ──────────────────── Global paths ───────────────────────────────
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
EX_DIR = BASE_DIR / "examples"
TASK_DIR = BASE_DIR / "tasks"
PAR_DIR = TASK_DIR / "parsers"

for _d in (DATA_DIR, EX_DIR, TASK_DIR, PAR_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────── Streamlit config ───────────────────────────
st.set_page_config(
    page_title="LLM Extractinator Studio",
    page_icon="🧩",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/your‑org/llm‑extractinator",
        "About": "Built with ❤️  &  Streamlit",
    },
)
PAGE = st.session_state.get("page", "studio")  # studio | builder

# ──────────────────── Sidebar – navigation ───────────────────────
with st.sidebar:
    st.title("🧩 Studio")

    if PAGE == "studio":
        if st.button(
            "🛠️ Open Parser Builder", help="Switch to the visual parser‑schema builder"
        ):
            st.session_state["page"] = "builder"
            st.rerun()
    else:
        if st.button("← Back to Studio", help="Return to the main Studio page"):
            st.session_state["page"] = "studio"
            st.rerun()

    if st.button("🔄 Reset Session", help="Clear cache & reload with fresh state"):
        for k in list(st.session_state.keys()):
            if k.startswith("task_") or k in {
                "data_path",
                "examples_path",
                "parser_path",
                "input_field",
                "task_ready",
                "task_choice",
            }:
                del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.caption(f"📁 Working directory: `{BASE_DIR}`")

    st.divider()
    st.caption("Built with ❤️ & Streamlit • Luc Builtjes 2025")


# ──────────────────── Helpers ────────────────────────────────────


def preview_file(path: Path, n_rows: int = 5) -> None:
    """Render a lightweight preview of the given file inside the app."""
    if not path.exists():
        return
    try:
        match path.suffix.lower():
            case ".csv":
                st.dataframe(pd.read_csv(path).head(n_rows), use_container_width=True)
            case ".json":
                st.dataframe(pd.read_json(path).head(n_rows), use_container_width=True)
            case ".py":
                st.code(path.read_text(), language="python")
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not preview file → {exc}")


def bash(cmd: list[str]):
    """Pretty‑print a bash command."""
    st.code(" ".join(map(str, cmd)), language="bash")


def pick_or_upload(
    label: str,
    dir_path: Path,
    suffixes: tuple[str, ...],
    *,
    optional: bool = False,
):
    """Reusable widget to pick an existing file or upload a new one."""

    st.markdown(f"**{label}**")
    modes = ["Use existing", "Upload new"] + (["Skip"] if optional else [])
    mode = st.radio(
        label,
        modes,
        horizontal=True,
        key=f"{label}_mode",
        label_visibility="collapsed",
        help="Choose whether to select an existing file, upload a new one, or skip this input.",
    )

    if mode == "Use existing":
        files = [f.name for f in dir_path.iterdir() if f.suffix.lower() in suffixes]
        if not files:
            st.info("No matching files in folder.")
            return None
        choice = st.selectbox(
            "Choose file",
            files,
            key=f"{label}_select",
            help="Pick a file from the project folder",
        )
        path = dir_path / choice
        preview_file(path)
        return path

    if mode == "Upload new":
        upload = st.file_uploader(
            "Drag a file",
            type=[s.strip(".") for s in suffixes],
            key=f"{label}_uploader",
            help="Drop a local file to add it to the project",
        )
        if upload is None:
            return None
        path = dir_path / upload.name
        path.write_bytes(upload.getbuffer())
        st.toast(f"Saved → {path.relative_to(BASE_DIR)}")
        preview_file(path)
        return path

    return None  # Skip


def next_free_task_id() -> str:
    """Return the next available 3‑digit Task ID (as a string)."""

    used = {
        int(m.group(1))
        for p in TASK_DIR.glob("Task*.json")
        if (m := re.match(r"Task(\d{3})", p.name))
    }
    for i in range(1000):
        if i not in used:
            return f"{i:03d}"
    raise RuntimeError("All 1000 Task IDs are taken!")


# ──────────────────── Parser Builder page ───────────────────────
if PAGE == "builder":
    st.header("🛠️ Parser Builder")
    render_schema_builder(embed=True)
    st.stop()

# ──────────────────── Main Studio page ───────────────────────────

tab_qs, tab_build, tab_run = st.tabs(["🚀 Quick‑start", "🛠️ Build Task", "▶️ Run"])

# 1️⃣ QUICK‑START TAB
with tab_qs:
    st.header("🚀 Quick‑start with an existing Task")
    tasks = sorted(TASK_DIR.glob("Task*.json"))
    if tasks:
        task_labels = [p.name for p in tasks]
        task_choice = st.selectbox(
            "Select a Task JSON",
            ["—"] + task_labels,
            index=0,
            help="Pick a pre‑configured Task file to load",
        )
        if task_choice != "—":
            path = TASK_DIR / task_choice
            st.json(json.loads(path.read_text()), expanded=False)
            if st.button(
                "✅ Use this Task", help="Load the selected Task into the Run tab"
            ):
                st.session_state.update(
                    {"task_choice": task_choice, "task_ready": True}
                )
                st.toast("Task selected → switch to ▶️ Run tab", icon="🎉")
    else:
        st.info("No Task files found. Build one in the next tab →")

# 2️⃣ BUILD‑TASK TAB
with tab_build:
    st.header("🛠️ Build a new Task file")

    files_complete = False

    # ─── Files sub‑step ──
    st.subheader("Step 1 • Select or upload your files")
    data_path = pick_or_upload("Dataset (.csv / .json)", DATA_DIR, (".csv", ".json"))

    input_field = None
    if data_path:
        try:
            df = (
                pd.read_csv(data_path)
                if data_path.suffix == ".csv"
                else pd.read_json(data_path)
            )
            text_cols = [c for c in df.columns if df[c].dtype == "object"]
            if text_cols:
                input_field = st.selectbox(
                    "Text column",
                    text_cols,
                    help="Which column contains the raw text the model should parse?",
                )
            else:
                st.error("No text columns detected.")
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")

    parser_path = pick_or_upload("Output parser (.py)", PAR_DIR, (".py",))

    examples_path = pick_or_upload(
        "Examples (.json) [optional]",
        EX_DIR,
        (".json",),
        optional=True,
    )

    if data_path and parser_path and input_field:
        files_complete = True
        st.success("✓ Files ready")

    # ─── Description sub‑step ──
    if files_complete:
        st.subheader("Step 2 • Describe the task")
        desc = st.text_area(
            "Task description",
            st.session_state.get("task_description", ""),
            help="Explain in plain language what this Task should accomplish.",
        )
        task_id = st.text_input(
            "3‑digit Task ID",
            st.session_state.get("task_id", next_free_task_id()),
            max_chars=3,
            help="Unique identifier (000‑999) – auto‑suggested if left blank.",
        )
        if desc.strip() and task_id.isdigit() and int(task_id) < 1000:
            st.session_state.update(
                {"task_description": desc.strip(), "task_id": f"{int(task_id):03d}"}
            )
            st.success("✓ Description captured")

    # ─── Review & save sub‑step ──
    if files_complete and "task_description" in st.session_state:
        st.subheader("Step 3 • Review & save")
        task_json_obj: dict[str, str] = {
            "Description": st.session_state["task_description"],
            "Data_Path": Path(data_path).name,
            "Input_Field": input_field,
            "Parser_Format": Path(parser_path).name,
        }
        if examples_path:
            task_json_obj["Example_Path"] = Path(examples_path).name

        st.json(task_json_obj, expanded=False)

        task_json_path = TASK_DIR / f"Task{st.session_state['task_id']}.json"
        needs_write = (
            not task_json_path.exists()
            or json.loads(task_json_path.read_text()) != task_json_obj
        )
        if st.button(
            "💾 Save Task", disabled=not needs_write, help="Write the Task JSON to disk"
        ):
            task_json_path.write_text(json.dumps(task_json_obj, indent=4))
            st.toast(f"Saved → {task_json_path.relative_to(BASE_DIR)}", icon="💾")
            st.session_state.update(
                {"task_choice": task_json_path.name, "task_ready": True}
            )

# 3️⃣ RUN‑TASK TAB
with tab_run:
    st.header("▶️ Run Extractinator")

    if not st.session_state.get("task_ready"):
        st.info("Choose a Task in 🛠️ Build Task or 🚀 Quick‑start first.")
        st.stop()

    # ─── Task selection ──
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
        "Task file",
        [p.name for p in task_files],
        index=default_idx,
        key="task_choice",
        help="Select which Task configuration to execute",
    )

    # ─── Model & sampling settings ──
    st.subheader("🧠 Model settings")
    model_name = st.text_input(
        "Model name",
        value="phi4",
        help="Name or path of the language model to run. For hosted services, use the provider‑specific ID.",
    )
    reasoning = st.toggle(
        "Reasoning model?",
        value=False,
        help="Enable chain‑of‑thought or other reasoning‑enhanced variants. May impact speed.",
    )

    with st.expander("⚙️ Advanced flags"):
        general_tab, sampling_tab = st.tabs(["General", "Sampling & limits"])

        # — General flags —
        with general_tab:
            run_name = st.text_input(
                "Run Name",
                value="run",
                help="Folder prefix where outputs will be saved.",
            )
            n_runs = st.number_input(
                "Number of Runs",
                min_value=1,
                value=1,
                step=1,
                help="Repeat the Task multiple times with identical settings.",
            )
            colA, colB = st.columns(2)
            verbose = colA.checkbox(
                "Verbose output",
                help="Stream full raw model output & debug logs to the UI.",
            )
            overwrite = colA.checkbox(
                "Overwrite existing files",
                help="If the run folder already exists, delete & recreate it.",
            )
            seed_enabled = colB.checkbox(
                "Set seed", help="Fix RNG seed for reproducible generation."
            )
            seed = colB.number_input(
                "Seed",
                min_value=0,
                value=0,
                disabled=not seed_enabled,
                help="Integer seed to initialise random generators.",
            )

        # — Sampling flags —
        with sampling_tab:
            temperature = st.slider(
                "Temperature",
                0.0,
                1.0,
                0.0,
                0.05,
                help="0.0 = deterministic; 1.0 = very diverse output.",
            )
            num_predict = st.number_input(
                "Number of tokens to predict",
                min_value=1,
                value=512,
                help="Maximum generation length per response (before stop tokens).",
            )
            colC, colD = st.columns(2)
            topk_on = colC.checkbox(
                "Top‑k", help="Restrict sampling to the k most probable tokens."
            )
            top_k = colC.number_input(
                "Top‑k value",
                min_value=1,
                value=40,
                disabled=not topk_on,
            )
            topp_on = colD.checkbox(
                "Top‑p",
                help="Nucleus sampling – dynamic token pool based on cumulative probability.",
            )
            top_p = colD.slider(
                "Top‑p value",
                0.0,
                1.0,
                0.9,
                0.05,
                disabled=not topp_on,
            )
            max_ctx = st.text_input(
                "Context Length",
                "max",
                help="Force a custom context window size integer – or leave as 'max' for automatic calculation. Set as 'split' for a dataset with some high variability of report length.",
            )
            num_examples = st.number_input(
                "Number of Examples",
                min_value=0,
                value=0,
                help="Few‑shot examples to prepend to each prompt.",
            )

    # ─── Launch button ──
    launch = st.button(
        "🚀 Run",
        type="primary",
        help="Start the extractinate process with the above settings",
    )

    # ─── Execute CLI when launched ──
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
        if seed_enabled:
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
        bash(cmd)

        with st.spinner("Running extractinate…"):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered
            )

            # One box for the full log, one for the current progress line
            log_box = st.empty()
            status_box = st.empty()

            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            log_lines: list[str] = []

            # heuristics for "ephemeral" progress lines (Ollama)
            progress_re = re.compile(
                r"^(pulling|downloading|transferring)\b", re.IGNORECASE
            )

            for raw_line in process.stdout:
                # strip ANSI and trailing newline
                clean_line = ansi_escape.sub("", raw_line.rstrip("\n"))

                if not clean_line.strip():
                    continue

                # If this looks like a progress bar line, show it only in status_box
                if progress_re.match(clean_line):
                    status_box.write(clean_line)
                else:
                    log_lines.append(clean_line)
                    # keep the log bounded
                    tail = log_lines[-200:]
                    log_box.code("\n".join(tail), language="bash")

            return_code = process.wait()

        st.success("Finished successfully ✅" if return_code == 0 else "Failed ❌")
