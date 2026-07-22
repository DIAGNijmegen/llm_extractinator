from __future__ import annotations

# LLM Extractinator Studio
# -------------------------------------------------------------------
# A streamlined GUI for creating, managing, and running information
# extraction tasks using LLM Extractinator.
#
# The app follows a linear three-stage flow — Task -> Run -> Results — so the
# path from configuring a task to inspecting its output is always moving forward.
#
# NB: this must be a comment, not a module docstring. Because `from __future__`
# has to be the first statement, a triple-quoted string here would be a bare
# expression that Streamlit "magic" renders into the page.

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from schema_builder import render_schema_builder  # type: ignore
except ImportError:  # pragma: no cover
    render_schema_builder = lambda **_: st.info(
        "`schema_builder` missing – install or remove this call."
    )  # type: ignore[arg-type]

try:
    from theme import (  # type: ignore
        app_header,
        inject_theme,
        sidebar_brand,
        sidebar_flow,
        status_strip,
    )
except ImportError:  # pragma: no cover
    inject_theme = lambda: None  # type: ignore[assignment]
    app_header = lambda *a, **k: st.title(a[0] if a else "LLM Extractinator")  # type: ignore
    status_strip = lambda *a, **k: None  # type: ignore[assignment]
    sidebar_brand = lambda: st.markdown("### 🧩 Studio")  # type: ignore[assignment]
    sidebar_flow = lambda *a, **k: None  # type: ignore[assignment]

try:
    from importlib.metadata import version as _pkg_version

    APP_VERSION = _pkg_version("llm_extractinator")
except Exception:  # pragma: no cover
    APP_VERSION = ""

# ──────────────────── Global paths ───────────────────────────────
BASE_DIR = Path(os.environ.get("EXTRACTINATOR_BASE_DIR", Path.cwd()))
DATA_DIR = BASE_DIR / "data"
EX_DIR = BASE_DIR / "examples"
TASK_DIR = BASE_DIR / "tasks"
PAR_DIR = TASK_DIR / "parsers"
OUT_DIR = BASE_DIR / "output" / "run"

for _d in (DATA_DIR, EX_DIR, TASK_DIR, PAR_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────── Ollama helpers ─────────────────────────────
def _fetch_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """Return names of locally installed Ollama models, or [] if unreachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
        _EXCLUDE = {"nomic-embed-text"}
        return [
            m["name"]
            for m in data.get("models", [])
            if not any(m["name"].startswith(ex) for ex in _EXCLUDE)
        ]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        return []


def _fetch_model_thinking(model_name: str, host: str = "http://localhost:11434") -> bool:
    """Return True if the model advertises thinking capability via Ollama's show API."""
    try:
        data = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(
            f"{host}/api/show",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            info = json.loads(resp.read())
        return "thinking" in info.get("capabilities", [])
    except Exception:
        return False


# ──────────────────── Streamlit config ───────────────────────────
_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
st.set_page_config(
    page_title="LLM Extractinator Studio",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "🧩",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/DIAGNijmegen/llm_extractinator",
        "About": "LLM Extractinator Studio — structured extraction from unstructured text.",
    },
)
inject_theme()
app_header(
    "LLM Extractinator Studio",
    "Structured extraction from unstructured text, powered by local LLMs.",
    badge="Studio",
)

# ──────────────────── Sidebar ────────────────────────────────────
with st.sidebar:
    sidebar_brand()

    st.divider()

    st.markdown('<div class="lx-sb-label">Workflow</div>', unsafe_allow_html=True)
    sidebar_flow(
        [
            ("Task", "Configure or pick a task"),
            ("Run", "Choose a model and run it"),
            ("Results", "Explore the extracted output"),
        ]
    )

    st.divider()

    if st.button(
        "🔄 Reset session",
        help="Clear cached selections & reload with fresh state",
        width="stretch",
    ):
        for k in list(st.session_state.keys()):
            if k.startswith("task_") or k in {
                "data_path",
                "examples_path",
                "parser_path",
                "parser_choice",
                "parser_mode",
                "parser_select",
                "schema_builder",
                "schema_builder_last_saved",
                "input_field",
                "input_field_select",
                "task_ready",
                "task_choice",
                "task_mode",
                "model_name",
                "ollama_host",
                "view_run",
            }:
                del st.session_state[k]
        st.rerun()

    st.caption("Working directory")
    st.markdown(
        f"<div class='lx-workdir'>{BASE_DIR}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    _ver = f"v{APP_VERSION}" if APP_VERSION else "LLM Extractinator"
    st.markdown(
        '<div class="lx-sb-foot">'
        f"<span>{_ver}</span><span class='dot'>•</span>"
        '<a href="https://diagnijmegen.github.io/llm_extractinator/" target="_blank">Docs</a>'
        "<span class='dot'>•</span>"
        '<a href="https://github.com/DIAGNijmegen/llm_extractinator" target="_blank">GitHub</a>'
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────── Helpers ────────────────────────────────────


def preview_file(path: Path, n_rows: int = 5) -> None:
    """Render a lightweight preview of the given file inside the app."""
    if not path.exists():
        return
    try:
        match path.suffix.lower():
            case ".csv":
                st.dataframe(pd.read_csv(path).head(n_rows), width="stretch")
            case ".json":
                st.dataframe(pd.read_json(path).head(n_rows), width="stretch")
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


def list_output_runs() -> list[Path]:
    """Return run folders that contain a predictions file, newest first."""
    if not OUT_DIR.exists():
        return []
    runs = [
        p
        for p in OUT_DIR.iterdir()
        if p.is_dir() and (p / "nlp-predictions-dataset.json").exists()
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def classify_fields(record: dict) -> tuple[dict, dict]:
    """Split a record into scalar (input/meta) fields and structured (output) fields."""
    scalar: dict = {}
    structured: dict = {}
    for k, v in record.items():
        if k == "status":
            continue
        if isinstance(v, (list, dict)):
            structured[k] = v
        else:
            scalar[k] = v
    return scalar, structured


def render_task_summary(obj: dict, *, bordered: bool = True) -> None:
    """Show a friendly summary of a Task JSON object (instead of raw JSON).

    Set ``bordered=False`` when rendering inside another card to avoid nesting.
    """
    box = st.container(border=True) if bordered else st.container()
    with box:
        st.markdown(f"**Description**  \n{obj.get('Description', '—')}")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Dataset**  \n`{obj.get('Data_Path', '—')}`")
        c1.markdown(f"**Text column**  \n`{obj.get('Input_Field', '—')}`")
        c2.markdown(f"**Output schema**  \n`{obj.get('Parser_Format', '—')}`")
        c2.markdown(f"**Examples**  \n`{obj.get('Example_Path', '— (none)')}`")


# ──────────────────── Output-schema builder (modal) ─────────────


@st.dialog("🛠️ Build output schema", width="large")
def _schema_builder_dialog() -> None:
    """Visual schema builder in a modal so it doesn't crowd the task form."""
    st.caption(
        "Add the fields the model should extract, then click "
        "**💾 Save & use this schema**. It'll be selected for the task automatically."
    )
    saved = render_schema_builder(embed=True, use_sidebar=False, save_dir=PAR_DIR)
    if saved and saved.exists():
        st.success(f"Saved `{saved.name}` — click below to use it.")
        if st.button("Use this schema & close", type="primary", width="stretch"):
            # Hand the filename to parser_input via a pending flag so it can set
            # the selectbox's widget state *before* the widget is created on the
            # next run (setting it here would be too late — the widget already ran).
            st.session_state["parser_pending"] = saved.name
            st.session_state.pop("schema_builder_last_saved", None)
            st.rerun()


def parser_input() -> Path | None:
    """Pick an existing output schema, build a new one (modal), or upload a .py."""
    st.markdown("**Output schema**")
    st.caption(
        "Defines the fields the model should extract and their types "
        "(a Pydantic model whose top-level class is `OutputParser`)."
    )

    files = sorted(f.name for f in PAR_DIR.iterdir() if f.suffix.lower() == ".py")

    # A schema just built/saved in the modal announces itself here. Set the
    # selectbox's widget state now, before the widget is instantiated below.
    pending = st.session_state.pop("parser_pending", None)
    if pending and pending in files:
        st.session_state["parser_select"] = pending
        st.session_state["parser_choice"] = pending

    sel_col, btn_col = st.columns([3, 1])

    with btn_col:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button(
            "🛠️ Build new",
            width="stretch",
            key="open_builder",
            help="Design a schema visually in a pop-up",
        ):
            st.session_state.pop("schema_builder_last_saved", None)
            _schema_builder_dialog()

    path: Path | None = None
    with sel_col:
        if files:
            # Seed the widget's initial value from parser_choice (or the first
            # file) when it has no valid selection yet. Driving it purely through
            # session state avoids the "default value ignored" warning that comes
            # from passing `index` alongside a keyed widget.
            if st.session_state.get("parser_select") not in files:
                seed = st.session_state.get("parser_choice")
                st.session_state["parser_select"] = seed if seed in files else files[0]
            choice = st.selectbox(
                "Schema file",
                files,
                key="parser_select",
                help="Pick a schema from tasks/parsers/",
            )
            st.session_state["parser_choice"] = choice
            path = PAR_DIR / choice
        else:
            st.info("No schema files yet — click **Build new** to create one.")

    with st.expander("Upload a .py schema instead"):
        upload = st.file_uploader("Drag a .py file", type=["py"], key="parser_uploader")
        if upload is not None:
            up_path = PAR_DIR / upload.name
            up_path.write_bytes(upload.getbuffer())
            st.session_state["parser_choice"] = upload.name
            st.toast(f"Saved → {up_path.relative_to(BASE_DIR)}")
            path = up_path

    if path is not None and path.exists():
        with st.expander("Preview schema"):
            preview_file(path)
    return path


# ──────────────────── Task-building sub-flows ───────────────────


def use_existing_task() -> None:
    """Let the user pick a pre-configured Task file and mark it ready."""
    tasks = list(sorted(TASK_DIR.glob("Task*.json")))
    if not tasks:
        st.info("No task files found yet. Switch to **Build a new task** to create one.")
        return

    labels = [p.name for p in tasks]
    default = st.session_state.get("task_choice")
    idx = labels.index(default) if default in labels else 0
    choice = st.selectbox(
        "Task file",
        labels,
        index=idx,
        help="Pick a pre‑configured task to load",
    )
    path = TASK_DIR / choice
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        st.error(f"Could not read task file: {e}")
        return

    render_task_summary(obj)
    with st.expander("Raw JSON"):
        st.json(obj, expanded=False)

    if st.button("✅ Use this task", type="primary"):
        st.session_state.update({"task_choice": choice, "task_ready": True})
        st.toast("Task ready — open the ▶️ Run tab above.", icon="🎉")
        st.rerun()


def build_new_task() -> None:
    """Three-step form to compose and save a new Task JSON."""
    files_complete = False

    # ─── Step 1 · inputs ──
    with st.container(border=True):
        st.markdown('<span class="lx-step">Step 1 · inputs</span>', unsafe_allow_html=True)
        st.markdown("#### Select your data and output schema")
        data_path = pick_or_upload("Dataset (.csv / .json)", DATA_DIR, (".csv", ".json"))

        input_field = st.session_state.get("input_field")
        if data_path:
            try:
                df = (
                    pd.read_csv(data_path)
                    if data_path.suffix == ".csv"
                    else pd.read_json(data_path)
                )
                text_cols = [c for c in df.columns if df[c].dtype in ("object", "string")]
                if text_cols:
                    saved_col = st.session_state.get("input_field")
                    default_idx = text_cols.index(saved_col) if saved_col in text_cols else 0
                    input_field = st.selectbox(
                        "Text column",
                        text_cols,
                        index=default_idx,
                        key="input_field_select",
                        help="Which column contains the raw text the model should parse?",
                    )
                    st.session_state["input_field"] = input_field
                else:
                    st.error("No text columns detected.")
            except Exception as e:
                st.error(f"Failed to read dataset: {e}")

        st.divider()
        parser_path = parser_input()

        st.divider()
        examples_path = pick_or_upload(
            "Examples (.json) [optional]",
            EX_DIR,
            (".json",),
            optional=True,
        )

        if data_path and parser_path and input_field:
            files_complete = True
            st.success("✓ Inputs ready")

    # ─── Step 2 · describe ──
    if files_complete:
        with st.container(border=True):
            st.markdown('<span class="lx-step">Step 2 · describe</span>', unsafe_allow_html=True)
            st.markdown("#### Tell the model what to do")
            desc = st.text_area(
                "Task description",
                st.session_state.get("task_description", ""),
                help="Explain in plain language what this task should accomplish.",
            )
            id_col, _ = st.columns([1, 2])
            task_id = id_col.text_input(
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

    # ─── Step 3 · review & save ──
    if files_complete and "task_description" in st.session_state:
        with st.container(border=True):
            st.markdown('<span class="lx-step">Step 3 · review &amp; save</span>', unsafe_allow_html=True)
            st.markdown("#### Review and save the task")
            task_json_obj: dict[str, str] = {
                "Description": st.session_state["task_description"],
                "Data_Path": Path(data_path).name,
                "Input_Field": input_field,
                "Parser_Format": Path(parser_path).name,
            }
            if examples_path:
                task_json_obj["Example_Path"] = Path(examples_path).name

            render_task_summary(task_json_obj, bordered=False)

            task_json_path = TASK_DIR / f"Task{st.session_state['task_id']}.json"
            needs_write = (
                not task_json_path.exists()
                or json.loads(task_json_path.read_text()) != task_json_obj
            )
            if st.button(
                "💾 Save task",
                type="primary",
                disabled=not needs_write,
                help="Write the task JSON to disk and mark it ready to run",
            ):
                task_json_path.write_text(json.dumps(task_json_obj, indent=4))
                st.toast(f"Saved → {task_json_path.relative_to(BASE_DIR)}", icon="💾")
                st.session_state.update(
                    {"task_choice": task_json_path.name, "task_ready": True}
                )
                st.rerun()


# ──────────────────── Persistent status strip ──────────────────
# Reserve the strip's position here (just under the header) but fill it at the
# very end of the script — after the Run tab has had a chance to update the
# current model and any fresh run — so its values are never a step behind.
_strip_slot = st.container()


def _render_status_strip() -> None:
    runs = list_output_runs()
    task_ready = bool(st.session_state.get("task_ready"))
    task_label = st.session_state.get("task_choice", "None selected") if task_ready else "None selected"
    model = st.session_state.get("model_name")
    run_label = runs[0].name if runs else "None yet"
    status_strip(
        [
            ("Task", task_label, task_ready),
            ("Model", model or "Not chosen yet", bool(model)),
            ("Latest run", run_label, bool(runs)),
        ]
    )


# ──────────────────── Main flow: Task → Run → Results ──────────
tab_task, tab_run, tab_results = st.tabs(["📝 Task", "▶️ Run", "📊 Results"])

# 1️⃣ TASK
with tab_task:
    st.subheader("Configure a task")
    mode = st.radio(
        "How would you like to start?",
        ["Use an existing task", "Build a new task"],
        horizontal=True,
        key="task_mode",
        label_visibility="collapsed",
    )
    st.divider()
    if mode == "Use an existing task":
        use_existing_task()
    else:
        build_new_task()

# 2️⃣ RUN
# Defined as a function so an early `return` stops only this tab — using
# st.stop() inside a tab would halt the whole script and blank the other tabs.
def render_run_tab() -> None:
    st.subheader("Run the extractor")

    if not st.session_state.get("task_ready"):
        st.info("Choose or build a task on the **📝 Task** tab first.")
        return

    # ─── Task to run ──
    task_files = [p.name for p in sorted(TASK_DIR.glob("Task*.json"))]
    default_idx = next(
        (i for i, name in enumerate(task_files) if name == st.session_state.get("task_choice")),
        0,
    )
    task_choice = st.selectbox(
        "Task to run",
        task_files,
        index=default_idx,
        key="task_choice",
        help="Which task configuration to execute",
    )

    # ─── Model & sampling settings ──
    st.subheader("🧠 Model settings")
    ollama_host = st.text_input(
        "Ollama server URL (optional)",
        value=st.session_state.get("ollama_host", ""),
        placeholder="e.g. http://localhost:11500 — leave blank to auto-manage a local server",
        help=(
            "Point at an already-running Ollama server instead of having llm_extractinator "
            "start/stop its own. When set, the model must already be pulled on that server — "
            "llm_extractinator won't pull or unload models on a server it doesn't manage."
        ),
    ).strip()
    st.session_state["ollama_host"] = ollama_host
    _host_kwargs = {"host": ollama_host} if ollama_host else {}
    _installed_models = _fetch_ollama_models(**_host_kwargs)
    _OTHER = "Other (enter below)…"
    _options = _installed_models + [_OTHER] if _installed_models else []
    if _options:
        _selected = st.selectbox(
            "Model",
            options=_options,
            help="Models currently installed in your local Ollama instance.",
        )
    else:
        _selected = _OTHER
        st.caption(
            "No running Ollama instance found. "
            "Browse available models at [ollama.com/library](https://ollama.com/library)."
        )
    if _selected == _OTHER:
        model_name = st.text_input(
            "Model name",
            value="phi4",
            placeholder="e.g. qwen3:8b",
            help=(
                "Any Ollama model name. If not yet installed, Ollama will download it on first run. "
                "Browse [ollama.com/library](https://ollama.com/library)."
            ),
        )
    else:
        model_name = _selected
    st.session_state["model_name"] = model_name
    _thinking_detected = _fetch_model_thinking(model_name, **_host_kwargs)
    _tog_col1, _tog_col2 = st.columns(2)
    reasoning = _tog_col1.toggle(
        "Reasoning model?",
        value=_thinking_detected,
        key=f"reasoning_toggle_{model_name}",
        help=(
            "Auto-detected: this model supports thinking. Reasoning mode is enabled automatically. "
            "You can turn it off, but structured-output quality may suffer."
            if _thinking_detected else
            "Enable for thinking models (e.g. qwen3.5). "
            "Routes chain-of-thought tokens away from the JSON output so parsing stays reliable."
        ),
    )
    if _thinking_detected:
        _tog_col1.caption("⚡ Thinking model detected — auto-enabled")
    overwrite = _tog_col2.toggle(
        "Overwrite existing files",
        value=False,
        help="If the run folder already exists, delete & recreate it.",
    )

    with st.expander("⚙️ Advanced flags"):
        # — Run behaviour —
        st.markdown("**Run behaviour**")
        n_runs = st.number_input(
            "Number of runs",
            min_value=1,
            value=1,
            step=1,
            help="Repeat the task multiple times with identical settings.",
        )
        col1, col2 = st.columns(2)
        verbose = col1.checkbox(
            "Verbose output",
            help="Stream full raw model output & debug logs to the UI.",
        )
        seed_enabled = col2.checkbox(
            "Fix random seed",
            help="Fix RNG seed for reproducible generation.",
        )
        seed = st.number_input(
            "Seed value",
            min_value=0,
            value=0,
            disabled=not seed_enabled,
            help="Integer seed to initialise random generators.",
        )

        st.divider()

        # — Prompting —
        st.markdown("**Prompting**")
        num_examples = st.number_input(
            "Few-shot examples",
            min_value=0,
            value=0,
            help="Number of labelled examples to prepend to each prompt.",
        )
        ctx_mode = st.radio(
            "Context length strategy",
            options=["max", "split", "custom"],
            horizontal=True,
            help=(
                "**max** – set the context window to the minimum size needed for the longest input in your dataset. "
                "**split** – split the dataset into short/long subsets and run each with a right-sized context (recommended when report lengths vary a lot). "
                "**custom** – set an explicit token limit."
            ),
        )
        if ctx_mode == "custom":
            max_ctx = str(
                st.number_input(
                    "Custom context length (tokens)",
                    min_value=512,
                    value=4096,
                    step=512,
                )
            )
        else:
            max_ctx = ctx_mode

        st.divider()

        # — Sampling —
        st.markdown("**Sampling**")
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.0,
            0.05,
            help="Controls output randomness. 0.0 = deterministic; higher = more diverse.",
        )
        num_predict = st.number_input(
            "Max tokens to generate",
            min_value=1,
            value=512,
            help="Maximum number of tokens the model will produce per response.",
        )
        col4, col5 = st.columns(2)
        with col4:
            topk_on = st.checkbox(
                "Enable Top-k",
                help="Restrict sampling to the k most probable next tokens.",
            )
            top_k = st.number_input(
                "Top-k value",
                min_value=1,
                value=40,
                disabled=not topk_on,
            )
        with col5:
            topp_on = st.checkbox(
                "Enable Top-p",
                help="Nucleus sampling – keep the smallest token set whose cumulative probability exceeds p.",
            )
            top_p = st.slider(
                "Top-p value",
                0.0,
                1.0,
                0.9,
                0.05,
                disabled=not topp_on,
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
        if ollama_host:
            cmd += ["--ollama_host", ollama_host]

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
                r"^(pulling|downloading|transferring|verifying)\b", re.IGNORECASE
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

        # ─── Completion → hand off to Results ──
        if return_code == 0:
            st.success("Finished successfully ✅")
            runs_after = list_output_runs()
            if runs_after:
                newest = runs_after[0]
                st.session_state["view_run"] = newest.name
                try:
                    recs = json.loads(
                        (newest / "nlp-predictions-dataset.json").read_text(encoding="utf-8")
                    )
                    tot = len(recs)
                    ok = sum(1 for r in recs if r.get("status") == "success")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Records", tot)
                    m2.metric("✅ Successes", ok)
                    m3.metric("❌ Failures", tot - ok)
                except Exception:
                    pass
                st.info(
                    f"Saved to **{newest.name}** — open the **📊 Results** tab to explore it."
                )
        else:
            st.error("Failed ❌ — check the log above for details.")

# 3️⃣ RESULTS
def render_results_tab() -> None:
    st.subheader("Explore results")

    runs = list_output_runs()
    if not runs:
        st.info("No results yet. Run a task on the **▶️ Run** tab to generate output.")
        return

    run_labels = [p.name for p in runs]  # already newest-first
    default_run = st.session_state.get("view_run")
    idx = run_labels.index(default_run) if default_run in run_labels else 0
    selected_run = st.selectbox(
        "Run",
        run_labels,
        index=idx,
        help="Choose an output run to inspect (newest first)",
    )
    run_path = OUT_DIR / selected_run
    records: list[dict] = json.loads(
        (run_path / "nlp-predictions-dataset.json").read_text(encoding="utf-8")
    )

    # ─── Summary metrics ──
    total = len(records)
    n_ok = sum(1 for r in records if r.get("status") == "success")
    n_fail = total - n_ok
    col_total, col_ok, col_fail = st.columns(3)
    col_total.metric("Total records", total)
    col_ok.metric("✅ Successes", n_ok)
    col_fail.metric("❌ Failures", n_fail)

    # ─── Filters ──
    filt_col, search_col = st.columns([1, 2])
    status_filter = filt_col.radio(
        "Status",
        ["All", "Successes only", "Failures only"],
        horizontal=True,
    )
    search_text = search_col.text_input(
        "Search text", placeholder="Filter by any text in the record…"
    )

    # Apply filters
    filtered = records
    if status_filter == "Successes only":
        filtered = [r for r in filtered if r.get("status") == "success"]
    elif status_filter == "Failures only":
        filtered = [r for r in filtered if r.get("status") != "success"]
    if search_text.strip():
        q = search_text.strip().lower()
        filtered = [r for r in filtered if any(q in str(v).lower() for v in r.values())]

    if not filtered:
        st.warning("No records match the current filters.")
        return

    # ─── Build summary dataframe ──
    rows = []
    for r in filtered:
        sc, st_ = classify_fields(r)
        status_val = r.get("status", "")
        status_icon = "✅" if status_val == "success" else "❌"
        # Truncate the longest scalar string as the "text" preview
        text_preview = ""
        for v in sc.values():
            s = str(v)
            if len(s) > len(text_preview):
                text_preview = s
        row: dict = {
            "status": f"{status_icon} {status_val}",
            "text": text_preview[:120] + ("…" if len(text_preview) > 120 else ""),
        }
        for k, v in st_.items():
            if isinstance(v, list):
                row[k] = f"{len(v)} item{'s' if len(v) != 1 else ''}"
            elif isinstance(v, dict):
                row[k] = f"{len(v)} key{'s' if len(v) != 1 else ''}"
            else:
                row[k] = str(v)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # ─── Interactive table ──
    st.subheader(f"Records ({len(filtered)} shown)")
    event = st.dataframe(
        summary_df,
        width="stretch",
        hide_index=False,
        on_select="rerun",
        selection_mode="single-row",
    )

    # ─── Detail panel ──
    selected_rows = event.selection.rows if event else []
    if selected_rows:
        idx = selected_rows[0]
        record = filtered[idx]
        scalar_fields, structured_fields = classify_fields(record)
        status_val = record.get("status", "")

        st.divider()
        st.subheader(
            f"Record {idx} — {'✅ success' if status_val == 'success' else '❌ failure'}"
        )

        left, right = st.columns([2, 3])
        with left:
            st.markdown("**📄 Input / metadata**")
            for k, v in scalar_fields.items():
                st.markdown(f"**{k}**")
                st.markdown(str(v))

        with right:
            st.markdown("**📦 Extracted fields**")
            if structured_fields:
                for k, v in structured_fields.items():
                    st.markdown(f"**{k}**")
                    st.json(v, expanded=True)
            else:
                st.info("No structured output fields in this record.")
    else:
        st.caption("Select a row above to see the full record.")


# ──────────────────── Render Run & Results tabs ────────────────
with tab_run:
    render_run_tab()

with tab_results:
    render_results_tab()

# Fill the status strip now that model_name / latest run are up to date.
with _strip_slot:
    _render_status_strip()
