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
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from llm_extractinator.run_config import (  # type: ignore
        HARDWARE_PRESETS,
        RunSettings,
        THINKING_TEMPERATURE,
    )
    from llm_extractinator.utils import (  # type: ignore
        parse_test_run_size,
        run_folder_name,
    )
except ImportError:  # pragma: no cover - Streamlit runs this file as a script
    from run_config import (  # type: ignore
        HARDWARE_PRESETS,
        RunSettings,
        THINKING_TEMPERATURE,
    )
    from utils import parse_test_run_size, run_folder_name  # type: ignore

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


# ──────────────────── Hardware presets ───────────────────────────
_OTHER_MODEL = "Other (enter below)…"


_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Ollama's pull output redraws one line over and over. It belongs on the status
# line, not in the transcript, or it buries everything else.
_PULL_RE = re.compile(r"^(pulling|downloading|transferring|verifying)\b", re.IGNORECASE)

# tqdm writes "  40%|####      | 4/10 [00:12<00:18,  ...]" to stderr. The row
# counts are the only reliable progress signal we have — and as of 0.7.0 the
# callback counts *rows* rather than calls to the model, so a retried row no
# longer double-counts and this number can be trusted again.
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")

# The budget line the resolver logs, e.g.
# "context 4096 = 1119 prompt + 512 answer + 2465 headroom (fitted to the data)".
_BUDGET_RE = re.compile(r"context\s+[\d,]+\s*=\s*[\d,]+\s+prompt\b.*", re.IGNORECASE)


class RunProcess:
    """A running ``extractinate``, readable from the UI thread without blocking.

    The old Run tab iterated ``process.stdout`` inline. Streamlit runs one script
    thread per session, so that held the whole app: no other tab responded, and —
    the part that actually hurt on a real dataset — there was no way to offer a
    Stop button, because a click cannot be processed while the script is parked
    in a read loop. A full clinical run is hours of that.

    So the subprocess is drained by a daemon thread into a bounded deque, and the
    page polls. Nothing here touches ``st.*``: a thread without a script run
    context cannot use the Streamlit API, and the deque is shared by reference
    through session state instead.
    """

    MAX_LINES = 2000

    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.started = time.time()
        self.finished: float | None = None
        self.lines: deque[str] = deque(maxlen=self.MAX_LINES)
        self.status_line = ""
        self.budget_line = ""
        self.done_rows = 0
        self.total_rows = 0
        self.stopped_by_user = False
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self._process.stdout is not None
        for raw in self._process.stdout:
            line = _ANSI_RE.sub("", raw.rstrip("\n"))
            # tqdm redraws with \r, so one read can carry several frames.
            line = line.split("\r")[-1]
            if not line.strip():
                continue
            with self._lock:
                if match := _PROGRESS_RE.search(line):
                    self.done_rows, self.total_rows = int(match[1]), int(match[2])
                    self.status_line = line
                    continue
                if _PULL_RE.match(line):
                    self.status_line = line
                    continue
                if match := _BUDGET_RE.search(line):
                    self.budget_line = match[0]
                self.lines.append(line)
        self._process.wait()
        with self._lock:
            self.finished = time.time()

    # — read side, called from the script thread —

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "log": "\n".join(self.lines),
                "status_line": self.status_line,
                "budget_line": self.budget_line,
                "done_rows": self.done_rows,
                "total_rows": self.total_rows,
                "finished": self.finished,
            }

    @property
    def running(self) -> bool:
        return self._process.poll() is None

    @property
    def elapsed(self) -> float:
        with self._lock:
            end = self.finished
        return (end or time.time()) - self.started

    @property
    def return_code(self) -> int | None:
        return self._process.poll()

    def stop(self) -> None:
        """Ask the run to stop, then insist.

        ``terminate`` gives ``extractinate`` the chance to run its ``finally``
        and shut down the Ollama server it started; without that the server is
        left running and the next run inherits a loaded model. The kill is the
        fallback for a process that ignores it.
        """
        self.stopped_by_user = True
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()


def _apply_hw_preset(preset) -> None:
    """Fill the model and context-ceiling fields from a preset.

    Call this before those widgets are instantiated — Streamlit refuses a
    session-state write to a key whose widget has already rendered this run.

    The model is recorded as *pending* rather than forced into the free-text box:
    which server we are talking to may not be known yet, so whether this model is
    in the installed list can only be settled further down, once the host is.
    """
    st.session_state["preset_pending_model"] = preset.model
    st.session_state["model_name_free_text"] = preset.model
    st.session_state["context_cap_enabled"] = True
    st.session_state["context_cap_input"] = preset.context_cap


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

    sel_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")

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
                    if st.session_state.get("input_field_select") not in text_cols:
                        saved_col = st.session_state.get("input_field")
                        st.session_state["input_field_select"] = (
                            saved_col if saved_col in text_cols else text_cols[0]
                        )
                    input_field = st.selectbox(
                        "Text column",
                        text_cols,
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
_REASONING_MODES = {"Auto": None, "On": True, "Off": False}

# Every widget seeds through session state rather than a `value=`/`index=`
# default: passing a default alongside a key Streamlit already holds makes it log
# a warning and silently ignore the default. Collected here so the seeding is one
# readable block rather than a `setdefault` buried beside each control.
_WIDGET_DEFAULTS = {
    "scope_choice": "Full dataset",
    "test_run_rows": 5,
    "test_run_random": False,
    "context_cap_enabled": False,
    "context_cap_input": 8192,
    "chunking_enabled": False,
    "chunk_size_input": 50,
    "reasoning_mode": "Auto",
    "temp_manual": False,
    "num_predict_manual": False,
    "num_predict_input": 512,
    "topk_on": False,
    "top_k_input": 40,
    "topp_on": False,
    "num_examples_input": 0,
    "ctx_mode": "max",
    "custom_ctx_input": 4096,
    "n_runs_input": 1,
    "seed_enabled": False,
    "seed_input": 0,
    "overwrite": False,
    "verbose": False,
    "ollama_host": "",
}


# Widget values worth carrying across a run. Streamlit drops session-state
# entries whose widget did not render, and the run view renders none of the
# form — so without this, coming back from a run resets every setting.
_REMEMBERED_KEYS = tuple(_WIDGET_DEFAULTS) + (
    "task_choice",
    "model_dropdown_choice",
    "model_name_free_text",
    "hw_preset_choice",
)


def _remember_settings() -> None:
    """Copy the live widget values somewhere Streamlit will not garbage-collect.

    ``st.session_state["remembered"]`` is a plain dict, not a widget key, so it
    survives reruns in which the widgets themselves are absent.
    """
    st.session_state["remembered"] = {
        key: st.session_state[key] for key in _REMEMBERED_KEYS if key in st.session_state
    }


def _seed_widgets() -> None:
    """Restore last-known values, falling back to the defaults."""
    remembered = st.session_state.get("remembered", {})
    for key, value in _WIDGET_DEFAULTS.items():
        st.session_state.setdefault(key, remembered.get(key, value))
    for key in ("model_dropdown_choice", "model_name_free_text", "hw_preset_choice"):
        if key not in st.session_state and key in remembered:
            st.session_state[key] = remembered[key]


def _model_picker() -> str:
    """Server, model and preset — the three controls that decide *which model*.

    They live together because they are one decision, and because two of them
    write the third: entering a server URL repopulates the model list, and
    applying a preset sets the model outright. Scattering them across the page
    meant meeting them in the reverse of their dependency order — pick a model,
    then choose the server that decides which models exist, then apply a preset
    that overwrites the model again.
    """
    host = st.session_state.get("ollama_host", "")
    host_kwargs = {"host": host} if host else {}
    installed = _fetch_ollama_models(**host_kwargs)
    options = installed + [_OTHER_MODEL] if installed else []

    # A preset's model becomes a dropdown selection the moment we can see it on
    # the server — which may be several reruns later, once a host is entered.
    pending = st.session_state.get("preset_pending_model")
    if pending and options:
        st.session_state["model_dropdown_choice"] = (
            pending if pending in installed else _OTHER_MODEL
        )
        if pending in installed:
            del st.session_state["preset_pending_model"]

    if options:
        if st.session_state.get("model_dropdown_choice") not in options:
            st.session_state["model_dropdown_choice"] = options[0]
        selected = st.selectbox(
            "Model",
            options=options,
            key="model_dropdown_choice",
            help="Models currently installed on the Ollama server.",
        )
    else:
        selected = _OTHER_MODEL

    if selected == _OTHER_MODEL:
        st.session_state.setdefault("model_name_free_text", "phi4")
        model_name = st.text_input(
            "Model name",
            key="model_name_free_text",
            placeholder="e.g. qwen3:8b",
            help=(
                "Any Ollama model name; it is pulled on first run if missing. "
                "Browse [ollama.com/library](https://ollama.com/library)."
            ),
        )
    else:
        model_name = selected

    # The two things that change the list above, kept directly beneath it so
    # their effect is visible where it lands.
    where = host or "localhost:11434"
    found = f"{len(installed)} model(s)" if installed else "not reachable"
    st.caption(f"{where} · {found}")

    server_col, preset_col = st.columns(2)

    with server_col.popover("Server", width="stretch"):
        st.text_input(
            "Ollama server URL",
            key="ollama_host",
            placeholder="blank = manage a local server",
            help=(
                "Point at an already-running server instead of managing one. The "
                "model must already be pulled there — llm_extractinator does not "
                "pull or unload models on a server it does not own."
            ),
        )
        st.caption(
            "Changing this re-reads the model list above."
            if installed
            else "Nothing answered at this address, so the model name is not checked."
        )

    with preset_col.popover("Preset", width="stretch"):
        # An action, not a mode. Nothing here stays lit afterwards: a lasting
        # "Medium is active" marker stops being true the moment any field it
        # wrote is edited by hand. The toast reports what changed instead.
        choice = st.selectbox(
            "Recommended settings",
            list(HARDWARE_PRESETS),
            key="hw_preset_choice",
            help="A starting point sized for your GPU. Every field stays editable.",
        )
        preset = HARDWARE_PRESETS[choice]
        st.caption(f"Sets {preset.describe()}. {preset.note}")
        if st.button("Apply", width="stretch", key="apply_preset"):
            # Deferred to the top of the next run: the widgets this writes to
            # have already rendered by now, and Streamlit rejects the write.
            st.session_state["preset_to_apply"] = choice
            st.rerun()

    st.session_state["model_name"] = model_name
    return model_name


def _settings_panel(model_name: str, thinking_detected: bool) -> dict:
    """Everything that is not task, model or scope.

    Grouped by the question each answer belongs to rather than by which CLI flag
    it becomes. The previous grouping — *Repetition & logging* / *Prompting &
    context* / *Sampling* — put the context ceiling and the output budget in
    different sections, when they are the two halves of one window that the
    backend now resolves jointly. Anything the run decides for itself lives
    behind an explicit "set this manually" checkbox, so the panel shows overrides
    rather than a wall of numbers that may or may not be in force.
    """
    # "Execution" rather than "Run": the top-level tab is already called Run, and
    # two things with one name in the same view is exactly the kind of small
    # ambiguity that makes a page feel careless.
    hardware, generation, prompt, bookkeeping = st.tabs(
        ["Hardware", "Generation", "Prompt", "Execution"]
    )

    with hardware:
        st.caption(
            "What has to fit on the card, and what happens if the run is "
            "interrupted. The server and the presets live beside the model, "
            "since that is what they change."
        )
        cap_col, ceiling_col = st.columns([1, 2], vertical_alignment="bottom")
        context_cap_enabled = cap_col.checkbox(
            "Cap context length",
            key="context_cap_enabled",
            help=(
                "Upper bound on the context window. The window is still fitted to "
                "your data, but never grows past this — the context window, not "
                "the model weights, is usually what pushes a run out of VRAM."
            ),
        )
        context_cap = ceiling_col.number_input(
            "Context ceiling (tokens)",
            min_value=512,
            step=512,
            key="context_cap_input",
            disabled=not context_cap_enabled,
            help=(
                "The *whole* window: the prompt and the generated answer share "
                "it. Documents that no longer fit are truncated, and the run "
                "says how many."
            ),
        )
        chunk_col, rows_col = st.columns([1, 2], vertical_alignment="bottom")
        chunking_enabled = chunk_col.checkbox(
            "Process in chunks",
            key="chunking_enabled",
            help=(
                "Write each chunk's predictions to disk before starting the next, "
                "so an interrupted run can resume: completed chunks are skipped "
                "and everything is merged at the end."
            ),
        )
        chunk_size = rows_col.number_input(
            "Rows per chunk",
            min_value=1,
            step=1,
            key="chunk_size_input",
            disabled=not chunking_enabled,
        )

    with generation:
        st.caption("How the model produces text. Left alone, the run decides.")
        reasoning_mode = st.radio(
            "Reasoning",
            list(_REASONING_MODES),
            horizontal=True,
            key="reasoning_mode",
            help=(
                "Thinking models emit chain-of-thought before their answer. "
                "**Auto** lets the run decide once the model is pulled and can be "
                "inspected — right almost always. **On** forces it for a model "
                "that cannot be inspected yet; **Off** makes a thinking model "
                "answer directly."
            ),
        )
        reasoning = _REASONING_MODES[reasoning_mode]
        if reasoning is None:
            st.caption(
                f"Auto: **{model_name}** reports thinking support, so reasoning "
                "will be on."
                if thinking_detected
                else "Auto: settled at run time, once the model has been pulled."
            )
        thinking_now = reasoning if reasoning is not None else thinking_detected

        temp_manual = st.checkbox(
            "Set temperature manually",
            key="temp_manual",
            help=(
                "Left off, temperature is chosen for you: 0.0 (greedy) for an "
                "ordinary model, 0.6 for a thinking one — Qwen's own guidance is "
                "not to run a thinking model greedy, as it degrades output and "
                "can loop forever."
            ),
        )
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            THINKING_TEMPERATURE if thinking_now else 0.0,
            0.05,
            disabled=not temp_manual,
            help="0.0 = deterministic; higher = more diverse.",
        )
        if not temp_manual:
            st.caption(
                f"Auto: **{THINKING_TEMPERATURE}** — a thinking model must not run "
                "greedy."
                if thinking_now
                else "Auto: **0.0** — deterministic, which is what extraction wants."
            )
        elif thinking_now and temperature == 0:
            st.caption(
                "⚠️ Greedy decoding on a thinking model degrades output and can "
                "produce endless repetition. Use a seed for reproducibility instead."
            )

        # The budget sits next to the ceiling that has to contain it, and behind
        # the same manual/auto shape as temperature. Since 0.7.0 the backend
        # sizes this from the output schema and adds the reasoning allowance
        # itself, so a number here is an override, not a setting to be filled in.
        num_predict_manual = st.checkbox(
            "Set max output tokens manually",
            key="num_predict_manual",
            help=(
                "Left off, the budget is derived from your output schema — a "
                "three-field schema and a thirty-field one need very different "
                "amounts — and the chain-of-thought allowance is added on top for "
                "a thinking model."
            ),
        )
        num_predict = st.number_input(
            "Max output tokens",
            min_value=1,
            step=64,
            key="num_predict_input",
            disabled=not num_predict_manual,
            help="Maximum tokens the model may produce per row.",
        )
        if not num_predict_manual:
            st.caption(
                "Auto: sized from the output schema, floored at 512."
                + (
                    " A thinking model also gets the reasoning allowance on top."
                    if thinking_now
                    else ""
                )
            )

        topk_col, topp_col = st.columns(2)
        with topk_col:
            topk_on = st.checkbox(
                "Enable Top-k",
                key="topk_on",
                help="Restrict sampling to the k most probable next tokens.",
            )
            top_k = st.number_input(
                "Top-k value", min_value=1, key="top_k_input", disabled=not topk_on
            )
        with topp_col:
            topp_on = st.checkbox(
                "Enable Top-p",
                key="topp_on",
                help=(
                    "Nucleus sampling — keep the smallest token set whose "
                    "cumulative probability exceeds p."
                ),
            )
            top_p = st.slider(
                "Top-p value", 0.0, 1.0, 0.9, 0.05, disabled=not topp_on
            )

    with prompt:
        st.caption("What the model is shown for each row.")
        num_examples = st.number_input(
            "Few-shot examples",
            min_value=0,
            step=1,
            key="num_examples_input",
            help=(
                "Labelled examples prepended to each prompt. They are counted "
                "against the context window, answers included."
            ),
        )
        ctx_mode = st.radio(
            "Context length strategy",
            options=["max", "split", "custom"],
            horizontal=True,
            key="ctx_mode",
            help=(
                "**max** — fit the window to the longest input in the dataset. "
                "**split** — process short and long documents separately, each "
                "with a right-sized window (worth it when lengths vary a lot). "
                "**custom** — a fixed token count."
            ),
        )
        if ctx_mode == "custom":
            max_ctx = str(
                st.number_input(
                    "Fixed context length (tokens)",
                    min_value=512,
                    step=512,
                    key="custom_ctx_input",
                )
            )
        else:
            max_ctx = ctx_mode

    with bookkeeping:
        st.caption("Repetition, reproducibility and output handling.")
        n_runs = st.number_input(
            "Number of runs",
            min_value=1,
            step=1,
            key="n_runs_input",
            help="Repeat the task with identical settings.",
        )
        seed_col, seed_val_col = st.columns([1, 2], vertical_alignment="bottom")
        seed_enabled = seed_col.checkbox(
            "Fix random seed",
            key="seed_enabled",
            help="Makes sampling and random test-run selection reproducible.",
        )
        seed = seed_val_col.number_input(
            "Seed value", min_value=0, key="seed_input", disabled=not seed_enabled
        )
        overwrite = st.checkbox(
            "Overwrite existing output",
            key="overwrite",
            help=(
                "Re-run and replace existing output. Without this, a run whose "
                "output folder already exists is skipped and the previous results "
                "are kept."
            ),
        )
        verbose = st.checkbox(
            "Verbose output",
            key="verbose",
            help="Stream full raw model output and debug logs into the run log.",
        )

    return {
        "reasoning": reasoning,
        "n_runs": int(n_runs),
        "verbose": verbose,
        "overwrite": overwrite,
        "seed": int(seed) if seed_enabled else None,
        "chunk_size": int(chunk_size) if chunking_enabled else None,
        "num_examples": int(num_examples),
        "max_context_len": max_ctx,
        "max_context_cap": int(context_cap) if context_cap_enabled else None,
        "temperature": float(temperature) if temp_manual else None,
        "num_predict": int(num_predict) if num_predict_manual else None,
        "top_k": int(top_k) if topk_on else None,
        "top_p": float(top_p) if topp_on else None,
    }


def _render_run_view(run: RunProcess) -> None:
    """The page while a run is in flight, and immediately after it ends.

    The form is not drawn at all here. Leaving twenty controls on screen under a
    scrolling log was the single biggest reason this tab felt cluttered: during
    the one period when the user can change nothing, it showed them everything
    they could change.
    """
    st.markdown(f"**{'Running' if run.running else 'Finished'}** · `{run.cmd[2]}`")
    st.caption(" ".join(run.cmd))

    # The live view polls; the finished view is static. Keeping them on separate
    # branches is what stops the timer: a fragment with ``run_every`` set never
    # stops ticking on its own, and one that called ``st.rerun`` whenever it saw
    # a finished run would rerun the app forever.
    if run.running:
        _run_monitor(run)
    else:
        _run_panel(run)
        _render_run_outcome(run)


@st.fragment(run_every=1.0)
def _run_monitor(run: RunProcess) -> None:
    """The live view, refreshed once a second.

    A fragment rather than a ``sleep``-and-``st.rerun`` loop, because the tabs
    are built at module scope: a whole-app rerun every second would also re-run
    the Results tab, which reads the predictions file off disk. Polling one
    fragment keeps the cost proportional to what is actually changing.
    """
    _run_panel(run)
    if not run.running:
        # Hand back to the full script, which will take the static branch above
        # and render the outcome. This fires exactly once per run.
        st.rerun()


def _run_panel(run: RunProcess) -> None:
    """Progress, status, promoted budget line and log — live or finished."""
    state = run.snapshot()
    running = run.running

    head, stop = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        if state["total_rows"]:
            done, total = state["done_rows"], state["total_rows"]
            st.progress(
                min(done / total, 1.0),
                text=f"{done:,} of {total:,} rows · {_duration(run.elapsed)} elapsed",
            )
        else:
            st.progress(
                0.0,
                text=(
                    f"Starting up · {_duration(run.elapsed)} elapsed"
                    if running
                    else f"Ran for {_duration(run.elapsed)}"
                ),
            )
    if running and stop.button("Stop", width="stretch", help="Terminate the run"):
        run.stop()
        st.rerun()

    if state["status_line"]:
        st.caption(state["status_line"])
    if state["budget_line"]:
        # The one log line worth promoting: it is the whole point of 0.7.0 and
        # it would otherwise scroll past inside the transcript.
        st.info(state["budget_line"], icon=":material/straighten:")

    with st.expander("Run log", expanded=not running and run.return_code != 0):
        st.code(state["log"] or "…", language="bash")


def _render_run_outcome(run: RunProcess) -> None:
    """The result card. Ends the run by pointing at the thing to do next."""
    code = run.return_code
    if run.stopped_by_user:
        st.warning("Stopped. Partial output may have been written.")
    elif code == 0:
        st.success("Finished successfully.")
    else:
        st.error(f"Failed (exit code {code}) — see the run log above.")

    runs_after = list_output_runs()
    if code == 0 and runs_after:
        newest = runs_after[0]
        st.session_state["view_run"] = newest.name
        try:
            records = json.loads(
                (newest / "nlp-predictions-dataset.json").read_text(encoding="utf-8")
            )
            total = len(records)
            ok = sum(1 for r in records if r.get("status") == "success")
            m1, m2, m3 = st.columns(3)
            m1.metric("Records", f"{total:,}")
            m2.metric("Successes", f"{ok:,}")
            m3.metric("Failures", f"{total - ok:,}", delta=None)
            if total - ok:
                st.caption(
                    f"`failures.json` in **{newest.name}** has the model's own "
                    "output for each failed row."
                )
        except Exception:
            pass
        st.caption(f"Saved to **{newest.name}**. Open the Results tab to explore it.")

    if st.button("New run", type="primary"):
        st.session_state.pop("active_run", None)
        st.rerun()


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_run_tab() -> None:
    # A run in flight owns the page. Checked before anything is drawn, so the
    # form and the run view are never on screen together.
    active: RunProcess | None = st.session_state.get("active_run")
    if active is not None:
        _render_run_view(active)
        return

    if not st.session_state.get("task_ready"):
        st.info("Choose or build a task on the **Task** tab first.")
        return

    # A preset writes session state for widgets further down the script, which
    # Streamlit refuses once they have rendered. Applying it here — from a flag
    # set by last run's click — means the write always lands before any widget
    # exists, so controls can be placed by meaning rather than by write order.
    if pending := st.session_state.pop("preset_to_apply", None):
        preset = HARDWARE_PRESETS[pending]
        _apply_hw_preset(preset)
        # Say what moved. A preset silently rewriting the model you just picked
        # is the surprise; naming the change is what makes it an action rather
        # than something the page did behind your back.
        st.toast(
            f"Applied {pending} — model {preset.model}, "
            f"context ceiling {preset.context_cap:,}",
            icon=":material/check:",
        )

    if st.session_state.pop("pending_overwrite", False):
        st.session_state["overwrite"] = True

    _seed_widgets()

    # ── Launch bar ───────────────────────────────────────────────
    # Task, model and scope are what a routine run actually changes, so they sit
    # above the fold with the button. Everything else is an override.
    task_files = [p.name for p in sorted(TASK_DIR.glob("Task*.json"))]
    if st.session_state.get("task_choice") not in task_files:
        st.session_state["task_choice"] = task_files[0]

    task_col, model_col, scope_col = st.columns([2, 2, 2])

    with task_col:
        task_choice = st.selectbox(
            "Task", task_files, key="task_choice", help="Which task to execute."
        )

    with model_col:
        model_name = _model_picker()

    with scope_col:
        scope = st.radio(
            "Scope",
            ["Full dataset", "Test run"],
            horizontal=True,
            key="scope_choice",
            help=(
                "A test run processes a handful of rows — a quick check before "
                "committing to the full dataset. Its output lands in a separate "
                "folder, and the window is still sized from the full dataset so "
                "the check reflects the real run."
            ),
        )
        test_run_enabled = scope == "Test run"
        test_run_size, test_run_random = 5, False
        if test_run_enabled:
            rows_col, rand_col = st.columns([1, 1], vertical_alignment="bottom")
            test_run_size = rows_col.number_input(
                "Rows", min_value=1, step=1, key="test_run_rows"
            )
            test_run_random = rand_col.checkbox(
                "Random", key="test_run_random", help="Sample rather than take the first N."
            )

    ollama_host = st.session_state.get("ollama_host", "")
    thinking_detected = _fetch_model_thinking(
        model_name, **({"host": ollama_host} if ollama_host else {})
    )

    with st.expander("Settings", expanded=False):
        advanced = _settings_panel(model_name, thinking_detected)

    settings = RunSettings(
        task_file=task_choice,
        model_name=model_name,
        test_run_size=int(test_run_size) if test_run_enabled else None,
        test_run_random=test_run_random,
        ollama_host=ollama_host or None,
        **advanced,
    )

    # ── Review & launch ──────────────────────────────────────────
    status_strip(settings.summary())

    problem = _configuration_problem(settings)
    if problem:
        st.error(problem)

    stale = [] if settings.overwrite else _existing_output(settings)
    if stale:
        warn_col, fix_col = st.columns([4, 1], vertical_alignment="bottom")
        warn_col.warning(
            f"**{stale[0]}** already has results"
            + (f" (and {len(stale) - 1} more)" if len(stale) > 1 else "")
            + ". This run will be skipped and the existing output kept — so the "
            "Results tab would show the old predictions, not a new run."
        )
        if fix_col.button("Overwrite", width="stretch", help="Replace the existing output"):
            # Deferred like the preset: the `overwrite` checkbox has already
            # rendered by the time this warning is drawn, and Streamlit refuses a
            # session-state write to an instantiated widget.
            st.session_state["pending_overwrite"] = True
            st.rerun()

    launch_col, cmd_col = st.columns([1, 3], vertical_alignment="bottom")
    launch = launch_col.button(
        f"Run · test ({settings.test_run_size} rows)" if test_run_enabled else "Run",
        type="primary",
        width="stretch",
        disabled=problem is not None,
        help=(
            problem
            or "Start extractinate with the settings above"
        ),
    )
    with cmd_col.popover("Show command"):
        bash(settings.to_command())

    # Written on every render of the form, so whatever is on screen survives the
    # run view — during which none of these widgets exist and Streamlit would
    # otherwise discard their state.
    _remember_settings()

    if launch:
        st.session_state["active_run"] = RunProcess(settings.to_command())
        st.rerun()


def _configuration_problem(settings: RunSettings) -> str | None:
    """Why this configuration cannot run, or ``None``.

    Returned rather than rendered so the same answer can disable the button.
    Showing an error beside a live button invites the click it just argued
    against — and the backend now refuses these outright, so the click buys a
    traceback rather than a degraded run.
    """
    if settings.num_predict is None:
        return None
    if settings.max_context_cap is not None and settings.max_context_cap <= settings.num_predict:
        return (
            f"The context ceiling ({settings.max_context_cap:,}) has to be larger "
            f"than max output tokens ({settings.num_predict:,}) — the window holds "
            "the prompt *and* the answer, so this leaves no room for documents."
        )
    if settings.max_context_len not in ("max", "split"):
        fixed = int(settings.max_context_len)
        if fixed <= settings.num_predict:
            return (
                f"The fixed context length ({fixed:,}) has to be larger than max "
                f"output tokens ({settings.num_predict:,}); the window holds both "
                "the prompt and the answer."
            )
    return None


def _existing_output(settings: RunSettings) -> list[str]:
    """Run folders this configuration would write to that already hold results.

    The backend's guard is skip-not-clobber, which is the safe default but a
    silent one: without ``--overwrite`` the run appears to succeed and hands
    back the *previous* predictions. Better to say so before the click than to
    let someone read a stale file as a fresh result.
    """
    task_stem = Path(settings.task_file).stem
    hits = []
    for run_idx in range(settings.n_runs):
        folder = OUT_DIR / run_folder_name(task_stem, run_idx, settings.test_run_size)
        if (folder / "nlp-predictions-dataset.json").exists():
            hits.append(folder.name)
    return hits


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
        format_func=lambda name: (
            f"🧪 {name}" if parse_test_run_size(name) is not None else name
        ),
        help="Choose an output run to inspect (newest first). 🧪 marks test runs (subset of the dataset).",
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
