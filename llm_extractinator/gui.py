import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------- Global Paths -----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXAMPLES_DIR = BASE_DIR / "examples"
TASKS_DIR = BASE_DIR / "tasks"
PARSERS_DIR = TASKS_DIR / "parsers"

for directory in [DATA_DIR, EXAMPLES_DIR, TASKS_DIR, PARSERS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ----------------- Page Config -----------------
st.set_page_config(page_title="LLM Extractinator Studio", page_icon="🧩", layout="wide")

# ----------------- Helper Functions -----------------


def preview_file(path: Path, n_rows: int = 5):
    """Display a small preview for CSV or JSON lines files."""
    if not path.exists():
        return
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=n_rows)
        else:
            df = pd.read_json(path, lines=True, nrows=n_rows)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview file: {e}")


def code_block(cmd_list):
    """Utility to show shell commands nicely."""
    st.code(" ".join(str(x) for x in cmd_list), language="bash")


# ----------------- Sidebar Navigation -----------------
with st.sidebar:
    st.title("🔗 Navigation")
    page = st.radio(
        label="Choose a workspace",
        options=["Task Builder", "Run Extractinator"],
        help="Switch between creating tasks and executing them.",
    )
    st.markdown("***")
    st.caption("Built with ❤️ & Streamlit • v2")

# ============================================================================
# TASK BUILDER
# ============================================================================
if page == "Task Builder":
    st.title("🛠️ Task Builder")
    st.write(
        "Package your **dataset**, **examples**, **custom parser**, and **metadata** "
        "into a single task file to feed into `llm_extractinator`."
    )

    t1, t2, t3 = st.tabs(["📂 Data & Examples", "🧩 Custom Parser", "📝 Task Metadata"])

    # ----------------- TAB 1 ▸ DATA & EXAMPLES -----------------
    with t1:
        st.subheader("Upload Dataset")
        with st.form("data_form", clear_on_submit=False):
            data_file = st.file_uploader(
                "Dataset file (CSV or JSON)",
                type=["csv", "json"],
                help="The raw dataset to be processed.",
            )
            input_field = st.text_input(
                "Name of text column in the dataset",
                help="Column that contains the text prompt or passage.",
            )
            examples_file = st.file_uploader(
                "examples.json (optional)",
                type=["json"],
                help="Few-shot examples for evaluation or prompting.",
            )
            submitted = st.form_submit_button("💾 Save")
        if submitted:
            if data_file is None or input_field.strip() == "":
                st.error("Please upload a dataset **and** specify the text column.")
            else:
                data_path = DATA_DIR / data_file.name
                data_path.write_bytes(data_file.getbuffer())
                st.success(f"Dataset saved → `{data_path}`")
                st.session_state["data_filename"] = data_file.name
                st.session_state["input_field_name"] = input_field

                preview_file(data_path)

                if examples_file is not None:
                    examples_path = EXAMPLES_DIR / examples_file.name
                    examples_path.write_bytes(examples_file.getbuffer())
                    st.success(f"Examples saved → `{examples_path}`")
                    st.session_state["examples_filename"] = examples_file.name
                else:
                    st.session_state.pop("examples_filename", None)

    # ----------------- TAB 2 ▸ PARSER -----------------
    with t2:
        st.subheader("Upload & Build Parser")
        with st.form("parser_form", clear_on_submit=False):
            parser_file = st.file_uploader(
                "parser.py",
                type=["py"],
                help="Custom Python script that turns model output into structured data.",
            )
            col1, col2 = st.columns(2)
            with col1:
                save_clicked = st.form_submit_button("💾 Save Parser")
            with col2:
                build_clicked = st.form_submit_button("⚙️ Build Parser")

        if save_clicked:
            if parser_file is None:
                st.error("Please choose a `parser.py` first.")
            else:
                parser_path = PARSERS_DIR / parser_file.name
                parser_path.write_bytes(parser_file.getbuffer())
                st.success(f"Parser saved → `{parser_path}`")
                st.session_state["parser_filename"] = parser_file.name

        if build_clicked:
            result = subprocess.run(["build-parser"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("`build-parser` executed successfully.")
            else:
                st.error(f"`build-parser` failed:\n{result.stderr}")

    # ----------------- TAB 3 ▸ TASK METADATA -----------------
    with t3:
        st.subheader("Describe Your Task")
        with st.form("meta_form"):
            task_description = st.text_area(
                "Task description (supports Markdown)",
                help="Explain what the task is and any special considerations.",
            )
            user_task_id = st.text_input(
                "Choose a 3-digit Task ID (000-999)",
                max_chars=3,
                help="Each task needs a unique numeric identifier.",
            )
            create_clicked = st.form_submit_button("🚀 Create Task file")

        if create_clicked:
            # Validation
            if "data_filename" not in st.session_state:
                st.error("Please save a dataset first in **Data & Examples**.")
            elif task_description.strip() == "":
                st.error("Enter a task description.")
            elif (
                user_task_id.strip() == ""
                or not user_task_id.isdigit()
                or int(user_task_id) > 999
            ):
                st.error("Task ID must be a number between 000 and 999.")
            elif (
                "input_field_name" not in st.session_state
                or st.session_state["input_field_name"].strip() == ""
            ):
                st.error("Specify the text column name in **Data & Examples**.")
            else:
                task_id_padded = f"{int(user_task_id):03d}"
                task_json_path = TASKS_DIR / f"Task{task_id_padded}.json"

                # Assemble task object
                task_obj = {
                    "Description": task_description,
                    "Data_Path": st.session_state["data_filename"],
                    "Input_Field": st.session_state["input_field_name"],
                }
                if "examples_filename" in st.session_state:
                    task_obj["Example_Path"] = st.session_state["examples_filename"]
                if "parser_filename" in st.session_state:
                    task_obj["Parser_Format"] = st.session_state["parser_filename"]

                task_json_path.write_text(json.dumps(task_obj, indent=4))
                st.success(f"Task file created → `{task_json_path}`")
                st.json(task_obj, expanded=False)

# ============================================================================
# EXTRACTINATOR RUNNER
# ============================================================================
else:
    st.title("🧰 LLM Extractinator Runner")
    st.write(
        "Configure and launch **llm_extractinator** on the task of your choice. "
        "Standard flags are shown by default; expand *Advanced settings* for full control."
    )

    task_files = sorted(TASKS_DIR.glob("Task*.json"))
    if not task_files:
        st.warning(
            "No task files found. Please create a task first on the **Task Builder** page."
        )
        st.stop()

    with st.container():
        task_choice = st.selectbox(
            "Select a task file",
            [f.name for f in task_files],
            help="Choose which task definition to pass to the runner.",
        )
        selected_task_path = TASKS_DIR / task_choice

        # Derive task ID
        match = re.match(r"Task(\d{3})", task_choice)
        if not match:
            st.error("Invalid task file name format. Expected 'TaskXXX.json'.")
            st.stop()
        task_id = match.group(1)

    with st.form("runner_form"):
        st.subheader("🧠 Model Settings")
        model_name = st.text_input(
            "Model name",
            value="phi4",
            help="Name understood by your backend (e.g. `gpt-4o`, `mistral-large`).",
        )
        reasoning_model = st.checkbox(
            "Model is a reasoning model?",
            value=False,
            help="Enable reasoning-specific handling (e.g. Deepseek-r1).",
        )

        with st.expander("⚙️ Advanced settings", expanded=False):
            st.markdown("###### Run Control")
            col1, col2 = st.columns(2)
            with col1:
                run_name = st.text_input("--run_name", value="run")
            with col2:
                n_runs = st.number_input("--n_runs", min_value=1, step=1, value=1)
            col3, col4 = st.columns(2)
            with col3:
                verbose = st.checkbox("--verbose")
                overwrite = st.checkbox("--overwrite")
            with col4:
                seed_active = st.checkbox("Set --seed?")
                seed_value = st.number_input(
                    "--seed",
                    min_value=0,
                    step=1,
                    value=0,
                    disabled=not seed_active,
                )

            st.divider()
            st.markdown("###### Model Configuration")
            col5, col6 = st.columns(2)
            with col5:
                temperature = st.number_input(
                    "--temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                )
            with col6:
                num_predict = st.number_input(
                    "--num_predict", min_value=1, step=1, value=512
                )
            col7, col8 = st.columns(2)
            with col7:
                top_k_active = st.checkbox("Set --top_k?")
                top_k_value = st.number_input(
                    "--top_k",
                    min_value=1,
                    step=1,
                    value=40,
                    disabled=not top_k_active,
                )
            with col8:
                top_p_active = st.checkbox("Set --top_p?")
                top_p_value = st.number_input(
                    "--top_p",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    disabled=not top_p_active,
                )
            max_context_len = st.text_input(
                "--max_context_len",
                value="max",
                help='"max", "split", or explicit integer',
            )

            st.divider()
            st.markdown("###### Task Input Control")
            num_examples = st.number_input(
                "--num_examples",
                min_value=0,
                step=1,
                value=0,
                help="How many examples to prepend to each prompt.",
            )

        launch_clicked = st.form_submit_button("🚀 Run extractinator")

    if launch_clicked:
        cmd = [
            "extractinate",
            "--task_id",
            task_id,
            "--model_name",
            model_name,
        ]
        if reasoning_model:
            cmd.append("--reasoning_model")

        # Advanced flags
        if run_name != "run":
            cmd.extend(["--run_name", run_name])
        if n_runs != 1:
            cmd.extend(["--n_runs", str(int(n_runs))])
        if verbose:
            cmd.append("--verbose")
        if overwrite:
            cmd.append("--overwrite")
        if seed_active:
            cmd.extend(["--seed", str(int(seed_value))])

        if temperature != 0.0:
            cmd.extend(["--temperature", str(temperature)])
        if top_k_active:
            cmd.extend(["--top_k", str(int(top_k_value))])
        if top_p_active:
            cmd.extend(["--top_p", str(top_p_value)])
        if num_predict != 512:
            cmd.extend(["--num_predict", str(int(num_predict))])
        if max_context_len != "max":
            cmd.extend(["--max_context_len", str(max_context_len)])
        if num_examples != 0:
            cmd.extend(["--num_examples", str(int(num_examples))])

        st.markdown("##### Final command")
        code_block(cmd)

        # Execute
        with st.spinner("Running llm_extractinator…"):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("llm_extractinator finished successfully ✅")
            st.text(result.stdout or "(no stdout)")
        else:
            st.error("llm_extractinator failed ❌")
            st.text(result.stderr or "(no stderr)")
