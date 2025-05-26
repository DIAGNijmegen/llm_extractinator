from __future__ import annotations

import ast
import os
import textwrap
from typing import Any, Literal, Optional

import streamlit as st

################################################################################
# Page config ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

if "_PAGE_CONFIG_DONE" not in globals():
    try:
        st.set_page_config(
            page_title="Pydantic v2 Model Builder", layout="wide", page_icon="🛠️"
        )
    except Exception:
        pass
    _PAGE_CONFIG_DONE = True  # type: ignore # noqa: N806

st.title("🛠️ Pydantic Model Builder")
st.markdown(
    """
    Build and preview [Pydantic v2](https://docs.pydantic.dev/latest/) models without writing any code.

    **What can you do here?**
    - Create Python data models using a visual interface.
    - Add fields with built‑in types, collections, or nested models.
    - **Import** existing model files to continue editing them.
    - Export the resulting code to use in your projects.
    """
)

################################################################################
# Session state ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

if "models" not in st.session_state:
    st.session_state.models = {"OutputParser": []}

################################################################################
# Constants ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

PRIMITIVE_TYPES = ["str", "int", "float", "bool"]
SPECIAL_TYPES = ["list", "dict", "Any", "Literal"]

################################################################################
# Helper functions ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################


def _compose_type(
    field_type: str, *, subtype: str | None = None, lit_vals: str | None = None
) -> str:
    if field_type == "Literal" and lit_vals:
        return f"Literal[{', '.join(v.strip() for v in lit_vals.split(','))}]"
    if field_type in {"list", "dict"} and subtype:
        if field_type == "list":
            return f"list[{subtype}]"
        key_t, val_t = (subtype.split(":", 1) + ["str"])[0:2]
        return f"dict[{key_t.strip()}, {val_t.strip()}]"
    return field_type


def _detect_imports() -> list[str]:
    imports = {"from pydantic import BaseModel"}
    typing: set[str] = set()
    for fields in st.session_state.models.values():
        for f in fields:
            t = f["type"]
            if t.startswith("Optional["):
                typing.add("Optional")
                t = t.removeprefix("Optional[").removesuffix("]")
            if t == "Any" or "Any]" in t:
                typing.add("Any")
            if t.startswith("Literal["):
                typing.add("Literal")
            if t.startswith("list[") or t.startswith("dict["):
                typing.update({"list", "dict"})
    if typing:
        imports.add(f"from typing import {', '.join(sorted(typing))}")
    return sorted(imports)


def generate_code() -> str:
    code = _detect_imports() + ["\n"]
    for model_name, fields in reversed(st.session_state.models.items()):
        code.append(f"class {model_name}(BaseModel):")
        if not fields:
            code.append("    pass")
        else:
            for f in fields:
                line = f"    {f['name']}: {f['type']}"
                if f["type"].startswith("Optional["):
                    line += " = None"
                code.append(line)
        code.append("")
    return "\n".join(code)


def _parse_models_from_source(source: str) -> dict[str, list[dict[str, Any]]]:
    """Return a models‑dict like the GUI expects by static parsing of the file."""
    tree = ast.parse(source)
    models: dict[str, list[dict[str, Any]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        # ensure BaseModel inheritance
        if not any(
            isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases
        ):
            continue
        fields: list[dict[str, Any]] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field_name = stmt.target.id
                field_type = ast.get_source_segment(source, stmt.annotation)
                if stmt.value is None and field_type:
                    fields.append({"name": field_name, "type": field_type})
                elif field_type:
                    # optional (default value provided)
                    fields.append({"name": field_name, "type": field_type})
        models[node.name] = fields
    return models


################################################################################
# Sidebar ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

with st.sidebar:
    manager_tab, import_tab = st.tabs(["📦 Model Manager", "📂 Import file"])

    ############################################################################
    # 📦 Model Manager TAB
    ############################################################################
    with manager_tab:
        st.header("📦 Model Manager")
        with st.expander("ℹ️ What’s this app for?", expanded=False):
            st.markdown(
                textwrap.dedent(
                    """
                    This tool helps you **build Python data models** using [Pydantic](https://docs.pydantic.dev/latest/), a library for data validation and settings management.

                    **Key concepts:**
                    - A **model** defines a structure for data, like a form or schema.
                    - Each model has **fields** (like `name: str` or `age: int`) with types.
                    - You can use **primitive types** (`str`, `int`, etc.), **collections** (`list`, `dict`), or special types like `Optional`, `Literal`, or nested models.

                    You can:
                    - Add multiple models
                    - Define fields with various types
                    - Export the generated code

                    Use this to create OutputParser formats.
                    """
                )
            )

        new_model = st.text_input(
            "Enter new model name (e.g. User)",
            key="_new_model_name",
            help="Model names must begin with a capital letter and be valid Python identifiers.",
        )
        if st.button("➕ Add model", use_container_width=True):
            name = new_model.strip()
            if not name:
                st.warning("Please enter a model name.")
            elif not name.isidentifier() or not name[0].isupper():
                st.warning(
                    "Model names should start with a capital letter and be valid Python identifiers (letters, numbers, or underscores)."
                )
            elif name in st.session_state.models:
                st.warning(f"A model named **{name}** already exists.")
            else:
                st.session_state.models[name] = []
                st.success(f"Model **{name}** created.")

    ############################################################################
    # 📂 Import file TAB
    ############################################################################
    with import_tab:
        st.header("📂 Import existing models")
        uploaded_file = st.file_uploader(
            "Upload a Python file containing Pydantic BaseModel classes", type=["py"]
        )

        if uploaded_file:
            source_code = uploaded_file.read().decode("utf-8")
            if st.button("🔄 Load into editor", type="primary"):
                try:
                    imported_models = _parse_models_from_source(source_code)
                    if not imported_models:
                        st.warning(
                            "No BaseModel subclasses found in the uploaded file."
                        )
                    else:
                        st.session_state.models = imported_models
                        st.success(
                            "Models imported successfully! You can now edit them in the Design tab."
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Error while parsing file: {e}")

################################################################################
# Main tabs ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

design_tab, code_tab, export_tab = st.tabs(["🏗️ Design", "📝 Code", "💾 Export"])

with design_tab:
    for model_name in list(st.session_state.models.keys()):
        st.subheader(f"🧩 Define fields for `{model_name}`")
        cols = st.columns([2, 2, 1])
        field_name = cols[0].text_input(
            "Field name", key=f"name_{model_name}", help="e.g. username, age, is_active"
        )
        field_type = cols[1].selectbox(
            "Field type",
            PRIMITIVE_TYPES
            + SPECIAL_TYPES
            + [m for m in st.session_state.models if m != model_name],
            key=f"type_{model_name}",
            help="Choose the data type for this field. You can also select another model to nest it.",
        )
        is_optional = cols[2].checkbox(
            "Optional",
            key=f"opt_{model_name}",
            help="Check if this field is not required (i.e. can be None).",
        )

        sub_type = (
            st.text_input(
                "Element type (for list/dict)",
                key=f"subtype_{model_name}",
                help="For list, enter item type (e.g. int). For dict, use format key: value (e.g. str: float).",
            )
            if field_type in {"list", "dict"}
            else None
        )
        literal_vals = (
            st.text_input(
                "Literal values",
                key=f"lit_{model_name}",
                help="Comma-separated values (e.g. 'red, green, blue') for fixed choices.",
            )
            if field_type == "Literal"
            else None
        )

        if st.button("Add field", key=f"add_field_btn_{model_name}"):
            name = field_name.strip()
            if not name:
                st.warning("Please enter a field name.")
            elif any(f["name"] == name for f in st.session_state.models[model_name]):
                st.warning(f"Field **{name}** already exists in **{model_name}**.")
            elif field_type in {"list", "dict"} and not sub_type:
                st.warning("Please enter a subtype for the list or dict.")
            elif field_type == "Literal" and not literal_vals:
                st.warning("Please enter values for the Literal field.")
            else:
                final_type = _compose_type(
                    field_type, subtype=sub_type, lit_vals=literal_vals
                )
                if is_optional:
                    final_type = f"Optional[{final_type}]"
                st.session_state.models[model_name].append(
                    {"name": name, "type": final_type}
                )
                st.success(f"Field **{name}** added to **{model_name}**.")

        if st.session_state.models[model_name]:
            st.markdown("#### Fields")
            for i, field in enumerate(st.session_state.models[model_name]):
                cols = st.columns([3, 3, 3, 1])
                cols[0].markdown(f"`{field['name']}`")
                cols[1].markdown(f"`{field['type']}`")
                is_optional = field["type"].startswith("Optional[")
                cols[2].markdown("🔓 Optional" if is_optional else "🔒 Required")
                if cols[3].button(
                    "🗑️", key=f"del_{model_name}_{i}", help="Remove this field"
                ):
                    del st.session_state.models[model_name][i]
                    st.rerun()
        else:
            st.info("No fields yet. Add one using the inputs above.")

with code_tab:
    st.subheader("📝 Generated Python Code")
    st.markdown(
        "This is the source code for your models. You can copy it or download it as a `.py` file for use in your Python project."
    )
    source = generate_code()
    st.code(source, language="python")

with export_tab:
    st.subheader("📄 Save as file name (without .py)")

    st.text_input(
        "Filename",
        value="output_parser",
        key="export_file_name",
        help="Name for the generated Python file, without the .py extension.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "💾 Download .py file",
            data=source,
            file_name=f"{st.session_state.export_file_name}.py",
            mime="text/x-python",
        )

    with col2:
        if st.button("💾 Save to tasks/parsers/"):
            filename = st.session_state.export_file_name.strip()
            output_dir = os.path.join(os.getcwd(), "tasks", "parsers")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{filename}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(source)
            st.success(f"Saved to `{file_path}`")
