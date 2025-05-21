from __future__ import annotations

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
    - Add fields with built-in types, collections, or nested models.
    - Export the resulting code to use in your projects.
    """
)


################################################################################
# Session state ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

if "models" not in st.session_state:
    st.session_state.models: dict[str, list[dict[str, Any]]] = {"OutputParser": []}

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


################################################################################
# Sidebar ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
################################################################################

with st.sidebar:
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
            st.dataframe(
                st.session_state.models[model_name],
                hide_index=True,
                use_container_width=True,
            )
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
    file_name = st.text_input("", value="output_parser")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "💾 Download .py file",
            data=source,
            file_name=f"{file_name}.py",
            mime="text/x-python",
        )
    with col2:
        if st.button("💾 Save to tasks/parsers/"):
            output_dir = os.path.join(os.getcwd(), "tasks", "parsers")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{file_name.strip()}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(source)
            st.success(f"Saved to `{file_path}`")
