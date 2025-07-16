"""Pydantic v2 model-builder UI.

Run directly for a standalone Streamlit page **or** import and embed inside
another Streamlit app tab.

Usage
-----
Standalone:
    streamlit run pydantic_builder.py

Embedded:
    import streamlit as st
    from pydantic_builder import render_schema_builder

    st.set_page_config(layout="wide")  # your main app config
    tab_a, tab_b, tab_schema = st.tabs(["Data", "Run", "Schema Builder"])
    with tab_schema:
        render_schema_builder(embed=True)

The embedded mode suppresses page-level chrome (page_config, big title text) so
it plays nicely inside a larger multi-tab app.
"""

from __future__ import annotations

import ast
import os
from typing import Any

import streamlit as st

################################################################################
# Constants
################################################################################
PRIMITIVE_TYPES = ["str", "int", "float", "bool"]
SPECIAL_TYPES = ["list", "dict", "Any", "Literal"]
_DEFAULT_MODEL_NAME = "OutputParser"
_DEFAULT_EXPORT_NAME = "output_parser"
_STATE_KEY = "schema_builder"  # namespace all session_state under one key


################################################################################
# Session-state helpers
################################################################################


def _init_session() -> None:
    """Ensure our namespaced builder state exists in st.session_state."""
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = {
            "models": {_DEFAULT_MODEL_NAME: []},
            "export_file_name": _DEFAULT_EXPORT_NAME,
        }


def _state() -> dict[str, Any]:
    return st.session_state[_STATE_KEY]


def _models() -> dict[str, list[dict[str, Any]]]:
    return _state()["models"]


################################################################################
# Type composition & import detection
################################################################################


def _compose_type(
    field_type: str, *, subtype: str | None = None, lit_vals: str | None = None
) -> str:
    """Return the canonical type annotation string for the selected inputs."""
    if field_type == "Literal" and lit_vals:
        return f"Literal[{', '.join(v.strip() for v in lit_vals.split(','))}]"
    if field_type == "list" and subtype:
        return f"list[{subtype}]"
    if field_type == "dict" and subtype:
        key_t, val_t = (subtype.split(":", 1) + ["str"])[0:2]
        return f"dict[{key_t.strip()}, {val_t.strip()}]"
    return field_type


def _detect_imports() -> list[str]:
    """Infer the import lines required by the current set of models."""
    imports = {"from pydantic import BaseModel"}
    typing: set[str] = set()
    use_field = False

    for fields in _models().values():
        for f in fields:
            t = f["type"]
            if f.get("field_expr"):
                use_field = True
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
    if use_field:
        imports.add("from pydantic import Field")
    return sorted(imports)


################################################################################
# Code generation & parsing
################################################################################


def _generate_code() -> str:
    code_lines = _detect_imports() + ["\n"]
    # reverse so nested references work (child models defined first in UI? you pick)
    for model_name, fields in reversed(_models().items()):
        code_lines.append(f"class {model_name}(BaseModel):")
        if not fields:
            code_lines.append("    pass")
        else:
            for f in fields:
                line = f"    {f['name']}: {f['type']}"
                if f.get("field_expr"):
                    line += f" = {f['field_expr']}"
                elif f["type"].startswith("Optional["):
                    line += " = None"
                code_lines.append(line)
        code_lines.append("")
    return "\n".join(code_lines)


def _parse_models_from_source(source: str) -> dict[str, list[dict[str, Any]]]:
    """Extract BaseModel subclasses & field info from Python source text."""
    tree = ast.parse(source)
    models: dict[str, list[dict[str, Any]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases
        ):
            continue
        fields: list[dict[str, Any]] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field_name = stmt.target.id
                field_type = ast.get_source_segment(source, stmt.annotation)
                field_expr = (
                    ast.get_source_segment(source, stmt.value) if stmt.value else None
                )
                fields.append(
                    {"name": field_name, "type": field_type, "field_expr": field_expr}
                )
        models[node.name] = fields
    return models


################################################################################
# UI Subcomponents
################################################################################


def _manager_ui(container) -> None:
    """Model manager sidebar tab: create new models."""
    with container:
        st.header("📦 Model Manager")
        new_model = st.text_input(
            "Enter new model name (e.g. User)", key=f"sb_new_model_name"
        )
        if st.button("➕ Add model", use_container_width=True, key="sb_add_model_btn"):
            name = new_model.strip()
            if not name:
                st.warning("Please enter a model name.")
            elif not name.isidentifier() or not name[0].isupper():
                st.warning(
                    "Model names must be valid identifiers and start with a capital letter."
                )
            elif name in _models():
                st.warning(f"Model **{name}** already exists.")
            else:
                _models()[name] = []
                st.success(f"Model **{name}** created.")


def _import_ui(container) -> None:
    """Sidebar tab: import .py and load models."""
    with container:
        st.header("📂 Import existing models")
        uploaded_file = st.file_uploader(
            "Upload a Python file", type=["py"], key="sb_file_uploader"
        )
        if uploaded_file and st.button(
            "🔄 Load into editor", type="primary", key="sb_load_btn"
        ):
            try:
                source_code = uploaded_file.read().decode("utf-8")
                models = _parse_models_from_source(source_code)
                if models:
                    _state()["models"] = models
                    st.success("Models imported successfully.")
                    st.rerun()
                else:
                    st.warning("No BaseModel classes found.")
            except Exception as e:  # pragma: no cover - defensive
                st.error(f"Error: {e}")


def _design_ui(container) -> None:
    """Main design tab: add/delete fields per model."""
    with container:
        models_snapshot = list(_models())  # avoid changing dict while iterating
        for model_name in models_snapshot:
            with st.expander(f"🧰 {model_name}", expanded=False):
                st.markdown(f"### Define fields for `{model_name}`")

                # --- field inputs row --------------------------------------------------
                cols = st.columns([2, 2, 1])
                with cols[0]:
                    field_name = st.text_input(
                        "Field name", key=f"sb_name_{model_name}"
                    )
                with cols[1]:
                    field_type = st.selectbox(
                        "Field type",
                        PRIMITIVE_TYPES
                        + SPECIAL_TYPES
                        + [m for m in _models() if m != model_name],
                        key=f"sb_type_{model_name}",
                    )
                with cols[2]:
                    is_optional = st.checkbox("Optional", key=f"sb_opt_{model_name}")

                sub_type = literal_vals = None
                if field_type == "list":
                    sub_type = st.selectbox(
                        "List element type",
                        PRIMITIVE_TYPES + [m for m in _models()],
                        key=f"sb_subtype_list_{model_name}",
                    )
                elif field_type == "dict":
                    c1, c2 = st.columns(2)
                    key_type = c1.selectbox(
                        "Key type", PRIMITIVE_TYPES, key=f"sb_key_dict_{model_name}"
                    )
                    val_type = c2.selectbox(
                        "Value type",
                        PRIMITIVE_TYPES + [m for m in _models()],
                        key=f"sb_val_dict_{model_name}",
                    )
                    sub_type = f"{key_type}:{val_type}"
                elif field_type == "Literal":
                    literal_vals = st.text_input(
                        "Literal values", key=f"sb_lit_{model_name}"
                    )

                # --- advanced options toggle ------------------------------------------
                show_advanced = st.checkbox(
                    "Show advanced field options", key=f"sb_adv_{model_name}"
                )
                if show_advanced:
                    field_default = st.text_input(
                        "Default value (raw Python)", key=f"sb_default_{model_name}"
                    )
                    field_desc = st.text_input(
                        "Description", key=f"sb_desc_{model_name}"
                    )
                    field_extra = st.text_input(
                        "Extra Field args", key=f"sb_extra_{model_name}"
                    )
                else:
                    field_default = field_desc = field_extra = ""

                if st.button("Add field", key=f"sb_add_field_btn_{model_name}"):
                    name = field_name.strip()
                    if not name:
                        st.warning("Enter a field name.")
                    elif any(f["name"] == name for f in _models()[model_name]):
                        st.warning(f"Field {name} already exists.")
                    elif field_type in {"list", "dict"} and not sub_type:
                        st.warning("Please specify subtype for list or dict.")
                    elif field_type == "Literal" and not literal_vals:
                        st.warning("Please enter values for Literal.")
                    else:
                        final_type = _compose_type(
                            field_type, subtype=sub_type, lit_vals=literal_vals
                        )
                        if is_optional:
                            final_type = f"Optional[{final_type}]"

                        # Compose Field(...) expression
                        field_args = []
                        if field_desc:
                            field_args.append(f'description="{field_desc}"')
                        if field_extra:
                            field_args.append(field_extra)

                        field_expr = None
                        if field_default:
                            field_expr = (
                                f"Field({field_default}, {', '.join(field_args)})"
                                if field_args
                                else f"Field({field_default})"
                            )
                        elif field_args:
                            field_expr = f"Field({', '.join(field_args)})"

                        _models()[model_name].append(
                            {
                                "name": name,
                                "type": final_type,
                                "field_expr": field_expr,
                            }
                        )
                        st.success(f"Added field {name} to {model_name}.")

                # --- existing fields table -------------------------------------------
                if _models()[model_name]:
                    st.markdown("#### Fields")
                    for i, field in enumerate(_models()[model_name]):
                        cols = st.columns([3, 3, 3, 1])
                        cols[0].markdown(f"`{field['name']}`")
                        cols[1].markdown(f"`{field['type']}`")
                        optional = field["type"].startswith("Optional[")
                        cols[2].markdown("🔓 Optional" if optional else "🔒 Required")
                        if cols[3].button("🗑️", key=f"sb_del_{model_name}_{i}"):
                            del _models()[model_name][i]
                            st.rerun()
                else:
                    st.info("No fields yet.")


def _code_ui(container) -> None:
    with container:
        st.subheader("📝 Generated Python Code")
        code = _generate_code()
        st.code(code, language="python")


def _export_ui(container) -> None:
    with container:
        st.subheader("📅 Export")
        export_name_key = "sb_export_file_name"
        export_name = st.text_input(
            "Filename",
            value=_state().get("export_file_name", _DEFAULT_EXPORT_NAME),
            key=export_name_key,
        )
        # keep in state
        _state()["export_file_name"] = export_name

        code = _generate_code()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "💾 Download .py",
                data=code,
                file_name=f"{export_name}.py",
                mime="text/x-python",
                key="sb_download_btn",
            )
        with col2:
            if st.button("💾 Save to tasks/parsers/", key="sb_save_btn"):
                path = os.path.join("tasks", "parsers")
                os.makedirs(path, exist_ok=True)
                with open(
                    os.path.join(path, f"{export_name}.py"), "w", encoding="utf-8"
                ) as f:
                    f.write(code)
                st.success("Saved successfully.")


################################################################################
# Public renderer
################################################################################


def render_schema_builder(*, embed: bool = False) -> None:
    """Render the Pydantic-model builder UI.

    Parameters
    ----------
    embed : bool, default False
        False → full standalone page (page_config, title, intro copy).
        True  → slim/embedded mode for use inside another app/tab.
    """
    _init_session()

    # ------------------------------------------------------------------ Page shell
    if not embed:
        # set_page_config can only be called once; ignore duplicate attempts
        try:  # pragma: no cover - streamlit runtime side-effect
            st.set_page_config(
                page_title="Pydantic v2 Model Builder", layout="wide", page_icon="🛠️"
            )
        except Exception:  # catch RuntimeError on re-run or prior set
            pass

        st.title("🛠️ Pydantic Model Builder")
        st.markdown(
            """
            Build and preview **Pydantic v2** models without writing any code.

            **What can you do here?**
            - Create Python data models using a visual interface.
            - Add fields with built‑in types, collections, or nested models.
            - **Import** existing model files to continue editing them.
            - Export the resulting code to use in your projects.
            """
        )

    # ------------------------------------------------------------------ Sidebar (still shown when embedded; host app controls layout)
    with st.sidebar:
        manager_tab, import_tab = st.tabs(["📦 Model Manager", "📂 Import file"])
        _manager_ui(manager_tab)
        _import_ui(import_tab)

    # ------------------------------------------------------------------ Main area
    design_tab, code_tab, export_tab = st.tabs(["🔗 Design", "📝 Code", "📅 Export"])
    _design_ui(design_tab)
    _code_ui(code_tab)
    _export_ui(export_tab)


################################################################################
# Standalone entry point
################################################################################
if __name__ == "__main__":
    render_schema_builder(embed=False)
