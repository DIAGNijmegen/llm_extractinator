from __future__ import annotations

import ast
import os
import textwrap
from typing import Any, Literal, Optional

import streamlit as st

################################################################################
# Page config
################################################################################

if "_PAGE_CONFIG_DONE" not in globals():
    try:
        st.set_page_config(
            page_title="Pydantic v2 Model Builder", layout="wide", page_icon="🛠️"
        )
    except Exception:
        pass
    _PAGE_CONFIG_DONE = True

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

if "models" not in st.session_state:
    st.session_state.models = {"OutputParser": []}

PRIMITIVE_TYPES = ["str", "int", "float", "bool"]
SPECIAL_TYPES = ["list", "dict", "Any", "Literal"]


def _compose_type(
    field_type: str, *, subtype: str | None = None, lit_vals: str | None = None
) -> str:
    if field_type == "Literal" and lit_vals:
        return f"Literal[{', '.join(v.strip() for v in lit_vals.split(','))}]"
    if field_type == "list" and subtype:
        return f"list[{subtype}]"
    if field_type == "dict" and subtype:
        key_t, val_t = (subtype.split(":", 1) + ["str"])[0:2]
        return f"dict[{key_t.strip()}, {val_t.strip()}]"
    return field_type


def _detect_imports() -> list[str]:
    imports = {"from pydantic import BaseModel"}
    typing: set[str] = set()
    use_field = False
    for fields in st.session_state.models.values():
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


def generate_code() -> str:
    code = _detect_imports() + ["\n"]
    for model_name, fields in reversed(st.session_state.models.items()):
        code.append(f"class {model_name}(BaseModel):")
        if not fields:
            code.append("    pass")
        else:
            for f in fields:
                line = f"    {f['name']}: {f['type']}"
                if f.get("field_expr"):
                    line += f" = {f['field_expr']}"
                elif f["type"].startswith("Optional["):
                    line += " = None"
                code.append(line)
        code.append("")
    return "\n".join(code)


def _parse_models_from_source(source: str) -> dict[str, list[dict[str, Any]]]:
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


with st.sidebar:
    manager_tab, import_tab = st.tabs(["📦 Model Manager", "📂 Import file"])

    with manager_tab:
        st.header("📦 Model Manager")
        new_model = st.text_input(
            "Enter new model name (e.g. User)", key="_new_model_name"
        )
        if st.button("➕ Add model", use_container_width=True):
            name = new_model.strip()
            if not name:
                st.warning("Please enter a model name.")
            elif not name.isidentifier() or not name[0].isupper():
                st.warning(
                    "Model names must be valid identifiers and start with a capital letter."
                )
            elif name in st.session_state.models:
                st.warning(f"Model **{name}** already exists.")
            else:
                st.session_state.models[name] = []
                st.success(f"Model **{name}** created.")

    with import_tab:
        st.header("📂 Import existing models")
        uploaded_file = st.file_uploader("Upload a Python file", type=["py"])
        if uploaded_file and st.button("🔄 Load into editor", type="primary"):
            try:
                source_code = uploaded_file.read().decode("utf-8")
                models = _parse_models_from_source(source_code)
                if models:
                    st.session_state.models = models
                    st.success("Models imported successfully.")
                    st.rerun()
                else:
                    st.warning("No BaseModel classes found.")
            except Exception as e:
                st.error(f"Error: {e}")

design_tab, code_tab, export_tab = st.tabs(["🔗 Design", "📝 Code", "📅 Export"])

with design_tab:
    for model_name in list(st.session_state.models):
        with st.expander(f"🧰 {model_name}", expanded=False):
            st.markdown(f"### Define fields for `{model_name}`")

            cols = st.columns([2, 2, 1])
            with cols[0]:
                field_name = st.text_input("Field name", key=f"name_{model_name}")
            with cols[1]:
                field_type = st.selectbox(
                    "Field type",
                    PRIMITIVE_TYPES
                    + SPECIAL_TYPES
                    + [m for m in st.session_state.models if m != model_name],
                    key=f"type_{model_name}",
                )
            with cols[2]:
                is_optional = st.checkbox("Optional", key=f"opt_{model_name}")

            sub_type = literal_vals = None
            if field_type == "list":
                sub_type = st.selectbox(
                    "List element type",
                    PRIMITIVE_TYPES + [m for m in st.session_state.models],
                    key=f"subtype_list_{model_name}",
                )
            elif field_type == "dict":
                c1, c2 = st.columns(2)
                key_type = c1.selectbox(
                    "Key type", PRIMITIVE_TYPES, key=f"key_dict_{model_name}"
                )
                val_type = c2.selectbox(
                    "Value type",
                    PRIMITIVE_TYPES + [m for m in st.session_state.models],
                    key=f"val_dict_{model_name}",
                )
                sub_type = f"{key_type}:{val_type}"
            elif field_type == "Literal":
                literal_vals = st.text_input("Literal values", key=f"lit_{model_name}")

            # ✅ Toggle instead of nested expander
            show_advanced = st.checkbox(
                "Show advanced field options", key=f"adv_{model_name}"
            )
            if show_advanced:
                field_default = st.text_input(
                    "Default value (raw Python)", key=f"default_{model_name}"
                )
                field_desc = st.text_input("Description", key=f"desc_{model_name}")
                field_extra = st.text_input(
                    "Extra Field args", key=f"extra_{model_name}"
                )
            else:
                field_default = field_desc = field_extra = ""

            if st.button("Add field", key=f"add_field_btn_{model_name}"):
                name = field_name.strip()
                if not name:
                    st.warning("Enter a field name.")
                elif any(
                    f["name"] == name for f in st.session_state.models[model_name]
                ):
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

                    st.session_state.models[model_name].append(
                        {"name": name, "type": final_type, "field_expr": field_expr}
                    )
                    st.success(f"Added field {name} to {model_name}.")

            if st.session_state.models[model_name]:
                st.markdown("#### Fields")
                for i, field in enumerate(st.session_state.models[model_name]):
                    cols = st.columns([3, 3, 3, 1])
                    cols[0].markdown(f"`{field['name']}`")
                    cols[1].markdown(f"`{field['type']}`")
                    optional = field["type"].startswith("Optional[")
                    cols[2].markdown("🔓 Optional" if optional else "🔒 Required")
                    if cols[3].button("🗑️", key=f"del_{model_name}_{i}"):
                        del st.session_state.models[model_name][i]
                        st.rerun()
            else:
                st.info("No fields yet.")


with code_tab:
    st.subheader("📝 Generated Python Code")
    code = generate_code()
    st.code(code, language="python")

with export_tab:
    st.subheader("📅 Export")
    st.text_input("Filename", value="output_parser", key="export_file_name")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download .py",
            data=code,
            file_name=f"{st.session_state.export_file_name}.py",
            mime="text/x-python",
        )
    with col2:
        if st.button("💾 Save to tasks/parsers/"):
            path = os.path.join("tasks", "parsers")
            os.makedirs(path, exist_ok=True)
            with open(
                os.path.join(path, f"{st.session_state.export_file_name}.py"), "w"
            ) as f:
                f.write(code)
            st.success("Saved successfully.")
