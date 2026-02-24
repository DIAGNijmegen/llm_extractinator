"""
Unit tests for output_parsers.py.

test_pydantic_model.py already covers the happy-path of create_pydantic_model_from_json
and load_parser for predefined types, so this file focuses on:
  - Error cases for create_field
  - Error cases for load_parser
  - All paths of load_parser_pydantic (file not found, no class, wrong base class, ok)
"""

from typing import Literal, get_args, get_origin

import pytest
from pydantic import BaseModel, create_model

from llm_extractinator.output_parsers import (
    create_field,
    load_parser,
    load_parser_pydantic,
)


# ── create_field ─────────────────────────────────────────────────


def test_create_field_list_without_items_raises():
    with pytest.raises(ValueError, match="'items' must be defined"):
        create_field({"type": "list", "name": "things"}, "Parent")


def test_create_field_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported field type"):
        create_field({"type": "bytes"}, "Parent")


def test_create_field_optional_has_none_default():
    field_type, field = create_field({"type": "str", "optional": True}, "Parent")
    assert field.default is None


def test_create_field_required_validates_in_model():
    field_type, field_obj = create_field({"type": "str"}, "Parent")
    M = create_model("M", value=(field_type, field_obj))
    with pytest.raises(Exception):
        M()  # required field missing → ValidationError
    assert M(value="hello").value == "hello"


def test_create_field_with_literals():
    field_type, _ = create_field({"type": "str", "literals": ["x", "y", "z"]}, "Parent")
    assert get_origin(field_type) is Literal
    assert set(get_args(field_type)) == {"x", "y", "z"}


def test_create_field_list_of_primitives():
    field_type, _ = create_field(
        {"type": "list", "items": {"type": "str"}}, "Parent"
    )
    M = create_model("M", items=(field_type, ...))
    instance = M(items=["a", "b"])
    assert instance.items == ["a", "b"]


def test_create_field_nested_dict():
    field_type, _ = create_field(
        {
            "type": "dict",
            "name": "Profile",
            "properties": {
                "bio": {"type": "str"},
                "age": {"type": "int"},
            },
        },
        "Parent",
    )
    assert issubclass(field_type, BaseModel)
    instance = field_type(bio="hello", age=30)
    assert instance.bio == "hello"
    assert instance.age == 30


# ── load_parser ───────────────────────────────────────────────────


def test_load_parser_example_generation():
    model = load_parser("Example Generation", None)
    instance = model(reasoning="some thought")
    assert instance.reasoning == "some thought"


def test_load_parser_translation():
    model = load_parser("Translation", None)
    instance = model(translation="bonjour")
    assert instance.translation == "bonjour"


def test_load_parser_custom_without_format_raises():
    with pytest.raises(ValueError, match="parser_format must be provided"):
        load_parser("Custom", None)


def test_load_parser_custom_with_format():
    fmt = {"label": {"type": "str", "description": "a label"}}
    model = load_parser("Custom", fmt)
    assert model(label="yes").label == "yes"


# ── load_parser_pydantic ─────────────────────────────────────────


def test_load_parser_pydantic_ok(tmp_path):
    parser_file = tmp_path / "valid_parser.py"
    parser_file.write_text(
        "from pydantic import BaseModel\n"
        "class OutputParser(BaseModel):\n"
        "    value: str\n"
    )
    model = load_parser_pydantic(parser_file)
    assert issubclass(model, BaseModel)
    assert model(value="test").value == "test"


def test_load_parser_pydantic_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_parser_pydantic(tmp_path / "ghost.py")


def test_load_parser_pydantic_no_output_parser_class(tmp_path):
    parser_file = tmp_path / "no_class_parser.py"
    parser_file.write_text("x = 42\n")
    with pytest.raises(ImportError, match="No OutputParser class"):
        load_parser_pydantic(parser_file)


def test_load_parser_pydantic_not_a_basemodel(tmp_path):
    parser_file = tmp_path / "bad_base_parser.py"
    parser_file.write_text("class OutputParser:\n    pass\n")
    with pytest.raises(TypeError, match="not a subclass of BaseModel"):
        load_parser_pydantic(parser_file)
