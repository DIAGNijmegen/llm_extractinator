import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, Field, create_model
from pydantic.types import StrictBool

type_mapping = {
    "str": str,
    "int": int,
    "float": float,
    "bool": StrictBool,
    "list": list,
    "dict": dict,
    "any": Any,
}


def create_field(field_info: Dict[str, Any], parent_name: str) -> Tuple[Any, Field]:
    """
    Create a Pydantic field with a type, description, and optional Literal values.
    """
    field_type = type_mapping.get(field_info["type"], None)
    description = field_info.get("description", None)
    is_optional = field_info.get("optional", False)

    # Handle nested dict objects
    if field_info["type"] == "dict":
        model_name = f"{parent_name}_{field_info.get('name', 'Dict')}"
        nested_model = create_pydantic_model_from_json(
            field_info["properties"], model_name=model_name
        )
        field_type = nested_model

    # Handle list types
    elif field_info["type"] == "list":
        item_type_info = field_info.get("items")
        if not item_type_info:
            raise ValueError("'items' must be defined for list type fields.")

        item_type, _ = create_field(item_type_info, parent_name + "Item")
        field_type = List[item_type]

    elif field_type is None:
        raise ValueError(f"Unsupported field type: {field_info['type']}")

    # Handle literals if specified
    literals = field_info.get("literals")
    if literals:
        field_type = Literal[tuple(literals)]

    if is_optional:
        field_type = Optional[field_type]

    return field_type, Field(
        default=None if is_optional else ..., description=description
    )


def create_pydantic_model_from_json(
    data: Dict[str, Any], model_name: str = "OutputParser"
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model from a dictionary specification.
    """
    fields = {}

    for key, field_info in data.items():
        field_type, pydantic_field = create_field(field_info, parent_name=model_name)
        fields[key] = (field_type, pydantic_field)

    return create_model(model_name, **fields)


def load_parser(
    task_type: str, parser_format: Optional[Dict[str, Any]]
) -> Type[BaseModel]:
    """
    Load a predefined Pydantic model based on task type, or generate one dynamically.
    """
    predefined_models = {
        "Example Generation": lambda: create_model(
            "ExampleGenerationOutput",
            reasoning=(
                str,
                Field(description="The thought process leading to the answer"),
            ),
        ),
        "Translation": lambda: create_model(
            "TranslationOutput",
            translation=(str, Field(description="The text translated to English")),
        ),
    }

    if task_type in predefined_models:
        return predefined_models[task_type]()

    if parser_format is None:
        raise ValueError("parser_format must be provided for custom task types.")

    return create_pydantic_model_from_json(parser_format)


def load_parser_pydantic(parser_path: Path) -> BaseModel:
    """
    Load a Pydantic model from a python file
    """
    if parser_path.exists():
        module_name = parser_path.stem  # Get the filename without .py
        spec = importlib.util.spec_from_file_location(module_name, str(parser_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "OutputParser"):
            return getattr(module, "OutputParser")
        else:
            raise ImportError(f"No OutputParser class found in {parser_path}")

    else:
        raise FileNotFoundError(f"Parser file not found in {parser_path}.")
