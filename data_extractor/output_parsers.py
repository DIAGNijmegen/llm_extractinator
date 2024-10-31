from typing import Any, Dict, List, Literal, Optional, Type, Union, get_args

from pydantic import BaseModel, Field, create_model, field_validator

type_mapping = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "any": Any,
}


def create_field(field_info: Dict[str, Any]) -> Any:
    """
    Create a Pydantic field with a type, description, and optional Literal values.
    """
    # Determine if the field is optional
    is_optional = field_info.get("optional", False)
    description = field_info.get("description", None)

    # Handle nested object (dictionary) types
    if field_info["type"] == "dict":
        nested_model = create_pydantic_model_from_json(
            field_info["properties"], model_name="DictionaryModel"
        )
        field_type = nested_model
    else:
        # Get the basic type from the type mapping
        field_type = type_mapping.get(field_info["type"], Any)

        # Handle literals if specified
        literals = field_info.get("literals")
        if literals:
            field_type = Literal[tuple(literals)]

    # If the field is optional, wrap the type with Optional
    if is_optional:
        field_type = Optional[field_type]

    # Create a Pydantic Field with a description
    return (
        field_type,
        Field(default=None if is_optional else ..., description=description),
    )


def create_pydantic_model_from_json(
    data: Dict[str, Any], model_name: str = "OutputParser"
) -> Type[BaseModel]:
    fields = {}
    for key, field_info in data.items():
        fields[key] = create_field(field_info)

    return create_model(model_name, **fields)


def load_parser(
    task_type: str, parser_format: Optional[Dict[str, Any]]
) -> Type[BaseModel]:
    if task_type == "Example Generation":

        class ExampleGenerationOutput(BaseModel):
            reasoning: str = Field(
                description="The thought process leading to the answer"
            )

        return ExampleGenerationOutput
    else:
        return create_pydantic_model_from_json(data=parser_format)
