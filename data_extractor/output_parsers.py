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
    elif field_info["type"] == "list":
        # Handle list types
        item_type_info = field_info.get("items")
        if not item_type_info:
            raise ValueError("'items' must be defined for list type fields.")

        if "type" not in item_type_info:
            raise KeyError("'type' key is missing in 'items' for list field.")

        # If items are dictionaries, create a nested model for them
        if item_type_info["type"] == "dict":
            nested_model = create_pydantic_model_from_json(
                item_type_info["properties"], model_name="DictionaryModel"
            )
            item_type = nested_model
        else:
            # Otherwise, handle items as basic types or literals
            item_type, _ = create_field(item_type_info)

        field_type = List[item_type]
    elif field_info["type"] == "bool":
        # Handle boolean type with strict validation
        field_type = bool
    elif field_info["type"] in type_mapping:
        # Handle basic types using type_mapping
        field_type = type_mapping[field_info["type"]]
    else:
        # Raise an error for unknown types
        raise ValueError(f"Unsupported field type: {field_info['type']}")

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
    validators = {}

    for key, field_info in data.items():
        field_type, pydantic_field = create_field(field_info)
        fields[key] = (field_type, pydantic_field)

        # Add custom validators for boolean fields
        if field_info["type"] == "bool":

            @field_validator(key)
            def strict_bool_validator(cls, value):
                if not isinstance(value, bool):
                    raise TypeError(f"Field '{key}' must be a boolean.")
                return value

            validators[f"validate_{key}"] = strict_bool_validator

    # Create the model dynamically
    model = create_model(model_name, **fields)

    # Add the validators dynamically
    for validator_name, validator_func in validators.items():
        setattr(model, validator_name, validator_func)

    return model


def load_parser(
    task_type: str, parser_format: Optional[Dict[str, Any]]
) -> Type[BaseModel]:
    if task_type == "Example Generation":

        class ExampleGenerationOutput(BaseModel):
            reasoning: str = Field(
                description="The thought process leading to the answer"
            )

        return ExampleGenerationOutput
    elif task_type == "Translation":

        class TranslationOutput(BaseModel):
            translation: str = Field(description="The text translated to English")

        return TranslationOutput
    else:
        return create_pydantic_model_from_json(data=parser_format)
