import random
from typing import Any, Dict, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError


def handle_failure(annotation):
    """
    Return default/failure values depending on annotation type.
    Called only after all fix attempts have failed.
    """
    # If it's a Literal, pick a random valid literal
    if get_origin(annotation) is Literal:  # Corrected this check
        literals = get_args(annotation)
        if literals:
            return random.choice(literals)

    # Basic defaults for common types
    type_defaults = {str: "", int: 0, float: 0.0, bool: False, list: [], dict: {}}
    if annotation in type_defaults:
        return type_defaults[annotation]

    # If it's Optional[X] or Union[X, None], handle X
    if get_origin(annotation) in {Optional, Union}:
        # Get first item that isn't None
        subtypes = [t for t in get_args(annotation) if t is not type(None)]
        if subtypes:
            return handle_failure(subtypes[0])
        else:
            return None

    # If annotation is a nested Pydantic BaseModel, recursively build defaults
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        nested_data = {}
        for field_name, field_def in annotation.model_fields.items():
            nested_data[field_name] = handle_failure(field_def.annotation)
        return annotation.model_construct(**nested_data)

    # Fallback
    return None


def handle_prediction_failure(
    error: Exception, input_data: Dict[str, Any], parser_model: BaseModel
) -> Dict[str, Any]:
    """
    Handle failures during prediction by logging the error and returning a default response.
    """

    default_values = {}
    for field_name, field_def in parser_model.model_fields.items():
        default_values[field_name] = handle_failure(field_def.annotation)

    return {**default_values, "status": "failure"}


def validate_results(
    item: Any,  # could be a dict or a pydantic model
    parser_model: BaseModel,
):
    """
    Final fallback validator, in case OutputFixingParser's attempts did not yield a valid object.
    Returns either a valid Pydantic model or a default-filled model.
    """
    # If we already have a pydantic model, check it:
    if isinstance(item, BaseModel):
        # quick try: re-validate by dumping & re-parsing
        try:
            return parser_model.model_validate(item.model_dump())
        except ValidationError:
            pass
    else:
        # If it's a dict
        try:
            return parser_model.model_validate(item)
        except ValidationError:
            pass

    # If we get here, everything has failed. We provide a final fallback.
    print("[validate_results] Provided data is invalid. Assigning defaults.")
    fallback_data = {}
    for field_name, field_def in parser_model.model_fields.items():
        fallback_data[field_name] = handle_failure(field_def.annotation)

    return parser_model.model_construct(**fallback_data)
