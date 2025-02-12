import json
import random
import re
import time
from pathlib import Path
from typing import Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel


def save_json(
    data,
    outpath: Path,
    filename: Optional[str] = None,
    retries: int = 3,
    delay: float = 1.0,
):
    path = outpath / filename if filename else outpath
    if isinstance(data, BaseModel):
        data = data.model_dump()

    attempt = 0
    while attempt < retries:
        try:
            with path.open("w+") as f:
                json.dump(data, f, indent=4)
            print(f"Data successfully saved to {path}")
            break
        except IOError as e:
            attempt += 1
            print(f"Error saving data to {path}: {e}. Retrying {attempt}/{retries}...")
            time.sleep(delay)
    else:
        print(f"Failed to save data after {retries} attempts.")


def extract_json_from_text(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return str(match.group(0))
        except json.JSONDecodeError:
            pass
    return "{}"


def handle_failure(annotation):
    """Handle various types and return default values for failed cases."""
    if get_origin(annotation) is Literal:
        return random.choice(get_args(annotation))
    type_defaults = {str: "", int: 0, float: 0.0, bool: False, list: [], dict: {}}

    if annotation in type_defaults:
        return type_defaults[annotation]
    if get_origin(annotation) is Optional or get_origin(annotation) is Union:
        return handle_failure(get_args(annotation)[0])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation(
            **{
                field: handle_failure(field_type)
                for field, field_type in annotation.__annotations__.items()
            }
        )
    return None
