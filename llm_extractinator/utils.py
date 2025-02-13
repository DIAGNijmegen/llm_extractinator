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


def extract_json_from_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_text = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {}


def handle_failure(annotation):
    """Handle various types and return default values for failed cases."""
    if get_origin(annotation) is Literal:
        return random.choice(get_args(annotation))

    type_defaults = {str: "", int: 0, float: 0.0, bool: False, list: [], dict: {}}

    if annotation in type_defaults:
        return type_defaults[annotation]

    if get_origin(annotation) in {Optional, Union}:
        return handle_failure(get_args(annotation)[0])

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        nested_instance = {
            field_name: handle_failure(field)
            for field_name, field in annotation.__annotations__.items()
        }
        return annotation.model_construct(**nested_instance)

    return None
