import json
import time
from pathlib import Path
from typing import Optional

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
