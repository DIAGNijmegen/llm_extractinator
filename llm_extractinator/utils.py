import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
            logger.info(f"Data saved to {path}.")
            break
        except IOError as e:
            attempt += 1
            logger.error(
                f"Failed to save data to {path}. Retrying ({attempt}/{retries})..."
            )
            time.sleep(delay)
    else:
        logger.error(f"Failed to save data to {path} after {retries} attempts.")


def extract_json_from_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_text = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {}
