import json

import pytest
from pydantic import BaseModel

from llm_extractinator.utils import save_json

# ── save_json ─────────────────────────────────────────────────────


def test_save_json_writes_dict(tmp_path):
    data = {"hello": "world"}
    outfile = tmp_path / "output.json"
    save_json(data, outfile)
    assert json.loads(outfile.read_text()) == {"hello": "world"}


def test_save_json_with_filename(tmp_path):
    save_json([1, 2, 3], tmp_path, filename="results.json")
    assert json.loads((tmp_path / "results.json").read_text()) == [1, 2, 3]


def test_save_json_pydantic_model(tmp_path):
    class MyModel(BaseModel):
        name: str
        value: int

    outfile = tmp_path / "model.json"
    save_json(MyModel(name="test", value=42), outfile)
    assert json.loads(outfile.read_text()) == {"name": "test", "value": 42}


def test_save_json_indented(tmp_path):
    outfile = tmp_path / "pretty.json"
    save_json({"a": 1}, outfile)
    raw = outfile.read_text()
    # indent=4 means the file contains actual newlines
    assert "\n" in raw
