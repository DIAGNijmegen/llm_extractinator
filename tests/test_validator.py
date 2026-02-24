from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel

from llm_extractinator.validator import (
    handle_failure,
    handle_prediction_failure,
    validate_results,
)


class SimpleModel(BaseModel):
    name: str
    score: int


class NestedModel(BaseModel):
    label: str
    count: int


# ── handle_failure ────────────────────────────────────────────────


def test_handle_failure_str():
    assert handle_failure(str) == ""


def test_handle_failure_int():
    assert handle_failure(int) == 0


def test_handle_failure_float():
    assert handle_failure(float) == 0.0


def test_handle_failure_bool():
    assert handle_failure(bool) is False


def test_handle_failure_list():
    assert handle_failure(list) == []


def test_handle_failure_dict():
    assert handle_failure(dict) == {}


def test_handle_failure_literal_returns_valid_choice():
    result = handle_failure(Literal["a", "b", "c"])
    assert result in {"a", "b", "c"}


def test_handle_failure_optional_str():
    assert handle_failure(Optional[str]) == ""


def test_handle_failure_union_int_none():
    assert handle_failure(Union[int, None]) == 0


def test_handle_failure_nested_basemodel():
    result = handle_failure(NestedModel)
    assert isinstance(result, NestedModel)
    assert result.label == ""
    assert result.count == 0


def test_handle_failure_unknown_type_returns_none():
    class Arbitrary:
        pass

    assert handle_failure(Arbitrary) is None


# ── handle_prediction_failure ─────────────────────────────────────


def test_handle_prediction_failure_status_is_failure():
    result = handle_prediction_failure(ValueError("oops"), {}, SimpleModel)
    assert result["status"] == "failure"


def test_handle_prediction_failure_fields_are_defaults():
    result = handle_prediction_failure(RuntimeError("bad"), {}, SimpleModel)
    assert result["name"] == ""
    assert result["score"] == 0


def test_handle_prediction_failure_does_not_use_input_values():
    result = handle_prediction_failure(
        ValueError("err"), {"name": "Alice", "score": 99}, SimpleModel
    )
    assert result["name"] == ""
    assert result["score"] == 0


# ── validate_results ──────────────────────────────────────────────


def test_validate_results_valid_dict():
    result = validate_results({"name": "Alice", "score": 10}, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == "Alice"
    assert result.score == 10


def test_validate_results_valid_pydantic_model():
    model = SimpleModel(name="Bob", score=5)
    result = validate_results(model, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == "Bob"


def test_validate_results_invalid_dict_falls_back_to_defaults():
    # "not_an_int" cannot be coerced to int → ValidationError → fallback
    result = validate_results({"name": "Alice", "score": "not_an_int"}, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == ""
    assert result.score == 0


def test_validate_results_invalid_model_falls_back_to_defaults():
    # model_construct bypasses validation, creating an intentionally broken object
    bad = SimpleModel.model_construct(name=999, score="oops")
    result = validate_results(bad, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == ""
    assert result.score == 0
