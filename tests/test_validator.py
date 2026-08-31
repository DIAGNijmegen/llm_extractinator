"""What a failed row carries.

The rule under test is that a failure never looks like an answer. The previous
implementation filled type defaults and picked a ``random.choice`` for enum
fields, so a failed row was indistinguishable from a real extraction except for
the ``status`` column — and not even reproducible between runs.
"""

from typing import Literal, Optional

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from llm_extractinator.validator import (
    DIAGNOSTIC_FIELDS,
    MAX_CAPTURED_CHARS,
    handle_prediction_failure,
    raw_model_output,
    success_diagnostics,
)


class SimpleModel(BaseModel):
    name: str
    score: int


class EnumModel(BaseModel):
    severity: Literal["mild", "moderate", "severe"]
    note: Optional[str] = None


# ── never invent a value ──────────────────────────────────────────


def test_status_is_failure():
    result = handle_prediction_failure(ValueError("oops"), SimpleModel)
    assert result["status"] == "failure"


def test_every_schema_field_is_none():
    """Not "" and not 0.

    A type default is a value: ``0`` for a failed heart rate is a real zero in a
    mean, where ``None`` becomes NaN and drops out of the statistic.
    """
    result = handle_prediction_failure(RuntimeError("bad"), SimpleModel)
    assert result["name"] is None
    assert result["score"] is None


def test_enum_fields_are_not_filled_with_a_random_valid_choice():
    """The regression this module exists for.

    ``random.choice`` over a Literal produced a plausible, non-reproducible
    answer on a row where the model said nothing usable. Repeated, because the
    old behaviour would only be caught intermittently by a single call.
    """
    for _ in range(20):
        result = handle_prediction_failure(ValueError("x"), EnumModel)
        assert result["severity"] is None, (
            "a failed row was given a valid-looking severity the model "
            "never produced"
        )


def test_no_keys_beyond_the_schema_and_the_diagnostics():
    """The caller merges the source row on top of this one, so anything extra
    here duplicates a column — or worse, looks like an extraction result."""
    result = handle_prediction_failure(ValueError("err"), SimpleModel)
    assert set(result) == {"name", "score", "status", *DIAGNOSTIC_FIELDS}


# ── keep the evidence ─────────────────────────────────────────────


def test_error_type_and_message_are_recorded():
    error = OutputParserException("Invalid json output: blah")
    result = handle_prediction_failure(error, SimpleModel)
    assert result["error_type"] == "OutputParserException"
    assert "Invalid json output" in result["error_message"]


def test_raw_model_output_is_kept():
    """The model's own words are the whole diagnosis.

    Truncated JSON means the generation budget ran out; prose means the model
    ignored the schema. Those need opposite fixes and are indistinguishable
    without the text.
    """
    error = OutputParserException("Invalid json output", llm_output='{"name": "Ali')
    result = handle_prediction_failure(error, SimpleModel)
    assert result["raw_output"] == '{"name": "Ali'


def test_raw_output_is_found_through_the_cause():
    inner = OutputParserException("inner", llm_output="the model said this")
    outer = ValueError("wrapped")
    outer.__cause__ = inner
    assert raw_model_output(outer) == "the model said this"


def test_raw_output_is_none_when_the_model_never_answered():
    """A transport failure has no output, and that absence is informative."""
    assert raw_model_output(ConnectionError("Failed to connect to Ollama")) is None


def test_long_output_is_clipped_and_says_so():
    huge = "x" * (MAX_CAPTURED_CHARS + 500)
    captured = raw_model_output(OutputParserException("e", llm_output=huge))
    assert len(captured) < len(huge)
    assert "truncated" in captured
    assert str(len(huge)) in captured


def test_successful_rows_carry_the_same_diagnostic_keys():
    """One uniform set of keys per file, whether or not anything failed."""
    assert set(success_diagnostics()) == set(DIAGNOSTIC_FIELDS)
    assert all(value is None for value in success_diagnostics().values())
