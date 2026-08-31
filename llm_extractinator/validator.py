"""What a failed prediction leaves behind.

A row that could not be parsed still has to appear in the output — downstream
code joins on row order — but *what it carries* decides whether the failure is
diagnosable or merely invisible. Two rules follow from that:

**Never invent a value.** Every schema field on a failed row is ``None``. The
previous behaviour filled type defaults (``""``, ``0``, ``[]``) and, for a
``Literal`` field, ``random.choice`` over the permitted values — so a failed row
carried a plausible, non-reproducible answer that no consumer could distinguish
from a real one. ``0`` is the dangerous case: a failed heart-rate row is a real
zero in a mean, where ``None`` becomes ``NaN`` and drops out.

**Keep the evidence.** The exception reaching this module is the model's own
words. ``OutputParserException`` carries ``llm_output`` — exactly what came
back. Discarding it is what made a failed run look like an empty one; the most
likely cause of unparseable output is the generation budget running out
mid-object, and that is obvious from the raw text and invisible without it.
"""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Long enough to show a truncated JSON object and what it was in the middle of;
# short enough that a run with thousands of failures stays readable.
MAX_CAPTURED_CHARS = 4000

# Columns a failed row adds. Successful rows carry them as None so every row in
# an output file has the same keys.
DIAGNOSTIC_FIELDS = ("error_type", "error_message", "raw_output")


def _clip(text: Optional[Any]) -> Optional[str]:
    """Bound a captured string, saying so rather than silently shortening it."""
    if text is None:
        return None
    text = str(text)
    if len(text) <= MAX_CAPTURED_CHARS:
        return text
    return f"{text[:MAX_CAPTURED_CHARS]}… [truncated, {len(text)} chars total]"


def raw_model_output(error: BaseException) -> Optional[str]:
    """The model's own output, if the exception carried it.

    ``OutputParserException`` sets ``llm_output`` to the completion it failed to
    parse. ``__cause__`` is checked too because a parser may re-raise wrapping
    the original. Returns ``None`` when the failure happened before the model
    said anything — a transport error, for instance — which is itself
    informative: no output means the request never completed.
    """
    output = getattr(error, "llm_output", None)
    if output is None and error.__cause__ is not None:
        output = getattr(error.__cause__, "llm_output", None)
    return _clip(output)


def handle_prediction_failure(
    error: BaseException, parser_model: BaseModel
) -> Dict[str, Any]:
    """Build the output row for a prediction that failed.

    Schema fields are ``None``; the diagnostic fields describe the failure. The
    input is deliberately not echoed here — the caller merges the source row
    onto this one, so repeating it would only duplicate a column.
    """
    row: Dict[str, Any] = {name: None for name in parser_model.model_fields}
    row["status"] = "failure"
    row["error_type"] = type(error).__name__
    row["error_message"] = _clip(error)
    row["raw_output"] = raw_model_output(error)

    logger.debug("Prediction failed (%s): %s", row["error_type"], row["error_message"])
    return row


def success_diagnostics() -> Dict[str, Any]:
    """The diagnostic columns a successful row carries — all empty.

    Present so that a predictions file has one uniform set of keys whether or
    not anything failed, rather than columns that appear only on bad runs.
    """
    return {field: None for field in DIAGNOSTIC_FIELDS}
