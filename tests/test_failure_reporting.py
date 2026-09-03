"""A failed row has to be diagnosable from the output alone.

The symptom this covers: a run reports progress past 100%, finishes, and leaves
rows that are empty with no indication of why. Three separate causes fed it —
the exception was discarded, the progress bar counted LLM calls instead of rows,
and every parse failure was retried three times with a byte-identical request.
"""

import json
from pathlib import Path
from uuid import uuid4

import ollama
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.predictor import RETRY_ATTEMPTS, RETRYABLE_ERRORS
from tests.conftest import FakeChatModel, RecordingFakeChatModel, load_predictions

HR_RESPONSE = '{"HR": 78, "Name": "Alice"}'
UNPARSEABLE = "The report describes a heart rate but I cannot produce JSON."
TASK_NAME = "Task999_example"


def run_dir(out: Path, run_name: str) -> Path:
    return out / run_name / f"{TASK_NAME}-run0"


# ── the evidence reaches the output ───────────────────────────────


def test_a_failed_row_carries_the_models_own_output(offline_run):
    out = offline_run(responses=[UNPARSEABLE], task_id=999, run_name="ev")
    preds = load_predictions(out, "ev", TASK_NAME)

    for row in preds:
        assert row["status"] == "failure"
        assert row["raw_output"] == UNPARSEABLE
        assert row["error_type"] == "OutputParserException"
        assert row["HR"] is None and row["Name"] is None


def test_successful_rows_have_the_same_keys_as_failed_ones(offline_run):
    """Otherwise the diagnostic columns appear only on bad runs, and any
    consumer reading them has to special-case their absence."""
    good = load_predictions(
        offline_run(responses=[HR_RESPONSE], task_id=999, run_name="ok"), "ok", TASK_NAME
    )
    bad = load_predictions(
        offline_run(responses=[UNPARSEABLE], task_id=999, run_name="bad"), "bad", TASK_NAME
    )

    assert set(good[0]) == set(bad[0])
    assert good[0]["error_type"] is None
    assert good[0]["raw_output"] is None


def test_failures_json_collects_what_went_wrong(offline_run):
    out = offline_run(responses=[UNPARSEABLE], task_id=999, run_name="fj")
    failures = json.loads((run_dir(out, "fj") / "failures.json").read_text("utf-8"))

    assert len(failures) == 5
    first = failures[0]
    assert first["row"] == 0
    assert first["raw_output"] == UNPARSEABLE
    assert first["error_type"] == "OutputParserException"
    # Enough of the input to know which document this was.
    assert "Alice" in first["input"]


def test_no_failures_file_on_a_clean_run(offline_run):
    """A stray empty artefact on a good run is its own kind of confusing."""
    out = offline_run(responses=[HR_RESPONSE], task_id=999, run_name="clean")
    assert not (run_dir(out, "clean") / "failures.json").exists()


def test_failures_are_collected_across_chunks(offline_run):
    """Chunked runs merge before writing, so the failures file must too."""
    out = offline_run(
        responses=[UNPARSEABLE], task_id=999, run_name="fchunk", chunk_size=2
    )
    failures = json.loads((run_dir(out, "fchunk") / "failures.json").read_text("utf-8"))
    assert [f["row"] for f in failures] == [0, 1, 2, 3, 4]


# ── retrying only what a retry can fix ────────────────────────────


def test_a_deterministic_parse_failure_is_not_retried(offline_run):
    """At temperature 0 the reply cannot change, so three attempts bought three
    identical failures at three times the cost."""
    model = RecordingFakeChatModel(responses=[UNPARSEABLE])
    offline_run(model=model, task_id=999, run_name="once", temperature=0.0)

    assert len(model.seen) == 5, (
        f"5 rows should mean 5 model calls, got {len(model.seen)} — a parse "
        f"failure is being retried"
    )


def _retrying(model):
    return model.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        stop_after_attempt=RETRY_ATTEMPTS,
        wait_exponential_jitter=False,
    )


def test_a_transport_error_is_still_retried_and_can_recover():
    """The other half of the rule: narrowing the retry must not remove it.

    ``ollama`` re-raises ``httpx.ConnectError`` as a plain ``ConnectionError``
    (see ``ollama/_client.py``), which is what a server that is starting up or
    briefly unreachable looks like from here.
    """
    state = {"calls": 0}

    class Flaky(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            if state["calls"] <= 2:
                raise ConnectionError("Failed to connect to Ollama. Is it running?")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    result = _retrying(Flaky(responses=[HR_RESPONSE])).invoke("anything")

    assert state["calls"] == 3
    assert result.content == HR_RESPONSE


@pytest.mark.parametrize("status", [400, 404])
def test_a_client_side_response_error_is_not_retried(status):
    """A 400 or 404 is an answer. Asking again produces the same answer."""
    state = {"calls": 0}

    class Refusing(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            raise ollama.ResponseError("does not support thinking", status)

    with pytest.raises(ollama.ResponseError):
        _retrying(Refusing(responses=["x"])).invoke("anything")

    assert state["calls"] == 1


@pytest.mark.parametrize("status", [503, 429])
def test_a_busy_server_response_error_is_retried(status):
    """503 (unavailable) and 429 (rate limited) on a shared Ollama are the
    retryable case: the request is fine, the server was momentarily saturated,
    and the next attempt meets a different server state — not a byte-identical
    dead end like a 400."""
    state = {"calls": 0}

    class Busy(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            if state["calls"] <= 2:
                raise ollama.ResponseError("server busy", status)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    result = _retrying(Busy(responses=[HR_RESPONSE])).invoke("anything")

    assert state["calls"] == 3
    assert result.content == HR_RESPONSE


def test_a_busy_server_error_that_never_clears_stops_after_the_attempt_cap():
    """The carve-out is still bounded — 503 does not loop forever."""
    state = {"calls": 0}

    class AlwaysBusy(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            raise ollama.ResponseError("server busy", 503)

    with pytest.raises(ollama.ResponseError):
        _retrying(AlwaysBusy(responses=["x"])).invoke("anything")

    assert state["calls"] == RETRY_ATTEMPTS


# ── the progress bar tells the truth ──────────────────────────────


def test_progress_bar_counts_rows_not_llm_calls():
    """One row, three attempts, one tick.

    The bar previously advanced on every ``on_llm_end``, so five rows retried
    three times each reported ``15it`` against a total of 5 — which reads as
    "everything passed" right before the rows turn out to be empty.
    """
    callback = BatchCallBack(total=2)
    row_a, row_b = uuid4(), uuid4()

    callback.on_llm_error(ConnectionError(), run_id=uuid4(), parent_run_id=row_a)
    callback.on_llm_error(ConnectionError(), run_id=uuid4(), parent_run_id=row_a)
    callback.on_llm_end(None, run_id=uuid4(), parent_run_id=row_a)
    callback.on_llm_end(None, run_id=uuid4(), parent_run_id=row_b)

    assert callback.count == 2, "a retried row was counted more than once"
    assert callback.llm_errors == 2


def test_progress_bar_advances_for_a_row_that_errored():
    """Without an ``on_llm_error`` handler the bar just stopped short."""
    callback = BatchCallBack(total=1)
    callback.on_llm_error(ConnectionError(), run_id=uuid4(), parent_run_id=uuid4())
    assert callback.count == 1


def test_progress_bar_handles_a_call_with_no_parent():
    """A bare model invocation has no parent run; ``run_id`` stands in."""
    callback = BatchCallBack(total=1)
    callback.on_llm_end(None, run_id=uuid4(), parent_run_id=None)
    assert callback.count == 1


# ── translation degrades to the original text ─────────────────────


def test_a_failed_translation_keeps_the_original_text(offline_run):
    """It used to write the failure default into the document itself.

    That was an empty string, so a failed translation silently blanked the
    source text and the extraction step then ran on nothing. With ``None`` as
    the default that would be worse, not better — so a failed row keeps its
    untranslated original and the run continues on real content.
    """

    class FailingTranslator(FakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            def _fail(_input):
                raise OutputParserException("no translation", llm_output="???")

            return RunnableLambda(_fail)

    out = offline_run(
        model=FailingTranslator(responses=[HR_RESPONSE]),
        task_id=999,
        run_name="trfail",
        translate=True,
        max_context_len="max",
    )

    translated = json.loads((out / "translations" / "999.json").read_text("utf-8"))
    assert translated[0]["text"] == "Alice has a heart rate of 78 bpm."
