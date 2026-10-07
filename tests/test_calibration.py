"""Checking the token estimate against the model's own count.

``count_tokens`` uses OpenAI's ``cl100k_base`` against Qwen and Phi models,
drifts further on non-English text, and never sees the chat template's role
markers at all. Everything else in this codebase treats that as an estimate with
a margin. This is the one place it gets compared with the truth — and the truth
is only available after the server is up and the prompt is assembled, which is
what the phase 3 reorder made possible.
"""

import json

import pytest

from llm_extractinator.budget import ContextBudgetError
from llm_extractinator.ollama_server import measure_prompt_tokens
from tests.conftest import DATA_DIR, PROBE_CALLS, load_predictions

HR_RESPONSE = '{"HR": 78, "Name": "Alice"}'
PRODUCTS_RESPONSE = '{"products": [{"name": "Widget", "price": 9.99}]}'
MESSAGES = [
    {"role": "system", "content": "You are an extraction system."},
    {"role": "user", "content": "Alice has a heart rate of 78 bpm."},
]


class _FakeOllamaClient:
    """Stands in for ollama.Client, recording how it was called."""

    last_call = {}

    def __init__(self, host=None):
        type(self).last_call = {"host": host}

    def chat(self, model, messages, options=None):
        type(self).last_call.update(
            model=model, messages=messages, options=options or {}
        )
        return dict(self.response)


def _client_returning(response):
    return type("Stub", (_FakeOllamaClient,), {"response": response})


# ── measure_prompt_tokens ─────────────────────────────────────────


def test_the_models_own_count_is_returned(monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "Client", _client_returning({"prompt_eval_count": 527}))
    assert measure_prompt_tokens("some-model", MESSAGES) == 527


def test_only_one_token_is_generated(monkeypatch):
    """The prompt is the thing being measured; the answer is pure waste."""
    import ollama

    stub = _client_returning({"prompt_eval_count": 42})
    monkeypatch.setattr(ollama, "Client", stub)
    measure_prompt_tokens("some-model", MESSAGES, num_ctx=4096)

    assert stub.last_call["options"]["num_predict"] == 1
    assert stub.last_call["options"]["num_ctx"] == 4096


def test_an_object_response_works_too(monkeypatch):
    """Newer ollama clients return a pydantic model rather than a dict."""
    import ollama

    class _Response:
        prompt_eval_count = 314

    class _Client:
        def __init__(self, host=None):
            pass

        def chat(self, model, messages, options=None):
            return _Response()

    monkeypatch.setattr(ollama, "Client", _Client)
    assert measure_prompt_tokens("some-model", MESSAGES) == 314


def test_an_implausible_count_is_ignored(monkeypatch):
    """``prompt_eval_count`` has a history of being wrong in one direction.

    ollama-python#271 reports it stuck at 1026 tokens across prompts of ~57,000
    characters. An under-reported count would make the estimate look safer than
    it is, which is the only direction that can hurt — so a count that cannot be
    a real tokenization of the text is discarded rather than believed.
    """
    import ollama

    long_messages = [{"role": "user", "content": "x" * 57_000}]
    monkeypatch.setattr(
        ollama, "Client", _client_returning({"prompt_eval_count": 1026})
    )
    assert measure_prompt_tokens("some-model", long_messages) is None


def test_a_missing_or_zero_count_is_ignored(monkeypatch):
    import ollama

    for response in ({}, {"prompt_eval_count": 0}, {"prompt_eval_count": None}):
        monkeypatch.setattr(ollama, "Client", _client_returning(response))
        assert measure_prompt_tokens("some-model", MESSAGES) is None


def test_an_unreachable_server_is_not_an_error(monkeypatch):
    """A calibration that cannot be performed leaves the estimate unverified.
    That is where things stood before this existed; it must not stop a run."""
    import ollama

    class _Exploding:
        def __init__(self, host=None):
            raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(ollama, "Client", _Exploding)
    assert measure_prompt_tokens("some-model", MESSAGES) is None


# ── the prompt that gets measured is the prompt that gets sent ────


def test_assembled_prompt_uses_ollama_role_names(offline_run):
    """LangChain calls them system/human/ai; Ollama wants system/user/assistant.

    Measuring a prompt with the wrong role names would still return a number,
    just not the number the real request produces.
    """
    PROBE_CALLS.clear()
    offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="roles",
        measured_prompt_tokens=10,
    )

    messages = PROBE_CALLS[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "heart rate" in messages[1]["content"]


def test_the_longest_document_is_the_one_measured(offline_run):
    """Calibrating on a short row would prove nothing about the long tail."""
    PROBE_CALLS.clear()
    offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="longest",
        measured_prompt_tokens=10,
    )

    data = json.loads((DATA_DIR / "testdata.json").read_text(encoding="utf-8"))
    longest = max((row["text"] for row in data), key=len)
    assert PROBE_CALLS[0]["messages"][-1]["content"] == longest


def test_the_probe_uses_the_window_the_run_will_use(offline_run):
    """Measured under a different num_ctx is a different measurement."""
    PROBE_CALLS.clear()
    offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="ctx",
        max_context_len=4096, measured_prompt_tokens=10,
    )
    assert PROBE_CALLS[0]["num_ctx"] == 4096


# ── what the measurement is used for ──────────────────────────────


def test_a_verified_estimate_lets_the_run_proceed(offline_run):
    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="fits",
        measured_prompt_tokens=100,  # well inside the 448 reserved
    )
    assert len(load_predictions(out, "fits", "Task999_example")) == 5


def test_an_explicit_window_is_not_overridden_when_the_estimate_was_under(
    offline_run,
):
    """``--max_context_len N`` is the user's decision, so it is not grown.

    The failure being prevented is silent: if the real prompt is larger than the
    window reserved for it, Ollama drops the beginning of every long document
    and returns worse output with no indication why.
    """
    with pytest.raises(ContextBudgetError, match="does not fit the window"):
        offline_run(
            responses=[HR_RESPONSE], task_id=999, run_name="under",
            measured_prompt_tokens=500,  # reserved is 512 - 64 = 448
        )


def test_the_refusal_says_what_to_change(offline_run):
    with pytest.raises(ContextBudgetError) as raised:
        offline_run(
            responses=[HR_RESPONSE], task_id=999, run_name="advice",
            measured_prompt_tokens=500,
        )

    message = str(raised.value)
    assert "--max_context_len" in message
    assert "--num_examples" in message
    assert "52 more than the 448 reserved" in message


def test_a_fitted_window_grows_to_the_measured_prompt(record_model_kwargs):
    """A measurement is exact, so the window can be corrected rather than refused.

    When the window was fitted to the data — no explicit ``--max_context_len`` —
    an estimate that came out under is the estimator's problem, not the user's.
    Re-resolving from the measured number produces a window that is right, and
    the model is rebuilt against it.
    """
    records = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], task_id=998, run_name="grow",
        max_context_len="max", num_predict=64, measured_prompt_tokens=400,
    )

    assert len(records) == 2, (
        "expected the model to be rebuilt against the corrected window, "
        f"got {len(records)} construction(s)"
    )
    first, second = records
    assert second["num_ctx"] > first["num_ctx"]
    assert second["num_ctx"] >= 400 + second["num_predict"], (
        "the corrected window still does not hold the measured prompt"
    )


def test_growth_is_blocked_by_a_hardware_cap_and_the_run_stops(record_model_kwargs):
    """The cap exists to keep the run inside VRAM; correcting past it would
    trade a truncated prompt for an out-of-memory failure."""
    with pytest.raises(ContextBudgetError, match="--max_context_cap"):
        record_model_kwargs(
            responses=[PRODUCTS_RESPONSE], task_id=998, run_name="capped",
            max_context_len="max", num_predict=64, max_context_cap=300,
            measured_prompt_tokens=400,
        )


def test_a_verified_estimate_does_not_rebuild_the_model(record_model_kwargs):
    """Correction is for when the estimate was wrong, not a routine step."""
    records = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], task_id=998, run_name="nogrow",
        max_context_len="max", num_predict=64, measured_prompt_tokens=50,
    )
    assert len(records) == 1


def test_an_unverifiable_estimate_does_not_stop_the_run(offline_run):
    """No server, no measurement, no problem — just an unverified estimate."""
    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="unverified",
        measured_prompt_tokens=None,
    )
    assert len(load_predictions(out, "unverified", "Task999_example")) == 5


# ── the refusal names what actually blocked it ────────────────────


def _refusal(record_model_kwargs, **kwargs) -> str:
    with pytest.raises(ContextBudgetError) as raised:
        record_model_kwargs(
            responses=[PRODUCTS_RESPONSE], task_id=998, max_context_len="max",
            **kwargs,
        )
    return str(raised.value)


def test_a_native_limit_is_not_something_to_raise(record_model_kwargs):
    """"Raise the model's native context length" was advice nobody can follow."""
    message = _refusal(
        record_model_kwargs, run_name="native", num_predict=64,
        native_context=300, measured_prompt_tokens=400,
    )

    assert "Raise the model's native context length" not in message
    assert "use a model with a larger native context" in message
    assert "64 are reserved for the answer" in message
    assert "--num_predict" in message


def test_a_cap_limited_refusal_names_the_cap(record_model_kwargs):
    message = _refusal(
        record_model_kwargs, run_name="cap", num_predict=64,
        max_context_cap=300, measured_prompt_tokens=400,
    )

    assert "--max_context_cap holds the window at 300" in message
    assert "raise it if your GPU allows" in message


def test_the_refusal_separates_the_reasoning_allowance(record_model_kwargs):
    """On a thinking model the allowance is most of the reservation.

    The colleague's case: 11,664 of 16,384 went to the answer, and the message
    never said so.
    """
    from llm_extractinator.budget import REASONING_ALLOWANCE

    message = _refusal(
        record_model_kwargs, run_name="thinking", thinking=True, num_predict=64,
        native_context=REASONING_ALLOWANCE + 1000, measured_prompt_tokens=2000,
    )

    assert f"{REASONING_ALLOWANCE + 64} are reserved for the answer" in message
    assert "64 for the JSON answer" in message
    assert f"plus {REASONING_ALLOWANCE} for the model's reasoning" in message


def test_the_refusal_no_longer_blames_tokenizer_drift(record_model_kwargs):
    """It is raised after the model's own count, so drift cannot be the cause."""
    message = _refusal(
        record_model_kwargs, run_name="drift", num_predict=64,
        max_context_cap=300, measured_prompt_tokens=400,
    )

    assert "OpenAI" not in message
    assert "drifts" not in message
