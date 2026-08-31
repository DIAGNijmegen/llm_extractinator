"""The context-budget invariant: prompt + num_predict must fit inside num_ctx.

``num_ctx`` is not a knob that sits next to ``num_predict`` — it is the whole
window, and the generated answer is a slice of it. Every context bug in this
codebase has been a variation on treating the two as independent, so this module
asserts the relationship itself rather than any particular number:

    prompt_tokens + num_predict <= num_ctx

``prompt_tokens`` is measured from the messages the model was *actually* asked to
complete, not recomputed from the estimator, so an estimator that disagrees with
the pipeline shows up here rather than cancelling itself out.

``test_budget.py`` asserts the same relationship on the pure resolver, which is
how you find out *why* it broke. This module is how you find out *that* it did.

What this does and does not prove
---------------------------------
It proves the arithmetic is *self-consistent*: what we reserved is at least what
we sent. It does not prove the count is *right* — that needs the model's own
tokenizer, and is the job of the ``prompt_eval_count`` calibration (see
``context_budget_design.md``, phase 4). A tokenizer that under-counts by 20%
would satisfy every assertion here and still truncate on a real run.

Baseline, measured against the 0.6.1 tree before the budget work
----------------------------------------------------------------
28 ``ChatOllama`` constructions across the 24 cells (``split`` on a full run
builds two, one for short cases and one for long):

===================  ==========================================
thinking=False       14/14 records satisfied the invariant
thinking=True        14/14 records violated it
===================  ==========================================

Every thinking cell failed, in all three context modes, both run sizes and both
shot counts, because ``PredictionTask.__init__`` added 5000 to ``num_predict``
after ``TaskRunner._context_len`` had already sized the window. Representative
numbers: ``num_ctx=780`` against ``num_predict=5512``.

Those figures were measured where tiktoken's vocabulary could not be fetched, so
``count_tokens`` was on its word-count fallback; on a machine with the real
encoding they differ. The pass/fail split does not — the estimator and the
measurement go through the same counter, so the inequality is scale-free.

All 24 cells pass since the reorder. The model is now inspected before the
window is sized, so the reasoning allowance is part of what gets sized rather
than something added behind it. The ``xfail(strict=True)`` markers those cells
carried are gone, which is what making them strict was for.
"""

import itertools

import pytest

from llm_extractinator.budget import ContextBudgetError
from tests.conftest import _NoopOllamaManager, load_predictions

PRODUCTS_RESPONSE = '{"products": [{"name": "Widget", "price": 9.99}]}'

# Task 998 is the one with an Example_Path, so it can run both zero- and
# few-shot; holding the task fixed keeps num_examples the only variable.
TASK_ID = 998

# Task 999 has the simple HR/Name schema, used where row order matters.
HR_RESPONSE_999 = '{"HR": 78, "Name": "Alice"}'

# A fixed window comfortably larger than anything these fixtures build — the
# reasoning allowance included, so the thinking cells exercise the budget
# arithmetic rather than the refusal. A window too small for the allowance is
# its own test below.
ROOMY_FIXED_WINDOW = 16_384


def _cells():
    for thinking, ctx_mode, test_run, shots in itertools.product(
        [False, True], ["max", ROOMY_FIXED_WINDOW, "split"], [False, True], [0, 2]
    ):
        cell_id = (
            f"{'thinking' if thinking else 'plain'}"
            f"-{ctx_mode}"
            f"-{'testrun' if test_run else 'full'}"
            f"-{shots}shot"
        )
        yield pytest.param(thinking, ctx_mode, test_run, shots, id=cell_id)


@pytest.mark.parametrize("thinking,ctx_mode,test_run,shots", list(_cells()))
def test_generation_budget_fits_inside_the_context_window(
    record_model_kwargs, thinking, ctx_mode, test_run, shots
):
    records = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE],
        thinking=thinking,
        task_id=TASK_ID,
        run_name="budget",
        num_examples=shots,
        max_context_len=ctx_mode,
        test_run_size=2 if test_run else None,
    )

    for phase, record in enumerate(records):
        where = f"phase {phase} of {len(records)}"
        num_ctx = record["num_ctx"]
        num_predict = record["num_predict"]
        prompt = record["prompt_tokens"]

        assert num_predict < num_ctx, (
            f"{where}: the generation budget is not smaller than the whole "
            f"window — num_predict={num_predict}, num_ctx={num_ctx}. There is "
            f"no room left for a prompt."
        )
        assert prompt + num_predict <= num_ctx, (
            f"{where}: prompt ({prompt}) + num_predict ({num_predict}) = "
            f"{prompt + num_predict} exceeds num_ctx ({num_ctx}) by "
            f"{prompt + num_predict - num_ctx} tokens."
        )


def test_thinking_detection_actually_reaches_the_model(record_model_kwargs):
    """Guards the matrix above against silently testing nothing.

    If the fixture stopped driving auto-detection, every thinking cell would
    behave like a plain one and the matrix would pass while covering half of
    what it claims. This says it directly: a model detected as thinking must be
    given a larger generation budget than one that is not.
    """
    plain = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
        run_name="plain", max_context_len=ROOMY_FIXED_WINDOW,
    )
    thinking = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=True, task_id=TASK_ID,
        run_name="think", max_context_len=ROOMY_FIXED_WINDOW,
    )

    assert thinking[0]["num_predict"] > plain[0]["num_predict"], (
        "Auto-detected thinking model got the same generation budget as an "
        "ordinary one; the reasoning allowance is not being applied."
    )


def test_the_window_grows_with_the_reasoning_allowance(record_model_kwargs):
    """The reorder in one assertion.

    A fitted window has to *contain* the allowance. Before, the allowance was
    added after the fitting, so the window stayed the size it would have been
    for a non-thinking model while the budget grew past it.
    """
    plain = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
        run_name="grow_plain", max_context_len="max",
    )
    thinking = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=True, task_id=TASK_ID,
        run_name="grow_think", max_context_len="max",
    )

    grew_by = thinking[0]["num_ctx"] - plain[0]["num_ctx"]
    budget_grew_by = thinking[0]["num_predict"] - plain[0]["num_predict"]
    assert grew_by == budget_grew_by, (
        f"the window grew by {grew_by} but the generation budget grew by "
        f"{budget_grew_by}; the allowance is not being sized in"
    )


def test_split_mode_sizes_each_phase_independently(record_model_kwargs):
    """``split`` runs the pipeline twice, so both windows must be checked.

    A budget change that only fixed the first phase would still ship a broken
    long-case run, and the long cases are exactly the ones that overflow.
    """
    records = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
        run_name="split", max_context_len="split",
    )

    assert len(records) == 2, (
        f"split mode should build two models (short cases, long cases), "
        f"got {len(records)}"
    )
    for record in records:
        assert record["prompt_tokens"] + record["num_predict"] <= record["num_ctx"]


def test_split_returns_rows_in_input_order(offline_run):
    """``split`` runs the two halves separately and concatenates them.

    Without restoring the original positions the output comes back ordered by
    document length — every short row, then every long one — which silently
    breaks a positional join back onto the source data. That is the same failure
    chunked runs had before their merge was ordered by row offset.
    """
    import json

    from tests.conftest import DATA_DIR

    out = offline_run(
        responses=[HR_RESPONSE_999], task_id=999, run_name="order",
        max_context_len="split",
    )
    rows = load_predictions(out, "order", "Task999_example")
    source = [row["text"] for row in json.loads(
        (DATA_DIR / "testdata.json").read_text(encoding="utf-8")
    )]

    assert [row["text"] for row in rows] == source


def test_split_leaves_no_bookkeeping_column_in_the_output(offline_run):
    """The position marker is an implementation detail, not a result column."""
    out = offline_run(
        responses=[HR_RESPONSE_999], task_id=999, run_name="clean_cols",
        max_context_len="split",
    )
    rows = load_predictions(out, "clean_cols", "Task999_example")

    assert not any(key.startswith("_") for key in rows[0])
    assert "extractinator_source_row" not in rows[0]


def test_split_writes_one_failures_file_for_the_merged_run(offline_run):
    """The predictions are merged and the per-phase files deleted, so leaving
    ``failures-short.json`` and ``failures-long.json`` behind would mean two
    artefacts whose row numbers point at files that no longer exist."""
    out = offline_run(
        responses=["not json at all"], task_id=999, run_name="split_fail",
        max_context_len="split",
    )
    run_dir = out / "split_fail" / "Task999_example-run0"
    written = sorted(p.name for p in run_dir.iterdir())

    assert written == ["failures.json", "nlp-predictions-dataset.json"]

    import json as _json

    failures = _json.loads((run_dir / "failures.json").read_text("utf-8"))
    assert [f["row"] for f in failures] == [0, 1, 2, 3, 4]


def test_the_server_is_started_once_for_the_whole_run(record_model_kwargs):
    """The structural half of the reorder.

    The model has to be pulled before it can be inspected, and it has to be
    inspected before the window can be sized — so the server now comes up once,
    at the top of the run, rather than being started and stopped around each
    phase. In ``split`` mode the old shape started, pulled and stopped twice,
    which unloaded the model from VRAM between the short and long halves only to
    load it again.
    """
    _NoopOllamaManager.calls.clear()
    record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
        run_name="lifecycle", max_context_len="split",
    )

    assert _NoopOllamaManager.calls.count("pull") == 1, (
        f"split mode pulled the model {_NoopOllamaManager.calls.count('pull')} "
        f"times: {_NoopOllamaManager.calls}"
    )
    assert _NoopOllamaManager.calls.count("start") == 1
    assert _NoopOllamaManager.calls.count("stop") == 1


# ── ceilings and refusals ─────────────────────────────────────────


def test_the_models_native_context_caps_the_window(record_model_kwargs):
    """A window larger than the model supports is not a larger window.

    ``/api/show`` reports the native context length, but nothing read it until
    the reorder made the model available before sizing — so a fitted window
    could ask for more than the model could use.
    """
    records = record_model_kwargs(
        responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
        run_name="native", max_context_len=100_000, native_context=2048,
    )
    assert records[0]["num_ctx"] == 2048


def test_fixed_window_smaller_than_the_output_budget_is_refused(record_model_kwargs):
    """A fixed window under num_predict leaves no room for any input at all.

    ``TaskConfig`` validates ``max_context_cap`` against ``num_predict`` but
    never validated a fixed ``max_context_len``, so this reached the model
    unchallenged — and Ollama truncates silently rather than complaining.
    """
    with pytest.raises(ContextBudgetError, match="must be smaller than num_ctx"):
        record_model_kwargs(
            responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
            run_name="tiny", max_context_len=256, num_predict=512,
        )


def test_a_fixed_window_too_small_for_the_reasoning_allowance_is_refused(
    record_model_kwargs,
):
    """A thinking model spends thousands of tokens before it writes any JSON.

    This is not something the resolver can quietly fix by shrinking something:
    the answer would be truncated inside ``<think>`` and no JSON would arrive.
    The message names the knob that made the window that size.
    """
    with pytest.raises(ContextBudgetError, match="--max_context_len"):
        record_model_kwargs(
            responses=[PRODUCTS_RESPONSE], thinking=True, task_id=TASK_ID,
            run_name="tight", max_context_len=4096,
        )


def test_a_configuration_error_is_not_reported_as_a_completed_run(record_model_kwargs):
    """``extractinate`` swallows exceptions, which is wrong for a config error.

    A budget that cannot work fails every row, so letting it fall through to
    "Task execution completed" — with no output written — is exactly the
    silent-success behaviour this work exists to remove.
    """
    with pytest.raises(ContextBudgetError):
        record_model_kwargs(
            responses=[PRODUCTS_RESPONSE], thinking=False, task_id=TASK_ID,
            run_name="fatal", max_context_len=128, num_predict=1024,
        )
