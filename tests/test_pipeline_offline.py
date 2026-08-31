"""End-to-end pipeline tests that run fully offline (no Ollama, no models).

These assert that the pipeline *plumbing* works: prompts build, the schema is
bound, output is parsed into the parser model, results are merged back onto the
input rows, and files land on disk. Because the model is faked and
deterministic, we can assert exact values here without flakiness — the value
being checked is the one the fake returned, so a mismatch means the plumbing
(parsing/merging/writing) broke, not that "the model was wrong".

Whether a *real* model produces sensible output is a separate concern, covered
by the opt-in smoke test in ``test_pipeline_smoke.py``.
"""

import pytest

from llm_extractinator.main import TaskConfig
from tests.conftest import load_predictions

HR_RESPONSE = '{"HR": 78, "Name": "Alice"}'
PRODUCTS_RESPONSE = '{"products": [{"name": "Widget", "price": 9.99}]}'


def test_zero_shot_runs_end_to_end(offline_run):
    out = offline_run(responses=[HR_RESPONSE], task_id=999, run_name="zs")
    preds = load_predictions(out, "zs", "Task999_example")

    assert len(preds) == 5  # one per input row
    for p in preds:
        assert {"HR", "Name", "status"} <= p.keys()
        assert p["status"] == "success"
        assert p["HR"] == 78 and p["Name"] == "Alice"


def test_input_columns_are_preserved(offline_run):
    """Original input fields must survive into the output rows."""
    out = offline_run(responses=[HR_RESPONSE], task_id=999, run_name="keep")
    preds = load_predictions(out, "keep", "Task999_example")

    for p in preds:
        assert "text" in p  # the input column
        assert "expected_output" in p  # any extra input columns pass through
        assert "token_count" in p  # added by the data loader


def test_nested_list_schema_runs(offline_run):
    """Task 998 has a nested list-of-objects parser (products)."""
    out = offline_run(responses=[PRODUCTS_RESPONSE], task_id=998, run_name="prod")
    preds = load_predictions(out, "prod", "Task998_example2")

    for p in preds:
        assert p["status"] == "success"
        assert isinstance(p["products"], list)
        assert p["products"][0]["name"] == "Widget"
        assert isinstance(p["products"][0]["price"], (int, float))


def test_few_shot_runs(offline_run):
    """num_examples > 0 exercises the example-selector / embeddings branch."""
    out = offline_run(
        responses=[PRODUCTS_RESPONSE], task_id=998, run_name="fs", num_examples=2
    )
    preds = load_predictions(out, "fs", "Task998_example2")
    assert all(p["status"] == "success" for p in preds)


@pytest.mark.parametrize("max_context_len", ["max", 512, "split"])
def test_context_length_modes(offline_run, max_context_len):
    out = offline_run(
        responses=[HR_RESPONSE],
        task_id=999,
        run_name=f"ctx_{max_context_len}",
        max_context_len=max_context_len,
    )
    preds = load_predictions(out, f"ctx_{max_context_len}", "Task999_example")
    assert len(preds) == 5  # split mode must recombine short + long
    assert all(p["status"] == "success" for p in preds)


def test_multiple_runs_produce_separate_folders(offline_run):
    out = offline_run(responses=[HR_RESPONSE], task_id=999, run_name="multi", n_runs=3)
    for run_idx in range(3):
        preds = load_predictions(out, "multi", "Task999_example", run_idx=run_idx)
        assert len(preds) == 5


def test_chunking_recombines_all_rows(offline_run):
    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="chunk", chunk_size=2
    )
    preds = load_predictions(out, "chunk", "Task999_example")
    assert len(preds) == 5  # 2 + 2 + 1, merged back to one file


def test_chunking_preserves_input_row_order(offline_run):
    """Merged chunks must come back in input order.

    Chunk files are named by row offset, so merging them in glob order (or in
    lexical order, where "-100" sorts before "-50") silently scrambles the rows
    relative to the input - which breaks any join back onto the source data.
    """
    unchunked = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="order_ref"
    )
    expected = [
        p["text"] for p in load_predictions(unchunked, "order_ref", "Task999_example")
    ]

    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="order_chunk", chunk_size=2
    )
    preds = load_predictions(out, "order_chunk", "Task999_example")

    assert [p["text"] for p in preds] == expected


def test_translation_mode_runs(offline_run):
    # Translation reserves the longest input plus a 5000-token buffer for the
    # translated text, so it needs a window fitted to that rather than the small
    # fixed one the other tests use. Before the budget resolver this mismatch was
    # invisible: a 512-token window was sent with a ~5000-token output budget.
    out = offline_run(
        responses=[HR_RESPONSE],
        translation="Alice has a heart rate of 78 bpm.",
        task_id=999,
        run_name="tr",
        translate=True,
        max_context_len="max",
    )
    preds = load_predictions(out, "tr", "Task999_example")
    assert all(p["status"] == "success" for p in preds)
    assert (out / "translations" / "999.json").exists()


def test_test_run_size_limits_to_first_n_rows(offline_run):
    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="tr_size", test_run_size=3
    )
    preds = load_predictions(out, "tr_size", "Task999_example", suffix="-test3")
    assert len(preds) == 3
    assert [p["text"] for p in preds] == [
        "Alice has a heart rate of 78 bpm.",
        "The heart rate recorded for Bob is 90 bpm.",
        "Charlie’s pulse is measured at 72 bpm.",
    ]


def test_test_run_size_larger_than_dataset_uses_full_dataset(offline_run):
    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="tr_big", test_run_size=10
    )
    preds = load_predictions(out, "tr_big", "Task999_example", suffix="-test10")
    assert len(preds) == 5


def test_test_run_random_is_reproducible_via_seed(offline_run):
    kwargs = dict(
        responses=[HR_RESPONSE],
        task_id=999,
        test_run_size=3,
        test_run_random=True,
        seed=7,
    )
    out1 = offline_run(run_name="tr_rand1", **kwargs)
    out2 = offline_run(run_name="tr_rand2", **kwargs)
    preds1 = load_predictions(out1, "tr_rand1", "Task999_example", suffix="-test3")
    preds2 = load_predictions(out2, "tr_rand2", "Task999_example", suffix="-test3")
    assert len(preds1) == 3
    assert [p["text"] for p in preds1] == [p["text"] for p in preds2]


def test_test_run_size_with_split_context_falls_back_to_max(offline_run):
    """split mode is degenerate on a tiny sample; the runner should fall back to 'max'."""
    out = offline_run(
        responses=[HR_RESPONSE],
        task_id=999,
        run_name="tr_split",
        test_run_size=3,
        max_context_len="split",
    )
    preds = load_predictions(out, "tr_split", "Task999_example", suffix="-test3")
    assert len(preds) == 3
    assert all(p["status"] == "success" for p in preds)


def test_test_run_sizes_do_not_reuse_each_others_output(offline_run):
    """Re-running at a different N must not hand back the earlier run's file.

    The skip-if-exists guard keys off the output folder, so if N were not part
    of the folder name a 5-row test run after a 2-row one would silently return
    the 2-row results (with overwrite off, which is the Studio default).
    """
    out = offline_run(
        responses=[HR_RESPONSE],
        task_id=999,
        run_name="sizes",
        test_run_size=2,
        overwrite=False,
    )
    out = offline_run(
        responses=[HR_RESPONSE],
        task_id=999,
        run_name="sizes",
        test_run_size=5,
        overwrite=False,
        output_dir=out,
    )

    assert len(load_predictions(out, "sizes", "Task999_example", suffix="-test2")) == 2
    assert len(load_predictions(out, "sizes", "Task999_example", suffix="-test5")) == 5


def test_test_run_folder_is_recognised_by_the_studio(offline_run):
    """The Studio's test-run marker must match the folder name the backend writes."""
    from llm_extractinator.utils import parse_test_run_size

    out = offline_run(
        responses=[HR_RESPONSE], task_id=999, run_name="marker", test_run_size=3
    )
    folders = [p.name for p in (out / "marker").iterdir() if p.is_dir()]

    assert [parse_test_run_size(name) for name in folders] == [3]
    # A task whose own name ends in "-test" must not be mistaken for a test run.
    assert parse_test_run_size("Task001_ab-test-run0") is None
    assert parse_test_run_size("Task999_example-run0") is None


def test_test_run_context_is_sized_from_the_full_dataset(record_task):
    """A test run must preview the run it stands in for, not a smaller one.

    Context length is fitted to the longest document. Sizing it from the sampled
    subset would let a 3-row check pass with a window the full run never uses -
    and the long tail is exactly what a pre-flight check should catch.
    """
    full = record_task(responses=[HR_RESPONSE], task_id=999, run_name="ctx_full")
    subset = record_task(
        responses=[HR_RESPONSE], task_id=999, run_name="ctx_sub", test_run_size=2
    )

    assert subset["max_context_len"] == full["max_context_len"]


def test_max_context_cap_bounds_the_window(record_task):
    """The cap is a ceiling on the fitted window, not a fixed size."""
    uncapped = record_task(responses=[HR_RESPONSE], task_id=999, run_name="cap_off")
    # Derive the cap from the uncapped estimate so this keeps testing the cap
    # rather than a hard-coded number that context-sizing changes invalidate.
    tight = int(uncapped["max_context_len"]) - 1
    capped = record_task(
        responses=[HR_RESPONSE], task_id=999, run_name="cap_on", max_context_cap=tight
    )
    generous = record_task(
        responses=[HR_RESPONSE],
        task_id=999,
        run_name="cap_high",
        max_context_cap=10_000_000,
    )

    assert capped["max_context_len"] == tight
    assert generous["max_context_len"] == uncapped["max_context_len"]


def test_max_context_cap_zero_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        TaskConfig(
            task_id=999,
            max_context_cap=0,
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "output" / "logs",
        )


def test_context_cap_below_the_output_budget_is_rejected(tmp_path):
    """num_ctx is the whole window, so a ceiling under num_predict leaves no prompt."""
    with pytest.raises(ValueError, match="must exceed num_predict"):
        TaskConfig(
            task_id=999,
            max_context_cap=256,
            num_predict=512,
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "output" / "logs",
        )


def test_context_cap_above_the_output_budget_is_accepted(tmp_path):
    TaskConfig(
        task_id=999,
        max_context_cap=8192,
        num_predict=512,
        output_dir=tmp_path / "output",
        log_dir=tmp_path / "output" / "logs",
    )


def test_test_run_size_zero_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        TaskConfig(
            task_id=999,
            test_run_size=0,
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "output" / "logs",
        )


def test_malformed_output_degrades_gracefully(offline_run):
    """A model that emits non-JSON must not crash the run.

    Every row is marked ``failure`` and still carries every schema key, so
    downstream code can rely on the shape — but the values are ``None``, not
    type defaults. See ``test_validator.py`` for why that distinction matters.
    """
    out = offline_run(responses=["this is not valid json"], task_id=999, run_name="bad")
    preds = load_predictions(out, "bad", "Task999_example")

    assert len(preds) == 5
    for p in preds:
        assert p["status"] == "failure"
        assert {"HR", "Name"} <= p.keys()  # shape preserved
        assert p["HR"] is None and p["Name"] is None  # but nothing invented
