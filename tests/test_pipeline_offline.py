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


def test_translation_mode_runs(offline_run):
    out = offline_run(
        responses=[HR_RESPONSE],
        translation="Alice has a heart rate of 78 bpm.",
        task_id=999,
        run_name="tr",
        translate=True,
    )
    preds = load_predictions(out, "tr", "Task999_example")
    assert all(p["status"] == "success" for p in preds)
    assert (out / "translations" / "999.json").exists()


def test_malformed_output_degrades_gracefully(offline_run):
    """A model that emits non-JSON must not crash the run.

    Every row should be marked ``failure`` and still carry default-valued
    fields so downstream code can rely on the schema being present.
    """
    out = offline_run(responses=["this is not valid json"], task_id=999, run_name="bad")
    preds = load_predictions(out, "bad", "Task999_example")

    assert len(preds) == 5
    for p in preds:
        assert p["status"] == "failure"
        assert {"HR", "Name"} <= p.keys()  # defaults still filled in
