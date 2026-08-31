import json

import pytest

from llm_extractinator.ollama_server import ModelInfo
from llm_extractinator.utils import chunk_files, parse_test_run_size, run_folder_name


def test_chunk_files_excludes_split_files(tmp_path):
    """Numeric chunk files are collected; short/long split files are ignored."""
    numeric = ["nlp-predictions-dataset-0.json", "nlp-predictions-dataset-100.json"]
    split = ["nlp-predictions-dataset-short.json", "nlp-predictions-dataset-long.json"]

    for name in numeric + split:
        (tmp_path / name).write_text("[]")

    result = {f.name for f in chunk_files(tmp_path)}

    assert result == set(numeric)
    assert not result.intersection(split)


def test_chunk_files_empty_when_no_chunks(tmp_path):
    """Returns empty list when only split files exist."""
    (tmp_path / "nlp-predictions-dataset-short.json").write_text("[]")
    (tmp_path / "nlp-predictions-dataset-long.json").write_text("[]")

    assert chunk_files(tmp_path) == []


def test_chunk_files_all_numeric(tmp_path):
    """Returns all files when only numeric chunk files exist."""
    names = [f"nlp-predictions-dataset-{i}.json" for i in range(0, 300, 100)]
    for name in names:
        (tmp_path / name).write_text("[]")

    result = {f.name for f in chunk_files(tmp_path)}
    assert result == set(names)


def test_chunk_files_are_ordered_by_row_offset(tmp_path):
    """Chunks must merge in row order, not glob or lexical order.

    Lexical sorting is the trap here: "-100" sorts before "-50", so a run
    chunked at 50 rows would emit rows 100-199 before rows 50-99.
    """
    offsets = [0, 50, 100, 150, 200]
    for offset in offsets:
        (tmp_path / f"nlp-predictions-dataset-{offset}.json").write_text("[]")

    found = [int(f.stem.split("-")[-1]) for f in chunk_files(tmp_path)]

    assert found == offsets


def test_chunk_files_does_not_delete_split_files(tmp_path):
    """Deleting chunk files leaves split files on disk (regression for the original bug)."""
    chunk_names = ["nlp-predictions-dataset-0.json", "nlp-predictions-dataset-100.json"]
    split_names = [
        "nlp-predictions-dataset-short.json",
        "nlp-predictions-dataset-long.json",
    ]

    for name in chunk_names + split_names:
        (tmp_path / name).write_text(json.dumps([{"status": "success"}]))

    for f in chunk_files(tmp_path):
        f.unlink()

    remaining = {f.name for f in tmp_path.glob("*.json")}
    assert remaining == set(split_names)


# ── Output folder naming ──────────────────────────────────────────────


def test_run_folder_name_round_trips_through_parse_test_run_size():
    """Whatever the writer names a folder, the reader must classify correctly."""
    full = run_folder_name("Task999_example", 0)
    test = run_folder_name("Task999_example", 0, test_run_size=25)

    assert full == "Task999_example-run0"
    assert test == "Task999_example-test25-run0"
    assert parse_test_run_size(full) is None
    assert parse_test_run_size(test) == 25


def test_run_folder_name_separates_sizes():
    """Different N must not collide, or the skip guard returns stale output."""
    assert run_folder_name("T", 0, 2) != run_folder_name("T", 0, 5)


def test_parse_test_run_size_ignores_task_named_test():
    """A task whose own name ends in '-test' is not a test run."""
    assert parse_test_run_size("Task001_ab-test-run0") is None


# ── Reasoning override ────────────────────────────────────────────────


def _build_task(monkeypatch, tmp_path, *, supports_thinking, **overrides):
    from llm_extractinator import prediction_task

    # Capability detection is now one /api/show call returning both the thinking
    # capability and the native context length. Constructed standalone like this,
    # PredictionTask still detects for itself (model_info is None).
    monkeypatch.setattr(
        prediction_task,
        "model_capabilities",
        lambda name, **kwargs: ModelInfo(supports_thinking=supports_thinking),
    )
    monkeypatch.setattr(
        prediction_task.PredictionTask, "initialize_model", lambda self: None
    )
    monkeypatch.setattr(prediction_task, "Predictor", lambda **kwargs: None)

    params = {p: None for p in prediction_task.PredictionTask.REQUIRED_PARAMS}
    params.update(
        model_name="some-model",
        num_predict=512,
        output_dir=tmp_path,
        run_name="run",
        ollama_host=None,
        reasoning_model=False,
        no_reasoning=False,
    )
    params.update(overrides)
    return prediction_task.PredictionTask(**params)


def _reasoning_sent(monkeypatch, tmp_path, *, supports_thinking, **overrides):
    """The `reasoning` value that actually reaches ChatOllama (Ollama's `think`)."""
    from llm_extractinator import prediction_task

    sent = {}

    def _recorder(**kwargs):
        sent.update(kwargs)
        return None

    monkeypatch.setattr(prediction_task, "ChatOllama", _recorder)
    # Capability detection is now one /api/show call returning both the thinking
    # capability and the native context length. Constructed standalone like this,
    # PredictionTask still detects for itself (model_info is None).
    monkeypatch.setattr(
        prediction_task,
        "model_capabilities",
        lambda name, **kwargs: ModelInfo(supports_thinking=supports_thinking),
    )
    monkeypatch.setattr(prediction_task, "Predictor", lambda **kw: None)

    params = {p: None for p in prediction_task.PredictionTask.REQUIRED_PARAMS}
    params.update(
        model_name="some-model",
        num_predict=512,
        output_dir=tmp_path,
        run_name="run",
        ollama_host=None,
        reasoning_model=False,
        no_reasoning=False,
    )
    params.update(overrides)
    prediction_task.PredictionTask(**params)
    return sent["reasoning"]


def test_thinking_model_asks_ollama_to_think(monkeypatch, tmp_path):
    assert _reasoning_sent(monkeypatch, tmp_path, supports_thinking=True) is True


def test_no_reasoning_sends_an_explicit_off_to_a_thinking_model(monkeypatch, tmp_path):
    """`None` would mean "use your default", which for an always-on model is *on*.

    Only an explicit False actually stops the model thinking, so switching
    reasoning off has to send False rather than simply not asking for it.
    """
    assert (
        _reasoning_sent(
            monkeypatch, tmp_path, supports_thinking=True, no_reasoning=True
        )
        is False
    )


def test_no_reasoning_stays_silent_for_a_non_thinking_model(monkeypatch, tmp_path):
    """Ollama answers 400 if `think` is sent to a model without the capability.

    There is nothing to disable on an ordinary model, so the flag must be a
    no-op rather than an explicit False.
    """
    assert (
        _reasoning_sent(
            monkeypatch, tmp_path, supports_thinking=False, no_reasoning=True
        )
        is None
    )


def test_plain_model_leaves_thinking_unset(monkeypatch, tmp_path):
    assert _reasoning_sent(monkeypatch, tmp_path, supports_thinking=False) is None


def test_thinking_model_is_auto_detected(monkeypatch, tmp_path):
    task = _build_task(monkeypatch, tmp_path, supports_thinking=True)
    assert task._is_thinking is True


def test_detection_does_not_change_the_generation_budget(monkeypatch, tmp_path):
    """The reasoning allowance is added before the window is sized, not here.

    ``PredictionTask.__init__`` used to add 5000 to ``num_predict`` on detecting
    a thinking model — after ``TaskRunner`` had already sized the window from the
    un-bumped value, so the answer could be given a budget several times larger
    than the window meant to hold it. The allowance now comes from
    ``TaskRunner._output_budget``, which runs after the model is inspected and
    before anything is sized. This object must leave the number alone.
    """
    for supports_thinking in (True, False):
        task = _build_task(
            monkeypatch, tmp_path, supports_thinking=supports_thinking
        )
        assert task.num_predict == 512, (
            "PredictionTask changed num_predict; the window was sized elsewhere "
            "and no longer matches it"
        )


def test_no_reasoning_overrides_auto_detection(monkeypatch, tmp_path):
    """The whole point of the flag: a detected thinking model can be switched off."""
    task = _build_task(monkeypatch, tmp_path, supports_thinking=True, no_reasoning=True)
    assert task._is_thinking is False


def test_no_reasoning_overrides_explicit_reasoning_model(monkeypatch, tmp_path):
    task = _build_task(
        monkeypatch,
        tmp_path,
        supports_thinking=False,
        reasoning_model=True,
        no_reasoning=True,
    )
    assert task._is_thinking is False


def test_reasoning_model_forces_on_when_detection_fails(monkeypatch, tmp_path):
    """Detection returns False for a model that is not pulled yet."""
    task = _build_task(
        monkeypatch, tmp_path, supports_thinking=False, reasoning_model=True
    )
    assert task._is_thinking is True
    assert task.num_predict == 512  # explicit request: TaskRunner handles the budget


def test_plain_model_gets_no_reasoning(monkeypatch, tmp_path):
    task = _build_task(monkeypatch, tmp_path, supports_thinking=False)
    assert task._is_thinking is False
    assert task.num_predict == 512
