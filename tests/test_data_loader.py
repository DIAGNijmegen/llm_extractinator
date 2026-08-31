import json

import pandas as pd
import pytest

from llm_extractinator.data_loader import DataLoader, TaskLoader


# ── DataLoader.validate_file ──────────────────────────────────────


def test_validate_file_json_ok(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("[]")
    DataLoader().validate_file(f)  # should not raise


def test_validate_file_csv_ok(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2")
    DataLoader().validate_file(f)  # should not raise


def test_validate_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataLoader().validate_file(tmp_path / "ghost.json")


def test_validate_file_unsupported_extension(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file format"):
        DataLoader().validate_file(f)


# ── DataLoader.count_tokens ───────────────────────────────────────


def test_count_tokens_returns_positive_int():
    dl = DataLoader()
    assert dl.count_tokens("Hello world this is a test") > 0


def test_count_tokens_empty_string():
    dl = DataLoader()
    assert dl.count_tokens("") == 0


def test_count_tokens_fallback_on_invalid_encoding():
    dl = DataLoader()
    # Provide a bogus encoding name to trigger the word-count fallback
    result = dl.count_tokens("hello world foo bar", model_name="not_a_real_encoding")
    assert isinstance(result, int)
    assert result > 0


# ── DataLoader.add_token_count ────────────────────────────────────


def test_add_token_count_adds_column():
    dl = DataLoader()
    df = pd.DataFrame({"text": ["hello world", "foo bar baz"]})
    result = dl.add_token_count(df, text_column="text", token_column="token_count")
    assert "token_count" in result.columns
    assert (result["token_count"] > 0).all()


# ── DataLoader.split_data ─────────────────────────────────────────


def test_split_data_partitions_correctly():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [10, 20, 30, 40, 50]})
    threshold = df["token_count"].quantile(0.8)
    short, long = dl.split_data(df, token_column="token_count", quantile=0.8)
    assert (short["token_count"] <= threshold).all()
    assert (long["token_count"] > threshold).all()


def test_split_data_quantile_1_all_short():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [5, 10, 15]})
    short, long = dl.split_data(df, token_column="token_count", quantile=1.0)
    assert len(short) == 3
    assert len(long) == 0


def test_split_data_quantile_0_mostly_long():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [10, 20, 30]})
    short, long = dl.split_data(df, token_column="token_count", quantile=0.0)
    # threshold = 10 (the 0th percentile), short = [10], long = [20, 30]
    assert len(short) == 1
    assert len(long) == 2


# ── DataLoader.get_max_input_tokens ──────────────────────────────


def test_get_max_input_tokens_no_examples():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [100, 200, 300]})
    result = dl.get_max_input_tokens(
        df, token_column="token_count", num_predict=512, buffer_tokens=1000, num_examples=0
    )
    # 300 * (0+1) + 1000 + 512 = 1812
    assert result == 1812


def test_get_max_input_tokens_with_examples():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [100, 200]})
    result = dl.get_max_input_tokens(
        df, token_column="token_count", num_predict=256, buffer_tokens=500, num_examples=3
    )
    # 200 * (3+1) + 500 + 256 = 1556
    assert result == 1556


# ── DataLoader.adapt_num_predict ──────────────────────────────────


def test_adapt_num_predict_no_flags():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [100, 200, 300]})
    result = dl.adapt_num_predict(
        df, token_column="token_count", buffer_tokens=1000,
        translate=False, reasoning_model=False, num_predict=512,
    )
    assert result == 512


def test_adapt_num_predict_translate():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [100, 200, 300]})
    result = dl.adapt_num_predict(
        df, token_column="token_count", buffer_tokens=1000,
        translate=True, reasoning_model=False, num_predict=512,
    )
    # num_predict = max(300) + 1000 = 1300
    assert result == 1300


def test_adapt_num_predict_reasoning():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [100, 200]})
    result = dl.adapt_num_predict(
        df, token_column="token_count", buffer_tokens=500,
        translate=False, reasoning_model=True, num_predict=256,
    )
    # num_predict = 256 + 500 = 756
    assert result == 756


def test_adapt_num_predict_both_flags():
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [100, 200]})
    result = dl.adapt_num_predict(
        df, token_column="token_count", buffer_tokens=1000,
        translate=True, reasoning_model=True, num_predict=256,
    )
    # translate first: 200 + 1000 = 1200; then reasoning: 1200 + 1000 = 2200
    assert result == 2200


# ── DataLoader.load_examples ─────────────────────────────────────


def test_load_examples_ok(tmp_path):
    data = [{"input": "hello", "output": "world"}, {"input": "foo", "output": "bar"}]
    f = tmp_path / "examples.json"
    f.write_text(json.dumps(data))
    dl = DataLoader(examples_path=str(f))
    examples = dl.load_examples()
    assert len(examples) == 2
    assert examples[0]["input"] == "hello"


def test_load_examples_missing_output_column(tmp_path):
    data = [{"input": "hello", "wrong_col": "world"}]
    f = tmp_path / "examples.json"
    f.write_text(json.dumps(data))
    dl = DataLoader(examples_path=str(f))
    with pytest.raises(ValueError, match="'input' and 'output' columns"):
        dl.load_examples()


def test_load_examples_no_path_raises():
    with pytest.raises(ValueError, match="No examples path"):
        DataLoader().load_examples()


# ── DataLoader.load_cases ────────────────────────────────────────


def test_load_cases_ok(tmp_path):
    data = [{"text": "case one"}, {"text": "case two"}]
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(data))
    dl = DataLoader(cases_path=str(f))
    df = dl.load_cases(text_column="text")
    assert "token_count" in df.columns
    assert len(df) == 2


def test_load_cases_missing_column(tmp_path):
    data = [{"body": "case one"}]
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(data))
    dl = DataLoader(cases_path=str(f))
    with pytest.raises(ValueError, match="'text' column not found"):
        dl.load_cases(text_column="text")


def test_load_cases_no_path_raises():
    with pytest.raises(ValueError, match="No cases path"):
        DataLoader().load_cases()


# ── TaskLoader ────────────────────────────────────────────────────


def test_task_loader_finds_and_loads(tmp_path):
    task_data = {"Description": "Test", "Data_Path": "data.csv"}
    (tmp_path / "Task001_test.json").write_text(json.dumps(task_data))
    loader = TaskLoader(str(tmp_path), task_id=1)
    result = loader.find_and_load_task()
    assert result["Description"] == "Test"


def test_task_loader_get_task_name(tmp_path):
    (tmp_path / "Task001_mytest.json").write_text('{"Description": "x"}')
    loader = TaskLoader(str(tmp_path), task_id=1)
    loader.find_and_load_task()
    assert loader.get_task_name() == "Task001_mytest"


def test_task_loader_folder_not_found():
    loader = TaskLoader("/nonexistent/path/abc", task_id=1)
    with pytest.raises(FileNotFoundError):
        loader.find_and_load_task()


def test_task_loader_no_matching_file(tmp_path):
    loader = TaskLoader(str(tmp_path), task_id=1)
    with pytest.raises(FileNotFoundError, match="No file found"):
        loader.find_and_load_task()


def test_task_loader_multiple_matches_raises(tmp_path):
    (tmp_path / "Task001_a.json").write_text("{}")
    (tmp_path / "Task001_b.json").write_text("{}")
    loader = TaskLoader(str(tmp_path), task_id=1)
    with pytest.raises(RuntimeError, match="Multiple files"):
        loader.find_and_load_task()


def test_task_loader_does_not_match_partial_id(tmp_path):
    # Task010 must not match task_id=1 (id 1 → pattern Task001)
    (tmp_path / "Task010.json").write_text("{}")
    loader = TaskLoader(str(tmp_path), task_id=1)
    with pytest.raises(FileNotFoundError):
        loader.find_and_load_task()


def test_task_loader_get_name_before_load_raises():
    loader = TaskLoader("/some/path", task_id=1)
    with pytest.raises(ValueError, match="No task file loaded"):
        loader.get_task_name()


# ── DataLoader.estimate_context_window ────────────────────────────


def _loader_with_examples(tmp_path, examples):
    import json as _json

    path = tmp_path / "examples.json"
    path.write_text(_json.dumps(examples), encoding="utf-8")
    dl = DataLoader(examples_path=path)
    dl.load_examples()
    return dl


def test_estimate_counts_scaffolding_longest_input_and_output_budget(tmp_path):
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [10, 500, 30]})

    estimate = dl.estimate_context_window(
        df, num_predict=100, scaffolding_tokens=200, safety_margin=0.0,
        min_margin_tokens=0,
    )

    assert estimate == 200 + 500 + 100  # scaffolding + longest input + generation


def test_estimate_applies_a_percentage_margin_to_the_prompt_only(tmp_path):
    """The margin covers tokenizer drift, which does not apply to what we generate."""
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [800]})

    estimate = dl.estimate_context_window(
        df, num_predict=100, scaffolding_tokens=200, safety_margin=0.10,
        min_margin_tokens=0,
    )

    assert estimate == 1000 + 100 + 100  # prompt + 10% of prompt + generation


def test_estimate_floors_the_margin_for_short_prompts(tmp_path):
    """A percentage of a tiny prompt would not cover the chat template's own tokens."""
    dl = DataLoader()
    df = pd.DataFrame({"token_count": [10]})

    estimate = dl.estimate_context_window(
        df, num_predict=0, scaffolding_tokens=0, safety_margin=0.15,
        min_margin_tokens=64,
    )

    assert estimate == 10 + 64


def test_example_tokens_uses_the_longest_examples_not_the_longest_document(tmp_path):
    """The old formula reserved `longest_document * (num_examples + 1)`.

    With five examples that is six times the longest document, spending exactly
    the VRAM the context ceiling exists to protect.
    """
    dl = _loader_with_examples(
        tmp_path,
        [
            {"input": "short", "output": "a"},
            {"input": "a much longer example input here", "output": "bb"},
            {"input": "mid length example", "output": "ccc"},
        ],
    )

    two = dl.example_tokens(2)
    three = dl.example_tokens(3)

    assert 0 < two < three  # picks the largest, and more examples cost more
    assert dl.example_tokens(0) == 0


def test_example_tokens_counts_the_answers_too(tmp_path):
    """Both halves of a few-shot exchange end up in the prompt."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    short_out = _loader_with_examples(a, [{"input": "same input", "output": "x"}])
    long_out = _loader_with_examples(
        b, [{"input": "same input", "output": "a considerably longer answer than x"}]
    )

    assert long_out.example_tokens(1) > short_out.example_tokens(1)


def test_estimate_grows_with_few_shot_examples(tmp_path):
    dl = _loader_with_examples(
        tmp_path, [{"input": "an example input", "output": "an example answer"}]
    )
    df = pd.DataFrame({"token_count": [100]})

    without = dl.estimate_context_window(df, num_examples=0, safety_margin=0.0, min_margin_tokens=0)
    with_one = dl.estimate_context_window(df, num_examples=1, safety_margin=0.0, min_margin_tokens=0)

    assert with_one > without
