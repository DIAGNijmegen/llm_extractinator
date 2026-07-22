"""Unit tests for command-line argument parsing.

These replace the old subprocess+real-model CLI test. They run instantly and
pin down the flag semantics that are easy to get wrong — notably that
``--translate`` / ``--verbose`` are boolean *switches*: their presence means
True and their absence means False. Passing ``--translate False`` does NOT
disable translation (it enables it and leaves a stray positional), which is a
mistake worth guarding against.
"""

import pytest

from llm_extractinator.main import parse_args


def _parse(monkeypatch, tmp_path, argv):
    # Point paths at tmp so logging setup doesn't touch the real cwd.
    full = argv + ["--output_dir", str(tmp_path)]
    monkeypatch.setattr("sys.argv", ["extractinate", *full])
    return parse_args()


def test_defaults(monkeypatch, tmp_path):
    cfg = _parse(monkeypatch, tmp_path, ["--task_id", "1"])
    assert cfg.task_id == 1
    assert cfg.translate is False
    assert cfg.verbose is False
    assert cfg.max_context_len == "max"


def test_store_true_flags_are_switches(monkeypatch, tmp_path):
    cfg = _parse(monkeypatch, tmp_path, ["--task_id", "1", "--translate", "--verbose"])
    assert cfg.translate is True
    assert cfg.verbose is True


@pytest.mark.parametrize(
    "value,expected",
    [("max", "max"), ("split", "split"), ("2048", 2048)],
)
def test_max_context_len_coercion(monkeypatch, tmp_path, value, expected):
    cfg = _parse(monkeypatch, tmp_path, ["--task_id", "1", "--max_context_len", value])
    assert cfg.max_context_len == expected


def test_invalid_max_context_len_exits(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, tmp_path, ["--task_id", "1", "--max_context_len", "banana"])
