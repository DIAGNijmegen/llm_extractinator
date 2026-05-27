import json

import pytest


def _chunk_files(output_path):
    """Mirror of the glob+filter logic in _run_single_prediction."""
    return [
        f for f in output_path.glob("nlp-predictions-dataset-*.json")
        if f.stem.split("-")[-1].isdigit()
    ]


def test_chunk_glob_excludes_split_files(tmp_path):
    """Numeric chunk files are collected; short/long split files are ignored."""
    numeric = ["nlp-predictions-dataset-0.json", "nlp-predictions-dataset-100.json"]
    split = ["nlp-predictions-dataset-short.json", "nlp-predictions-dataset-long.json"]

    for name in numeric + split:
        (tmp_path / name).write_text("[]")

    result = {f.name for f in _chunk_files(tmp_path)}

    assert result == set(numeric)
    assert not result.intersection(split)


def test_chunk_glob_empty_when_no_chunks(tmp_path):
    """Returns empty list when only split files exist."""
    (tmp_path / "nlp-predictions-dataset-short.json").write_text("[]")
    (tmp_path / "nlp-predictions-dataset-long.json").write_text("[]")

    assert _chunk_files(tmp_path) == []


def test_chunk_glob_all_numeric(tmp_path):
    """Returns all files when only numeric chunk files exist."""
    names = [f"nlp-predictions-dataset-{i}.json" for i in range(0, 300, 100)]
    for name in names:
        (tmp_path / name).write_text("[]")

    result = {f.name for f in _chunk_files(tmp_path)}
    assert result == set(names)


def test_chunk_glob_does_not_delete_split_files(tmp_path):
    """Deleting chunk files leaves split files on disk (regression for the original bug)."""
    chunk_names = ["nlp-predictions-dataset-0.json", "nlp-predictions-dataset-100.json"]
    split_names = ["nlp-predictions-dataset-short.json", "nlp-predictions-dataset-long.json"]

    for name in chunk_names + split_names:
        (tmp_path / name).write_text(json.dumps([{"status": "success"}]))

    chunks = _chunk_files(tmp_path)
    for f in chunks:
        f.unlink()

    remaining = {f.name for f in tmp_path.glob("*.json")}
    assert remaining == set(split_names)
