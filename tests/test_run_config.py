"""Tests for the Studio's run settings.

The Run tab used to build its ``extractinate`` command inline, inside a 1100-line
Streamlit module that cannot be imported without starting Streamlit — so the one
piece of the GUI with real logic in it had no coverage at all. It lives in
``run_config`` now, and this is that coverage.
"""

import pytest

from llm_extractinator.run_config import (
    HARDWARE_PRESETS,
    InvalidTaskName,
    RunSettings,
    THINKING_TEMPERATURE,
    resolved_temperature,
    suggested_num_predict,
    task_id_from_filename,
)


def _settings(**overrides) -> RunSettings:
    params = dict(task_file="Task007_reports.json", model_name="phi4")
    params.update(overrides)
    return RunSettings(**params)


# ── Task id parsing ───────────────────────────────────────────────


def test_task_id_is_read_from_the_filename():
    assert task_id_from_filename("Task007_reports.json") == "007"


def test_task_id_rejects_a_filename_without_one():
    """Better a clear error than `AttributeError: 'NoneType' has no group`."""
    with pytest.raises(InvalidTaskName):
        task_id_from_filename("my_task.json")


# ── Command construction ──────────────────────────────────────────


def test_default_run_is_a_minimal_command():
    """Defaults stay off the command line so what's shown stays readable."""
    assert _settings().to_command() == [
        "extractinate",
        "--task_id",
        "007",
        "--model_name",
        "phi4",
    ]


@pytest.mark.parametrize(
    "reasoning, expected",
    [(True, ["--reasoning_model"]), (False, ["--no_reasoning"]), (None, [])],
)
def test_reasoning_is_tri_state(reasoning, expected):
    """None must emit nothing, so the backend's auto-detection still applies."""
    cmd = _settings(reasoning=reasoning).to_command()
    assert [c for c in cmd if c.startswith("--reasoning") or c == "--no_reasoning"] == expected


def test_test_run_flags_travel_together():
    cmd = _settings(test_run_size=5, test_run_random=True).to_command()
    assert "--test_run_size" in cmd and cmd[cmd.index("--test_run_size") + 1] == "5"
    assert "--test_run_random" in cmd


def test_random_sampling_is_not_emitted_without_a_test_run():
    """The backend ignores it, but a command that implies otherwise misleads."""
    assert "--test_run_random" not in _settings(test_run_random=True).to_command()


def test_seed_zero_is_still_passed():
    """`if self.seed:` would drop 0 — the bug that was fixed in 0.6.0."""
    assert "--seed" in _settings(seed=0).to_command()


def test_no_seed_means_no_flag():
    assert "--seed" not in _settings().to_command()


def test_context_cap_is_emitted_when_set():
    cmd = _settings(max_context_cap=8192).to_command()
    assert cmd[cmd.index("--max_context_cap") + 1] == "8192"


def test_full_house_round_trips_through_the_cli_parser():
    """Whatever the Studio builds, `extractinate` must actually accept."""
    from llm_extractinator.main import parse_args

    settings = _settings(
        model_name="qwen3.5:35b",
        reasoning=False,
        n_runs=3,
        verbose=True,
        overwrite=True,
        seed=7,
        chunk_size=50,
        test_run_size=5,
        test_run_random=True,
        num_examples=2,
        max_context_len="split",
        max_context_cap=32768,
        temperature=0.4,
        num_predict=2048,
        top_k=40,
        top_p=0.9,
        ollama_host="http://localhost:11500",
    )
    cmd = settings.to_command()

    import sys

    original, sys.argv = sys.argv, cmd
    try:
        config = parse_args()
    finally:
        sys.argv = original

    assert config.task_id == 7
    assert config.model_name == "qwen3.5:35b"
    assert config.no_reasoning is True and config.reasoning_model is False
    assert config.n_runs == 3 and config.seed == 7
    assert config.chunk_size == 50
    assert config.test_run_size == 5 and config.test_run_random is True
    assert config.max_context_len == "split" and config.max_context_cap == 32768
    assert config.num_predict == 2048 and config.num_examples == 2
    assert config.ollama_host == "http://localhost:11500"


# ── Review strip ──────────────────────────────────────────────────


def test_summary_describes_a_plain_run():
    labels = dict((label, value) for label, value, _ in _settings().summary())
    assert labels["Scope"] == "Full dataset"
    assert labels["Context"] == "Fitted to data"
    assert labels["Reasoning"] == "Auto-detect"


def test_summary_spells_out_a_chunked_test_run():
    settings = _settings(test_run_size=5, test_run_random=True, chunk_size=50)
    assert settings.scope_label == "Test run · random 5 rows · chunks of 50"


def test_summary_highlights_only_what_deviates():
    """The strip earns its place by making the surprising settings stand out."""
    plain = {label: hot for label, _, hot in _settings().summary()}
    assert plain["Scope"] is False and plain["Context"] is False

    odd = {label: hot for label, _, hot in _settings(test_run_size=5).summary()}
    assert odd["Scope"] is True


def test_context_label_mentions_the_cap():
    assert _settings(max_context_cap=4096).context_label == "Fitted to data · max 4096"


# ── Presets ───────────────────────────────────────────────────────


def test_presets_carry_neither_reasoning_nor_output_length():
    """Both follow from something other than the size of the card.

    Reasoning is auto-detected after the model is pulled; output length follows
    the schema. A preset that pinned either would only be able to get it wrong.
    """
    for preset in HARDWARE_PRESETS.values():
        assert not hasattr(preset, "reasoning")
        assert not hasattr(preset, "num_predict")


def test_every_preset_caps_the_context():
    """A preset that skipped this would not actually bound anything."""
    for name, preset in HARDWARE_PRESETS.items():
        assert preset.context_cap > 0, name
        assert preset.min_vram_gb > 0, name
        assert preset.note, f"{name} needs a note saying what it is for"


def test_low_tier_allows_more_context_than_medium():
    """Counter-intuitive but correct, so worth pinning down.

    qwen3:4b needs 2.33GiB of weights on an 8GB card; phi4:14b needs 8.47GiB on
    a 12GB one. The small model leaves far more room for a KV cache, so the
    cheaper tier supports the longer window.
    """
    presets = HARDWARE_PRESETS
    assert presets["Low — 8GB VRAM"].context_cap > presets["Medium — 12GB VRAM"].context_cap


def test_high_tier_needs_more_than_a_24gb_card():
    """22.4GiB of weights does not fit 24GB with any usable context."""
    assert HARDWARE_PRESETS["High — 40GB+ VRAM"].min_vram_gb >= 40


def test_suggested_output_budget_makes_room_for_thinking():
    """A thinking model spends the same budget on chain-of-thought first."""
    assert suggested_num_predict(False) == 512
    assert suggested_num_predict(True) >= 2048


# ── Temperature ───────────────────────────────────────────────────


def test_temperature_defaults_to_auto():
    """Auto must emit no flag, so the backend can decide once it knows the model."""
    assert _settings().temperature is None
    assert "--temperature" not in _settings().to_command()


def test_explicit_zero_temperature_is_still_emitted():
    """`if self.temperature:` would drop it — the same falsy-zero trap as seed."""
    cmd = _settings(temperature=0.0).to_command()
    assert cmd[cmd.index("--temperature") + 1] == "0.0"


def test_auto_temperature_avoids_greedy_for_thinking_models():
    """Qwen: "DO NOT use greedy decoding... endless repetitions." """
    assert resolved_temperature(None, reasoning=True) == THINKING_TEMPERATURE
    assert THINKING_TEMPERATURE > 0


def test_auto_temperature_is_greedy_otherwise():
    """Extraction wants determinism when nothing argues against it."""
    assert resolved_temperature(None, reasoning=False) == 0.0
    assert resolved_temperature(None, reasoning=None) == 0.0


def test_an_explicit_temperature_always_wins():
    """Including an explicit 0 on a thinking model — the user gets to be wrong."""
    assert resolved_temperature(0.0, reasoning=True) == 0.0
    assert resolved_temperature(0.9, reasoning=False) == 0.9


def test_summary_reports_which_temperature_will_be_used():
    assert "Auto" in _settings().temperature_label
    assert _settings(temperature=0.42).temperature_label == "0.42"
