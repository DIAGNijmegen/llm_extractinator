"""Smoke tests for the Studio's Run tab, driven through Streamlit's AppTest.

``gui.py`` runs Streamlit at import time, so it cannot be imported the ordinary
way — AppTest is what makes it testable at all. These do not check appearance;
they check that the tab renders, that the controls are wired to each other, and
that it stays free of the session-state warning that dogged it.
"""

import logging
from pathlib import Path

import pytest

import llm_extractinator

from tests.conftest import DATA_DIR, TASK_DIR

# AppTest resolves a relative path against the *calling* file, so give it an
# absolute one taken from the installed package.
GUI = str(Path(llm_extractinator.__file__).parent / "gui.py")
SESSION_STATE_WARNING = "default value but also had its value set"


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """An AppTest for the Studio, pointed at a throwaway working directory."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    for name in ("tasks", "data", "examples", "output"):
        (tmp_path / name).mkdir()
    (tmp_path / "tasks" / "Task999_example.json").write_bytes(
        (TASK_DIR / "Task999_example.json").read_bytes()
    )
    (tmp_path / "data" / "testdata.json").write_bytes(
        (DATA_DIR / "testdata.json").read_bytes()
    )
    monkeypatch.setenv("EXTRACTINATOR_BASE_DIR", str(tmp_path))

    def _open(**session):
        at = AppTest.from_file(GUI, default_timeout=60)
        # Skip the Task tab; these tests are about the Run tab.
        at.session_state["task_ready"] = True
        at.session_state["task_choice"] = "Task999_example.json"
        # Non-widget state a returning session would still be carrying.
        for key, value in session.items():
            at.session_state[key] = value
        return at.run()

    return _open


def _button(at, label):
    """AppTest indexes buttons positionally; labels survive a layout change."""
    return next(b for b in at.button if b.label == label)


def test_run_tab_renders(studio):
    at = studio()
    assert not at.exception
    # The launch bar: what a routine run actually chooses, above the fold.
    assert at.selectbox(key="task_choice") is not None
    assert at.radio(key="scope_choice") is not None
    assert any(b.label == "Run" for b in at.button)


def test_applying_a_preset_fills_every_field_it_promises(studio):
    """The preset writes model and context ceiling, and leaves output length alone.

    Applying now defers the write to the top of the next run rather than relying
    on the button sitting above every widget it touches — so this also covers
    that the deferral actually lands.
    """
    at = studio()
    at.selectbox(key="hw_preset_choice").set_value("High — 40GB+ VRAM").run()
    _button(at, "Apply").click().run()

    assert not at.exception
    assert at.text_input(key="model_name_free_text").value == "qwen3.5:35b"
    assert at.checkbox(key="context_cap_enabled").value is True
    assert at.number_input(key="context_cap_input").value == 262_144
    # Output length is not a hardware decision, so a preset must leave it alone —
    # and leaving it alone now means leaving it on auto.
    assert at.checkbox(key="num_predict_manual").value is False


def test_choosing_a_test_run_reveals_its_options_and_relabels_the_button(studio):
    at = studio()
    assert any(b.label == "Run" for b in at.button)

    at.radio(key="scope_choice").set_value("Test run").run()

    assert not at.exception
    assert any("test (5 rows)" in b.label for b in at.button)
    assert at.number_input(key="test_run_rows") is not None


def test_the_command_reflects_the_chosen_settings(studio):
    at = studio()
    at.selectbox(key="hw_preset_choice").set_value("Low — 8GB VRAM").run()
    _button(at, "Apply").click().run()
    at.radio(key="scope_choice").set_value("Test run").run()

    command = next(c.value for c in at.code if "extractinate" in c.value)

    assert "--model_name qwen3:4b" in command
    assert "--test_run_size 5" in command
    assert "--max_context_cap 16384" in command


def test_output_budget_is_auto_until_it_is_asked_for(studio):
    """The regression that made this whole field untrustworthy.

    ``num_predict`` used to be an int whose default, 512, doubled as the "do not
    emit the flag" sentinel: the widget showed 512, the command omitted it, and
    the run derived something else from the schema. Auto must now be visibly
    auto, and a manual value must actually reach the command.
    """
    at = studio()
    command = next(c.value for c in at.code if "extractinate" in c.value)
    assert "--num_predict" not in command

    at.checkbox(key="num_predict_manual").set_value(True).run()
    at.number_input(key="num_predict_input").set_value(1024).run()

    command = next(c.value for c in at.code if "extractinate" in c.value)
    assert "--num_predict 1024" in command


def test_run_is_disabled_when_the_window_cannot_hold_the_answer(studio):
    """An error beside a live button invites the click it just argued against."""
    at = studio()
    at.checkbox(key="num_predict_manual").set_value(True).run()
    at.number_input(key="num_predict_input").set_value(4096).run()
    at.checkbox(key="context_cap_enabled").set_value(True).run()
    at.number_input(key="context_cap_input").set_value(2048).run()

    assert not at.exception
    assert _button(at, "Run").disabled is True
    assert any("no room for documents" in e.value for e in at.error)


def test_no_session_state_warnings(studio, caplog):
    """Regression: every widget must seed through session state, not a default.

    Passing `value=`/`index=` alongside a key Streamlit already holds makes it
    log a warning and silently ignore the default.
    """
    with caplog.at_level(logging.WARNING):
        at = studio()
        at.selectbox(key="hw_preset_choice").set_value("Low — 8GB VRAM").run()
        _button(at, "Apply").click().run()
        at.radio(key="scope_choice").set_value("Test run").run()

    offenders = [r.getMessage() for r in caplog.records if SESSION_STATE_WARNING in r.getMessage()]
    assert offenders == []


def test_settings_survive_a_run(studio):
    """The reported bug: coming back from a run reset every setting.

    Streamlit garbage-collects session-state entries whose widget did not render
    during a script run, and the run view renders none of the form. What is left
    afterwards is a session holding the non-widget keys only — which is what the
    second half of this test reconstructs.
    """
    at = studio()
    at.number_input(key="num_examples_input").set_value(3).run()
    at.checkbox(key="context_cap_enabled").set_value(True).run()

    remembered = at.session_state["remembered"]
    assert remembered["num_examples_input"] == 3
    assert remembered["context_cap_enabled"] is True

    returning = studio(remembered=remembered)

    assert not returning.exception
    assert returning.number_input(key="num_examples_input").value == 3
    assert returning.checkbox(key="context_cap_enabled").value is True


def test_a_fresh_session_still_gets_the_defaults(studio):
    """Remembering must not break the first-run experience."""
    at = studio()
    assert at.number_input(key="num_examples_input").value == 0
    assert at.checkbox(key="num_predict_manual").value is False


def test_existing_output_is_flagged_before_the_click(studio, tmp_path):
    """Skip-not-clobber is safe but silent: without this you read a stale file.

    The backend keeps the previous predictions and reports a completed run, so
    the Results tab would show the old output as though it were new.
    """
    at = studio()
    assert not any("already has results" in w.value for w in at.warning)

    run_dir = tmp_path / "output" / "run" / "Task999_example-run0"
    run_dir.mkdir(parents=True)
    (run_dir / "nlp-predictions-dataset.json").write_text("[]", encoding="utf-8")
    at = at.run()

    assert any("already has results" in w.value for w in at.warning)
    assert at.checkbox(key="overwrite").value is False

    _button(at, "Overwrite").click().run()

    assert at.checkbox(key="overwrite").value is True
    assert not any("already has results" in w.value for w in at.warning)


def test_a_test_run_does_not_collide_with_a_full_run(studio, tmp_path):
    """The row count is part of the folder name, so these are different runs."""
    run_dir = tmp_path / "output" / "run" / "Task999_example-run0"
    run_dir.mkdir(parents=True)
    (run_dir / "nlp-predictions-dataset.json").write_text("[]", encoding="utf-8")

    at = studio()
    assert any("already has results" in w.value for w in at.warning)

    at.radio(key="scope_choice").set_value("Test run").run()

    assert not any("already has results" in w.value for w in at.warning)
