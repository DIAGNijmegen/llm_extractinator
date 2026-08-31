"""The Studio's subprocess plumbing, tested without Streamlit.

``RunProcess`` is what makes a Stop button possible: the old Run tab iterated
``process.stdout`` on the script thread, which parked the whole app for the
duration of the run and left no thread free to notice a click. These tests drive
it with ``python -c`` rather than a real ``extractinate``, so they cover the
parsing and the lifecycle in milliseconds and without a model.
"""

import sys
import time

import pytest

from llm_extractinator.gui import RunProcess


def _emit(script: str) -> list[str]:
    return [sys.executable, "-u", "-c", script]


def _settle(run: RunProcess, timeout: float = 10.0) -> None:
    """Wait for the process *and* its drain thread to finish."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not run.running and run.snapshot()["finished"] is not None:
            return
        time.sleep(0.02)
    pytest.fail("RunProcess did not finish in time")


def test_transcript_survives_and_ansi_is_stripped():
    run = RunProcess(_emit(r"print('\x1b[32mhello\x1b[0m'); print('world')"))
    _settle(run)

    assert run.snapshot()["log"].splitlines() == ["hello", "world"]
    assert run.return_code == 0


def test_progress_is_read_in_rows_and_kept_out_of_the_transcript():
    """tqdm redraws over itself; only the counts matter, and only once."""
    run = RunProcess(
        _emit(r"print('  40%|####  | 4/10 [00:12<00:18, 1.2it/s]'); print('done')")
    )
    _settle(run)

    state = run.snapshot()
    assert (state["done_rows"], state["total_rows"]) == (4, 10)
    # A progress frame is status, not history — repeating it 10,000 times must
    # not push the useful lines out of a bounded transcript.
    assert state["log"].splitlines() == ["done"]


def test_carriage_returns_do_not_smuggle_stale_frames_through():
    run = RunProcess(_emit(r"print('1/9 [a]\r5/9 [a]\r9/9 [a]')"))
    _settle(run)

    assert run.snapshot()["done_rows"] == 9


def test_the_budget_line_is_promoted_out_of_the_log():
    """It is the one line 0.7.0 exists to produce; it must not scroll past."""
    line = "context 4096 = 1119 prompt + 512 answer + 2465 headroom (fitted to the data)"
    run = RunProcess(_emit(f"print('INFO {line}')"))
    _settle(run)

    state = run.snapshot()
    assert state["budget_line"] == line
    # Still in the transcript too — promoting it must not remove it from context.
    assert line in state["log"]


def test_pull_noise_goes_to_the_status_line_only():
    run = RunProcess(_emit(r"print('pulling manifest'); print('real message')"))
    _settle(run)

    state = run.snapshot()
    assert state["status_line"] == "pulling manifest"
    assert state["log"].splitlines() == ["real message"]


def test_stop_terminates_a_long_run():
    run = RunProcess(_emit("import time; time.sleep(120)"))
    assert run.running

    run.stop()

    assert not run.running
    assert run.stopped_by_user
    assert run.return_code != 0


def test_the_script_thread_is_never_blocked_by_a_slow_run():
    """The whole point: reading output must not cost the caller its thread.

    A run that produces nothing for a while used to freeze the entire Studio.
    Construction and every read here have to return promptly regardless.
    """
    run = RunProcess(_emit("import time; time.sleep(30)"))
    try:
        started = time.time()
        for _ in range(5):
            run.snapshot()
            assert run.running
        assert time.time() - started < 1.0
    finally:
        run.stop()
