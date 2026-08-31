"""Two things that fail quietly: a dropped parameter and a leaked process.

Neither shows up as an error. A parameter missing from ``REQUIRED_PARAMS`` is
discarded in silence, and a server left running after a run only becomes
visible as a port that is already in use.
"""

import subprocess
from dataclasses import fields

import pytest

from llm_extractinator.main import TaskConfig
from llm_extractinator.ollama_server import OllamaServerManager
from llm_extractinator.prediction_task import PredictionTask

# Settings the runner acts on itself and that ``PredictionTask`` has no use for.
# Listing them explicitly is the point: adding a field to ``TaskConfig`` should
# force a decision about whether the task needs it, rather than defaulting to
# "no" by omission.
RUNNER_ONLY_SETTINGS = {
    "log_dir",  # used to set up logging before any task exists
    "quantile",  # only ever used to split the dataset, in TaskRunner
    "test_run_random",  # decides which rows the runner selects, before the task
}

# Values that are not configuration at all: they are resolved at run time, after
# the model has been inspected and the budget resolved.
RESOLVED_AT_RUN_TIME = {"model_info", "budget"}


def test_required_params_accounts_for_every_task_config_field():
    """``REQUIRED_PARAMS`` is an allowlist, and omission is silent.

    ``PredictionTask`` is constructed from a dict of every ``TaskConfig`` field
    plus a few resolved extras, and copies across only the names it lists. A
    field that is not listed does not raise — it simply never arrives. That is
    how ``max_context_cap`` failed to reach the task for a whole release, which
    in turn is why the obvious fix for the reasoning-allowance bug would have
    blown straight through the user's VRAM ceiling.
    """
    config_fields = {f.name for f in fields(TaskConfig)}
    dropped = config_fields - PredictionTask.REQUIRED_PARAMS

    assert dropped == RUNNER_ONLY_SETTINGS, (
        "a TaskConfig field is being silently dropped on the way to "
        f"PredictionTask: {sorted(dropped - RUNNER_ONLY_SETTINGS)}. Either add "
        f"it to REQUIRED_PARAMS or add it to RUNNER_ONLY_SETTINGS with a reason."
    )


def test_required_params_has_nothing_that_comes_from_nowhere():
    """The other direction: a name listed but never supplied arrives as None,
    which is just as silent."""
    config_fields = {f.name for f in fields(TaskConfig)}
    extras = PredictionTask.REQUIRED_PARAMS - config_fields

    assert extras == RESOLVED_AT_RUN_TIME, (
        f"REQUIRED_PARAMS expects {sorted(extras - RESOLVED_AT_RUN_TIME)}, which "
        f"is neither a TaskConfig field nor documented as resolved at run time"
    )


# ── the server we start is the server we stop ─────────────────────


class _FakeProcess:
    """Stands in for the Popen handle of a server we launched."""

    def __init__(self, exits: bool = True):
        self.terminated = False
        self.killed = False
        self._exits = exits

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self._exits or self.killed:
            return 0
        raise subprocess.TimeoutExpired(cmd="ollama serve", timeout=timeout)


def _manager(tmp_path, **kwargs):
    return OllamaServerManager(tmp_path, **kwargs)


def test_a_server_we_started_is_terminated(tmp_path, monkeypatch):
    """``ollama stop <model>`` unloads the model but leaves the server up.

    We launched it, so nothing else will ever shut it down — every run leaked a
    process, and on a shared machine one still holding the port.
    """
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    manager = _manager(tmp_path)
    process = _FakeProcess()
    manager.process = process

    manager.stop("some-model")

    assert process.terminated
    assert manager.process is None


def test_a_server_that_will_not_exit_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    manager = _manager(tmp_path)
    process = _FakeProcess(exits=False)
    manager.process = process

    manager.stop("some-model")

    assert process.terminated and process.killed
    assert manager.process is None


def test_a_server_that_was_already_running_is_left_alone(tmp_path, monkeypatch):
    """Somebody else started it; it is not ours to stop."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    manager = _manager(tmp_path)
    manager._external = True  # detected as already running by start_server
    assert manager.process is None

    manager.stop("some-model")  # must not raise


def test_an_externally_managed_server_is_never_touched(tmp_path, monkeypatch):
    """``--ollama_host`` means connect only: never start, pull or stop."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    manager = _manager(tmp_path, host="http://elsewhere:11434")
    process = _FakeProcess()
    manager.process = process

    manager.stop("some-model")

    assert calls == []
    assert not process.terminated
