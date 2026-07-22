"""Real-model smoke test (opt-in).

Marked ``integration`` so it is skipped in CI (``pytest -m "not integration"``).
Run it locally with a real Ollama server to confirm an actual model flows
through the pipeline end to end::

    pytest -m integration

It intentionally asserts only *shape and validity*, never exact answers — the
model's accuracy is not what this suite is testing, and pinning exact output is
what made the old suite flaky. Override the model (default is small) with::

    EXTRACTINATOR_SMOKE_MODEL=qwen2.5:0.5b pytest -m integration
"""

import os

import pytest

from llm_extractinator.main import extractinate
from tests.conftest import DATA_DIR, TASK_DIR, load_predictions

pytestmark = pytest.mark.integration

SMOKE_MODEL = os.environ.get("EXTRACTINATOR_SMOKE_MODEL", "qwen2.5:0.5b")
ALLOWED_NAMES = {"Alice", "Bob", "Charlie", "David", "Emma"}


def test_real_model_runs_end_to_end(tmp_path):
    extractinate(
        model_name=SMOKE_MODEL,
        task_id=999,
        num_examples=0,
        n_runs=1,
        temperature=0.0,
        max_context_len="max",
        run_name="smoke",
        num_predict=256,
        output_dir=tmp_path,
        task_dir=TASK_DIR,
        data_dir=DATA_DIR,
        overwrite=True,
        seed=42,
    )

    preds = load_predictions(tmp_path, "smoke", "Task999_example")
    assert len(preds) == 5

    for p in preds:
        assert p["status"] in {"success", "failure"}
        if p["status"] == "success":
            # Validity only: types and allowed values, not the "correct" answer.
            assert isinstance(p["HR"], int)
            assert p["Name"] in ALLOWED_NAMES
