"""Shared test fixtures.

The key idea here is a *hermetic* end-to-end fixture: it runs the full
``extractinate`` pipeline (prompt building, schema binding, output parsing,
result merging, file writing) with the LLM and the Ollama server faked out.
That means the "does it actually run end-to-end" tests need no GPU, no model
downloads, and no running Ollama, so they execute in ~1s and run in CI.

Tests that need a *real* model live in ``test_pipeline_smoke.py`` behind the
``integration`` marker and assert only output validity, never exact answers.
"""

import json
from pathlib import Path
from typing import List, Optional

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

TESTS_DIR = Path(__file__).resolve().parent
TASK_DIR = TESTS_DIR / "testtasks"
DATA_DIR = TESTS_DIR / "testdata"
EXAMPLE_DIR = TESTS_DIR / "testexamples"


class FakeChatModel(BaseChatModel):
    """A deterministic stand-in for ``ChatOllama``.

    * Extraction path: ``_generate`` returns a canned JSON string, which flows
      through the same ``bind(format=...) | strip_think | PydanticOutputParser``
      chain the real code uses. Set ``responses`` to a non-JSON string to
      exercise the graceful-failure path.
    * Translation path: ``with_structured_output`` returns a runnable that emits
      a canned ``translation`` object (the only structured-output caller).

    A single canned response is returned for every row so results stay
    deterministic under batch concurrency.
    """

    responses: List[str] = ["{}"]
    translation: str = "translated text"

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(content=self.responses[0])
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        text = self.translation
        return RunnableLambda(lambda _input, _schema=schema: _schema(translation=text))


class _NoopOllamaManager:
    """Replaces OllamaServerManager so no server is started or model pulled."""

    def __init__(self, *args, **kwargs):
        pass

    def start_server(self):
        pass

    def pull_model(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


def load_predictions(
    output_dir: Path, run_name: str, task_name: str, run_idx: int = 0
) -> List[dict]:
    """Read the predictions file produced by a run."""
    path = (
        output_dir
        / run_name
        / f"{task_name}-run{run_idx}"
        / "nlp-predictions-dataset.json"
    )
    assert path.exists(), f"Expected output file was not created: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def offline_run(monkeypatch, tmp_path):
    """Run ``extractinate`` fully offline with a faked model.

    Returns a callable. Override any ``extractinate`` kwarg; ``responses`` and
    ``translation`` configure the fake. Yields the output directory (tmp_path).
    """
    from llm_extractinator import main, prediction_task, predictor

    def _run(
        responses: Optional[List[str]] = None,
        translation: str = "translated text",
        **kwargs,
    ) -> Path:
        fake = FakeChatModel(responses=responses or ["{}"], translation=translation)

        # Inject the fake model and neutralise every real I/O boundary.
        monkeypatch.setattr(
            prediction_task.PredictionTask, "initialize_model", lambda self: fake
        )
        monkeypatch.setattr(main, "OllamaServerManager", _NoopOllamaManager)
        monkeypatch.setattr(
            prediction_task, "model_supports_thinking", lambda name, **kwargs: False
        )
        # Few-shot path: skip the embedding-model pull, use offline embeddings.
        monkeypatch.setattr(predictor.ollama, "pull", lambda *a, **k: None)
        monkeypatch.setattr(
            predictor,
            "OllamaEmbeddings",
            lambda *a, **k: DeterministicFakeEmbedding(size=64),
        )

        params = dict(
            model_name="fake-model",
            num_examples=0,
            n_runs=1,
            temperature=0.0,
            max_context_len=512,
            num_predict=64,
            output_dir=tmp_path,
            task_dir=TASK_DIR,
            data_dir=DATA_DIR,
            example_dir=EXAMPLE_DIR,
            translation_dir=tmp_path / "translations",
            overwrite=True,
            verbose=False,
            seed=42,
        )
        params.update(kwargs)
        main.extractinate(**params)
        return tmp_path

    return _run
