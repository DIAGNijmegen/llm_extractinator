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
from pydantic import Field as PydanticField

#: Every prompt-calibration probe the fixtures intercept, so a test can assert
#: on *what* was measured as well as on what the measurement did. Clear it first.
PROBE_CALLS: List[dict] = []

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
    """Replaces OllamaServerManager so no server is started or model pulled.

    Records what it was asked to do on the class, so a test can assert on the
    server *lifecycle* — how many times a run starts a server or pulls a model —
    which is otherwise invisible from the outside. Clear ``calls`` first.
    """

    calls: List[str] = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append("construct")

    def start_server(self):
        type(self).calls.append("start")

    def pull_model(self, *args, **kwargs):
        type(self).calls.append("pull")

    def stop(self, *args, **kwargs):
        type(self).calls.append("stop")


def load_predictions(
    output_dir: Path,
    run_name: str,
    task_name: str,
    run_idx: int = 0,
    suffix: str = "",
) -> List[dict]:
    """Read the predictions file produced by a run.

    ``suffix`` matches the ``-test<N>`` tag ``PredictionTask`` adds to the
    per-task-run folder name when ``test_run_size`` is set.
    """
    path = (
        output_dir
        / run_name
        / f"{task_name}{suffix}-run{run_idx}"
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
    from llm_extractinator.ollama_server import ModelInfo

    def _run(
        responses: Optional[List[str]] = None,
        translation: str = "translated text",
        model: Optional[BaseChatModel] = None,
        measured_prompt_tokens: Optional[int] = None,
        **kwargs,
    ) -> Path:
        # ``model`` lets a test supply its own stand-in — a RecordingFakeChatModel
        # to count calls, or one that raises — while keeping every other boundary
        # faked exactly as it is here.
        fake = model or FakeChatModel(
            responses=responses or ["{}"], translation=translation
        )

        # Inject the fake model and neutralise every real I/O boundary.
        monkeypatch.setattr(
            prediction_task.PredictionTask, "initialize_model", lambda self: fake
        )
        # The prompt-calibration probe is a live request. Stubbed rather than
        # left to fail, so these tests do not depend on nothing listening — and
        # so a test can drive the calibration path by choosing what the model
        # "counts". None means "could not be measured", the offline default.
        def _measure(model, messages, host=None, num_ctx=None):
            PROBE_CALLS.append(
                {"model": model, "messages": messages, "num_ctx": num_ctx}
            )
            return measured_prompt_tokens

        monkeypatch.setattr(prediction_task, "measure_prompt_tokens", _measure)
        monkeypatch.setattr(main, "OllamaServerManager", _NoopOllamaManager)
        # Capability detection is a live /api/show call; without a server it
        # would fall back to empty defaults anyway, but saying so explicitly
        # keeps these tests independent of whether anything is listening.
        monkeypatch.setattr(main, "model_capabilities", lambda name, **kw: ModelInfo())
        monkeypatch.setattr(
            prediction_task, "model_capabilities", lambda name, **kw: ModelInfo()
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


@pytest.fixture
def record_task(monkeypatch, tmp_path):
    """Run ``extractinate`` but capture the config instead of predicting.

    ``PredictionTask`` is where the pipeline's decisions land, so recording the
    kwargs it is constructed with is how we assert on things the output files
    never show — the context window in particular.
    """
    from llm_extractinator import main

    def _run(responses=None, **kwargs) -> dict:
        captured: dict = {}

        class _Recorder:
            def __init__(self, **kw):
                captured.update(kw)

            def run(self):
                return []

        monkeypatch.setattr(main, "PredictionTask", _Recorder)
        monkeypatch.setattr(main, "OllamaServerManager", _NoopOllamaManager)

        params = dict(
            model_name="fake-model",
            num_examples=0,
            n_runs=1,
            num_predict=64,
            output_dir=tmp_path,
            task_dir=TASK_DIR,
            data_dir=DATA_DIR,
            example_dir=EXAMPLE_DIR,
            translation_dir=tmp_path / "translations",
            overwrite=True,
            seed=42,
        )
        params.update(kwargs)
        main.extractinate(**params)
        assert captured, "PredictionTask was never constructed"
        return captured

    return _run


class RecordingFakeChatModel(FakeChatModel):
    """A :class:`FakeChatModel` that also remembers every prompt it was given.

    The context-budget invariant is about the prompt the model is *actually*
    asked to complete, not about the estimator's opinion of it, so the fake has
    to keep what it saw.
    """

    seen: List[str] = PydanticField(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append("\n".join(str(m.content) for m in messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def record_model_kwargs(monkeypatch, tmp_path):
    """Run ``extractinate`` and capture what actually reached ``ChatOllama``.

    This is deliberately a *lower* observation point than :func:`record_task`.
    ``record_task`` captures ``PredictionTask``'s constructor kwargs — the values
    as they were *before* ``__init__`` ran — and ``__init__`` is precisely where
    ``num_predict`` gets mutated. A budget assertion made there cannot see the
    number the model is really given, which is how a generation budget eight
    times the size of the context window survived a suite that already covered
    context sizing.

    ``initialize_model`` is therefore left alone and ``ChatOllama`` itself is
    replaced, so every mutation the pipeline performs has already happened by the
    time a record is taken.

    Returns one record per ``ChatOllama`` construction — ``max_context_len="split"``
    builds two, one for short cases and one for long — each carrying the final
    ``num_ctx`` and ``num_predict`` plus ``prompt_tokens``, the largest prompt the
    model was asked to complete in that phase.
    """
    from llm_extractinator import main, prediction_task, predictor
    from llm_extractinator.data_loader import DataLoader
    from llm_extractinator.ollama_server import ModelInfo

    counter = DataLoader()

    def _run(
        responses: Optional[List[str]] = None,
        thinking: bool = False,
        native_context: Optional[int] = None,
        measured_prompt_tokens: Optional[int] = None,
        **kwargs,
    ):
        records: List[dict] = []
        info = ModelInfo(supports_thinking=thinking, native_context=native_context)

        def _fake_chat_ollama(**kw):
            model = RecordingFakeChatModel(responses=responses or ["{}"])
            records.append({"kwargs": kw, "model": model})
            return model

        # The prompt-calibration probe is a live request. Stubbed rather than
        # left to fail, so these tests do not depend on nothing listening — and
        # so a test can drive the calibration path by choosing what the model
        # "counts". None means "could not be measured", the offline default.
        def _measure(model, messages, host=None, num_ctx=None):
            PROBE_CALLS.append(
                {"model": model, "messages": messages, "num_ctx": num_ctx}
            )
            return measured_prompt_tokens

        monkeypatch.setattr(prediction_task, "measure_prompt_tokens", _measure)
        monkeypatch.setattr(prediction_task, "ChatOllama", _fake_chat_ollama)
        # TaskRunner inspects the model once, after the pull; this is that call.
        monkeypatch.setattr(main, "model_capabilities", lambda name, **kw: info)
        monkeypatch.setattr(prediction_task, "model_capabilities", lambda name, **kw: info)
        monkeypatch.setattr(main, "OllamaServerManager", _NoopOllamaManager)
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
            num_predict=512,
            output_dir=tmp_path,
            task_dir=TASK_DIR,
            data_dir=DATA_DIR,
            example_dir=EXAMPLE_DIR,
            translation_dir=tmp_path / "translations",
            overwrite=True,
            seed=42,
        )
        params.update(kwargs)
        main.extractinate(**params)

        assert records, (
            "ChatOllama was never constructed. extractinate() swallows "
            "exceptions, so this usually means the run raised — check the log."
        )
        for record in records:
            record["num_ctx"] = record["kwargs"].get("num_ctx")
            record["num_predict"] = record["kwargs"].get("num_predict")
            record["prompt_tokens"] = max(
                (counter.count_tokens(seen) for seen in record["model"].seen),
                default=0,
            )
        return records

    return _run
