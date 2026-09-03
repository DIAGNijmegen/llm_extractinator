import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import ollama
import pandas as pd

try:
    from langchain.output_parsers import PydanticOutputParser
except Exception:
    from langchain_core.output_parsers import PydanticOutputParser

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Errors worth trying again — and only these. A retry re-sends a byte-identical
# request, so it can only help when something *other than the request* changed
# between attempts — the transport, or the server's own availability:
#
#   ConnectionError          ollama's own translation of httpx.ConnectError (see
#                            ollama/_client.py) — the server is starting,
#                            restarting or briefly unreachable
#   httpx.TransportError     read/write timeouts and protocol errors, which the
#                            ollama client passes through untouched
#   RetryableResponseError   a ResponseError carrying HTTP 503 (server
#                            unavailable) or 429 (rate limited). On a busy shared
#                            Ollama this is the retryable case by definition: the
#                            request is fine, the server is momentarily saturated,
#                            and the next attempt meets a different server state.
#
# Deliberately not retried:
#
#   ollama.ResponseError  every *other* status — a 400 or a 404 is a settled
#                         answer about the request itself, and it will answer the
#                         same way next time. Only 503/429 are carved out, by
#                         status code, via RetryableResponseError below; the
#                         caught type is not widened.
#   OutputParserException the model answered and the answer did not parse. At
#                         temperature 0 — the default for a non-thinking model --
#                         or with a seed set, the reply is deterministic, so
#                         three attempts buy three identical failures at three
#                         times the cost. That was the previous behaviour:
#                         a bare .with_retry() retries on Exception.


class _RetryableResponseErrorMeta(type):
    """isinstance() hook: a ``ResponseError`` is retryable only for 503/429.

    ``RETRYABLE_ERRORS`` is a plain tuple of exception types because that is all
    ``Runnable.with_retry`` accepts, and the retry decision is ``isinstance``.
    Narrowing the exclusion by status code therefore means a type whose
    ``isinstance`` check inspects ``status_code`` rather than the class alone —
    every non-503/429 ``ResponseError`` stays outside the tuple's reach.
    """

    _RETRYABLE_STATUS = frozenset({429, 503})

    def __instancecheck__(cls, instance: object) -> bool:
        return (
            isinstance(instance, ollama.ResponseError)
            and getattr(instance, "status_code", None) in cls._RETRYABLE_STATUS
        )


class RetryableResponseError(Exception, metaclass=_RetryableResponseErrorMeta):
    """Marker for ``RETRYABLE_ERRORS`` — never raised, never instantiated.

    Only its ``isinstance`` behaviour is used: ``isinstance(err, this)`` is true
    for an ``ollama.ResponseError`` whose ``status_code`` is 503 or 429. It
    subclasses ``Exception`` so ``with_retry``'s type validation accepts it.
    """


RETRYABLE_ERRORS = (ConnectionError, httpx.TransportError, RetryableResponseError)
RETRY_ATTEMPTS = 3

# LangChain message types to the role names Ollama's chat API expects.
_OLLAMA_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def _strip_think_tags(msg):
    """Strip <think>...</think> blocks from model output before JSON parsing.

    Some models (e.g. qwen3.5) ignore think:false and always emit reasoning
    tokens as plain content, which breaks structured-output parsing.
    """
    if hasattr(msg, "content") and isinstance(msg.content, str):
        return msg.model_copy(update={"content": _THINK_RE.sub("", msg.content).strip()})
    return msg


from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import resolve_parser_model
from llm_extractinator.prompt_utils import (
    build_few_shot_prompt,
    build_zero_shot_prompt,
    describe_fields,
    extra_instructions,
    task_description,
)
from llm_extractinator.validator import (
    handle_prediction_failure,
    success_diagnostics,
)


# Configure logging
logger = logging.getLogger(__name__)


class _TruncatingEmbeddings(Embeddings):
    """Wraps an embeddings model and truncates texts to avoid exceeding the model's context limit.

    The clip is not free: MMR example selection ranks candidates by embedding
    similarity, so a document that loses its tail can be ranked differently and a
    different few-shot set reaches the model for that row. That used to happen
    with no trace at all. The clip still happens — this class only makes it
    *visible* when it bites, with an aggregate count rather than a line per text
    so a large example pool does not flood the log.
    """

    def __init__(self, base: Embeddings, max_chars: int = 2000) -> None:
        self._base = base
        self._max_chars = max_chars
        self._query_clip_warned = False

    def embed_documents(self, texts):
        clipped = sum(1 for t in texts if len(t) > self._max_chars)
        if clipped:
            logger.warning(
                "Truncated %d of %d example text(s) to %d chars before similarity "
                "embedding; this can change which few-shot examples are selected.",
                clipped,
                len(texts),
                self._max_chars,
            )
        return self._base.embed_documents([t[: self._max_chars] for t in texts])

    def embed_query(self, text: str):
        if len(text) > self._max_chars and not self._query_clip_warned:
            # One line per Predictor, not per row: select_examples() calls this
            # once for every input document.
            self._query_clip_warned = True
            logger.warning(
                "Truncated an input document to %d chars before similarity "
                "embedding; this can change which few-shot examples are selected. "
                "Further truncations this run are not logged.",
                self._max_chars,
            )
        return self._base.embed_query(text[: self._max_chars])


class Predictor:
    """
    A class responsible for generating and executing predictions on test data using a language model.
    """

    def __init__(
        self,
        model: ChatOllama,
        task_config: Dict[str, Any],
        examples_path: Path,
        num_examples: int,
        task_dir: Path = Path(os.getcwd()) / "tasks",
        ollama_host: Optional[str] = None,
    ) -> None:
        """
        Initialize the Predictor with the provided model, task configuration, and paths.
        """
        self.model = model
        self.task_config = task_config
        self.num_examples = num_examples
        self.examples_path = examples_path
        self.task_dir = task_dir
        self.ollama_host = ollama_host
        self._extract_task_info()

    def _extract_task_info(self) -> None:
        """
        Extract task information from the task configuration.
        """
        self.length = self.task_config.get("Length")
        self.description = task_description(self.task_config)
        self.extra_instructions = extra_instructions(self.task_config)
        self.input_field = self.task_config.get("Input_Field")
        self.test_path = self.task_config.get("Data_Path")
        self.parser_format = self.task_config.get("Parser_Format")

    def prepare_prompt_ollama(
        self, embedding_model: str, examples: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.
        """
        self.parser_model = resolve_parser_model(self.task_config, self.task_dir)
        self.base_parser = PydanticOutputParser(pydantic_object=self.parser_model)
        field_guide = describe_fields(self.parser_model)

        if examples:
            logger.info("Creating few-shot prompt.")
            if self.ollama_host:
                logger.info("Externally managed Ollama server — skipping embedding model pull.")
            else:
                ollama.pull(embedding_model)
            self.embedding_model = _TruncatingEmbeddings(
                OllamaEmbeddings(model=embedding_model, base_url=self.ollama_host)
            )
            from langchain_core.example_selectors import (
                MaxMarginalRelevanceExampleSelector,
            )

            self.example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                examples, self.embedding_model, Chroma, k=self.num_examples
            )
            self.prompt = build_few_shot_prompt(
                example_selector=self.example_selector,
            ).partial(
                description=self.description,
                fields=field_guide,
                extra_instructions=self.extra_instructions,
            )
        else:
            logger.info("Creating zero-shot prompt.")
            self.prompt = build_zero_shot_prompt().partial(
                description=self.description,
                fields=field_guide,
                extra_instructions=self.extra_instructions,
            )

    def assemble_prompt(self, text: str) -> List[Dict[str, str]]:
        """The exact messages this run would send for one input, as Ollama dicts.

        Goes through the same template and the same example selector the
        predictions use, so what gets measured is what will actually be sent —
        including the few-shot exchanges, which are the part an estimate is
        least able to predict.
        """
        return [
            {
                "role": _OLLAMA_ROLES.get(message.type, "user"),
                "content": str(message.content),
            }
            for message in self.prompt.format_messages(input=text)
        ]

    def predict(self, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.
        """
        logger.info("Starting prediction on test data with %d samples.", len(test_data))
        # `required` is left exactly as pydantic emits it. It used to be forced to
        # *every* property whenever pydantic omitted the key — and pydantic omits
        # it only when every field has a default, which is precisely when the
        # task author marked them all optional. That turned "everything optional"
        # into "everything mandatory" and made the model invent values for the
        # fields it had been given permission to leave out.
        response_format = self.parser_model.model_json_schema()
        bound_llm = self.model.bind(format=response_format)
        # The retry wraps the model call alone, not the parse that follows it.
        # Scoping it this way means a parse failure structurally cannot trigger
        # another generation, whatever RETRYABLE_ERRORS is later widened to.
        model = (
            bound_llm.with_retry(
                retry_if_exception_type=RETRYABLE_ERRORS,
                stop_after_attempt=RETRY_ATTEMPTS,
                wait_exponential_jitter=True,
            )
            | RunnableLambda(_strip_think_tags)
            | self.base_parser
        )
        chain = self.prompt | model
        test_data_processed = [
            {"input": row[self.input_field]} for _, row in test_data.iterrows()
        ]
        callbacks = BatchCallBack(len(test_data_processed))
        raw_results = chain.batch(
            test_data_processed,
            config={"callbacks": [callbacks]},
            return_exceptions=True,
        )
        callbacks.progress_bar.close()

        final_results = []
        for result in raw_results:
            if isinstance(result, Exception):
                final_results.append(
                    handle_prediction_failure(result, self.parser_model)
                )
            else:
                result_dict = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                result_dict["status"] = "success"
                result_dict.update(success_diagnostics())
                final_results.append(result_dict)

        self._log_outcome(final_results)
        return final_results

    @staticmethod
    def _log_outcome(results: List[Dict[str, Any]]) -> None:
        """Say what happened, in terms someone can act on.

        A bare count of failures does not distinguish "the server went away"
        from "the model wrote prose", and those need opposite responses. The
        breakdown by exception type is the cheapest thing that does.
        """
        failures = [r for r in results if r.get("status") == "failure"]
        if not failures:
            logger.info("Prediction completed: %d rows, no failures.", len(results))
            return

        breakdown = Counter(r.get("error_type") for r in failures)
        logger.warning(
            "%d of %d rows failed (%s). The model's own output for each is in "
            "failures.json alongside the predictions.",
            len(failures),
            len(results),
            ", ".join(f"{name} x{count}" for name, count in breakdown.most_common()),
        )
