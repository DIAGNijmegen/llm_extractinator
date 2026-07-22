import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _strip_think_tags(msg):
    """Strip <think>...</think> blocks from model output before JSON parsing.

    Some models (e.g. qwen3.5) ignore think:false and always emit reasoning
    tokens as plain content, which breaks structured-output parsing.
    """
    if hasattr(msg, "content") and isinstance(msg.content, str):
        return msg.model_copy(update={"content": _THINK_RE.sub("", msg.content).strip()})
    return msg


from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser, load_parser_pydantic
from llm_extractinator.prompt_utils import build_few_shot_prompt, build_zero_shot_prompt
from llm_extractinator.validator import handle_prediction_failure


class _TruncatingEmbeddings(Embeddings):
    """Wraps an embeddings model and truncates texts to avoid exceeding the model's context limit."""

    def __init__(self, base: Embeddings, max_chars: int = 2000) -> None:
        self._base = base
        self._max_chars = max_chars

    def embed_documents(self, texts):
        return self._base.embed_documents([t[: self._max_chars] for t in texts])

    def embed_query(self, text: str):
        return self._base.embed_query(text[: self._max_chars])

# Configure logging
logger = logging.getLogger(__name__)


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
        self.description = self.task_config.get("Description")
        self.input_field = self.task_config.get("Input_Field")
        self.test_path = self.task_config.get("Data_Path")
        self.parser_format = self.task_config.get("Parser_Format")

    def prepare_prompt_ollama(
        self, embedding_model: str, examples: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.
        """
        if isinstance(self.parser_format, dict):
            try:
                self.parser_model = load_parser(
                    task_type="Extraction", parser_format=self.parser_format
                )
            except KeyError as e:
                logger.error(
                    f"Missing required key in parser format dictionary: {e}"
                )
                raise ValueError(f"Invalid parser format dictionary: {e}") from e
            except Exception as e:
                logger.error(
                    f"Failed to load parser model from dictionary format: {type(e).__name__}: {e}"
                )
                raise
        else:
            parser_path = self.task_dir / "parsers" / self.parser_format
            try:
                if not parser_path.exists():
                    raise FileNotFoundError(f"Parser file not found: {parser_path}")
                self.parser_model = load_parser_pydantic(parser_path=parser_path)
            except FileNotFoundError as e:
                logger.error(str(e))
                raise
            except (ImportError, AttributeError, SyntaxError) as e:
                logger.error(
                    f"Failed to import parser from {parser_path}: {type(e).__name__}: {e}"
                )
                raise ValueError(
                    f"Parser file '{self.parser_format}' has invalid Python code or structure"
                ) from e
            except Exception as e:
                logger.error(
                    f"Failed to load parser model from {parser_path}: {type(e).__name__}: {e}"
                )
                raise
        self.base_parser = PydanticOutputParser(pydantic_object=self.parser_model)

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
            ).partial(description=self.description)
        else:
            logger.info("Creating zero-shot prompt.")
            self.prompt = build_zero_shot_prompt().partial(description=self.description)

    def predict(self, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.
        """
        logger.info("Starting prediction on test data with %d samples.", len(test_data))
        response_format = self.parser_model.model_json_schema()
        if "required" not in response_format:
            response_format["required"] = list(response_format.get("properties", {}).keys())
        bound_llm = self.model.bind(format=response_format)
        model = (
            bound_llm | RunnableLambda(_strip_think_tags) | self.base_parser
        ).with_retry()
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
        failure_counter = 0
        for input_data, result in zip(test_data_processed, raw_results):
            if isinstance(result, Exception):
                final_results.append(
                    handle_prediction_failure(result, input_data, self.parser_model)
                )
                failure_counter += 1
            else:
                result_dict = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                result_dict["status"] = "success"
                final_results.append(result_dict)

        logger.info("Prediction completed successfully.")
        logger.info(f"Failed predictions: {failure_counter}")
        return final_results
