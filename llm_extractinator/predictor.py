import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import pandas as pd
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser
from llm_extractinator.prompt_utils import build_few_shot_prompt, build_zero_shot_prompt
from llm_extractinator.validator import handle_prediction_failure

# Configure logging
logger = logging.getLogger(__name__)


def handle_failure(error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle failures during prediction by logging the error and returning a default response.
    """
    logger.error("Prediction failed for input: %s. Error: %s", input_data, str(error))
    return {
        "input": input_data.get("input", "Unknown"),
        "error": str(error),
        "prediction": "default_value",
    }


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
        output_format: str = "json",
    ) -> None:
        """
        Initialize the Predictor with the provided model, task configuration, and paths.
        """
        self.model = model
        self.task_config = task_config
        self.num_examples = num_examples
        self.examples_path = examples_path
        self.output_format = output_format
        self._extract_task_info()

    def _extract_task_info(self) -> None:
        """
        Extract task information from the task configuration.
        """
        self.task = self.task_config.get("Task")
        self.type = self.task_config.get("Type")
        self.length = self.task_config.get("Length")
        self.description = self.task_config.get("Description")
        self.input_field = self.task_config.get("Input_Field")
        self.train_path = self.task_config.get("Example_Path")
        self.test_path = self.task_config.get("Data_Path")
        self.task_name = self.task_config.get("Task_Name")
        self.parser_format = self.task_config.get("Parser_Format")

    def prepare_prompt_ollama(
        self, embedding_model: str, examples: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.
        """
        self.parser_model = load_parser(
            task_type=self.type, parser_format=self.parser_format
        )
        self.base_parser = PydanticOutputParser(pydantic_object=self.parser_model)
        self.format_instructions = self.base_parser.get_format_instructions()
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.base_parser,
            llm=self.model,
            max_retries=3,
        )

        if examples:
            logger.info("Creating few-shot prompt.")
            ollama.pull(embedding_model)
            self.embedding_model = OllamaEmbeddings(model=embedding_model)
            from langchain_core.example_selectors import (
                MaxMarginalRelevanceExampleSelector,
            )

            self.example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                examples, self.embedding_model, Chroma, k=self.num_examples
            )
            self.prompt = build_few_shot_prompt(
                task=self.task,
                description=self.description,
                format_instructions=self.format_instructions,
                example_selector=self.example_selector,
            )
        else:
            logger.info("Creating zero-shot prompt.")
            self.prompt = build_zero_shot_prompt(
                task=self.task,
                description=self.description,
                format_instructions=self.format_instructions,
            )

    def predict(self, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.
        """
        logger.info("Starting prediction on test data with %d samples.", len(test_data))
        chain = self.prompt | self.model | self.fixing_parser
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
