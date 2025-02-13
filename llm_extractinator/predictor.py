# predictor.py
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser
from llm_extractinator.prompt_utils import build_few_shot_prompt, build_zero_shot_prompt
from llm_extractinator.utils import extract_json_from_text
from llm_extractinator.validator import validate_and_fix_results


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

        Args:
            model: The language model to be used (e.g., ChatOllama).
            task_config: Configuration for the task, including task name, input, etc.
            examples_path: Path where the generated examples are saved.
            num_examples: Number of examples to select for few-shot learning.
            output_format: Expected output format ("json" by default).
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
        self, model_name: str, examples: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.
        """
        # Load the parser model for data extraction
        self.parser_model = load_parser(
            task_type=self.type, parser_format=self.parser_format
        )
        self.parser = JsonOutputParser(pydantic_object=self.parser_model)
        self.format_instructions = self.parser.get_format_instructions()

        if examples:
            ollama.pull(model_name)
            self.embedding_model = OllamaEmbeddings(model=model_name)

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
            self.prompt = build_zero_shot_prompt(
                task=self.task,
                description=self.description,
                format_instructions=self.format_instructions,
            )

    def predict(self, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.

        Args:
            test_data: The test data containing text in self.input_field.

        Returns:
            A list of prediction results.
        """
        chain = self.prompt | self.model | StrOutputParser() | extract_json_from_text

        test_data_processed = [
            {"input": row[self.input_field]} for _, row in test_data.iterrows()
        ]
        callbacks = BatchCallBack(len(test_data_processed))
        raw_results = chain.batch(
            test_data_processed,
            config={"callbacks": [callbacks]},
        )
        callbacks.progress_bar.close()

        # Validate and fix the results
        final_results = validate_and_fix_results(
            raw_results,
            parser_model=self.parser_model,
            model=self.model,
            output_format=self.output_format,
        )

        return final_results
