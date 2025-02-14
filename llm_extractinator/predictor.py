import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import pandas as pd
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser
from llm_extractinator.prompt_utils import build_few_shot_prompt, build_zero_shot_prompt
from llm_extractinator.validator import handle_prediction_failure


def handle_failure(error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle failures during prediction by logging the error and returning a default response.

    Args:
        error (Exception): The exception encountered during prediction.
        input_data (Dict[str, Any]): The input data that caused the error.

    Returns:
        Dict[str, Any]: A default response with an error message, ensuring automatic evaluation is not affected.
    """
    logging.error(f"Prediction failed for input: {input_data}. Error: {str(error)}")

    # Return a default response indicating failure, ensuring evaluation continues smoothly
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
        self, model_name: str, examples: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.
        """
        # Load the parser model for data extraction
        self.parser_model = load_parser(
            task_type=self.type, parser_format=self.parser_format
        )
        # We’ll build an actual LangChain PydanticOutputParser
        self.base_parser = PydanticOutputParser(pydantic_object=self.parser_model)

        # We also keep the usual "format_instructions" for instructing the model
        self.format_instructions = self.base_parser.get_format_instructions()

        # Create an OutputFixingParser that wraps the base parser
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.base_parser,
            llm=self.model,
            max_retries=3,
        )

        if examples:
            ollama.pull(model_name)  # Ensure local model is available
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

        Returns:
            A list of validated (or defaulted) prediction results.
        """
        # We'll do a simple chain: prompt -> model -> fix_parser
        chain = self.prompt | self.model | self.fixing_parser

        # Prepare input to the chain
        test_data_processed = [
            {"input": row[self.input_field]} for _, row in test_data.iterrows()
        ]

        # Callback for progress
        callbacks = BatchCallBack(len(test_data_processed))

        # Execute batch prediction with error handling
        raw_results = chain.batch(
            test_data_processed,
            config={"callbacks": [callbacks]},
            return_exceptions=True,
        )

        callbacks.progress_bar.close()

        final_results = []
        for input_data, result in zip(test_data_processed, raw_results):
            if isinstance(result, Exception):
                final_results.append(
                    handle_prediction_failure(result, input_data, self.parser_model)
                )
            else:
                result_dict = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                result_dict["status"] = "success"
                final_results.append(result_dict)

        return final_results
