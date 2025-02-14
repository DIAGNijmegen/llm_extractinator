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
from llm_extractinator.validator import validate_results


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
        base_parser = PydanticOutputParser(pydantic_object=self.parser_model)

        # We also keep the usual "format_instructions" for instructing the model
        self.format_instructions = base_parser.get_format_instructions()

        # Create an OutputFixingParser that wraps the base parser
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=base_parser,
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

        # The chain output will be Pydantic-validated or auto-fixed objects
        raw_results = chain.batch(
            test_data_processed,
            config={"callbacks": [callbacks]},
        )
        callbacks.progress_bar.close()

        # raw_results will be a list of pydantic objects (or dicts) if all went well
        # We'll do a final pass to ensure each item definitely matches schema,
        # or fallback to `handle_failure` if the parser gave up.
        final_results = []
        for item in raw_results:
            # The OutputFixingParser should yield actual BaseModel objects,
            # but we might still do a fallback check:
            final_results.append(validate_results(item, self.parser_model))

        # Return list of dicts (since model_dump() might be more convenient than raw pydantic objects)
        return [
            r.model_dump() if hasattr(r, "model_dump") else r for r in final_results
        ]
