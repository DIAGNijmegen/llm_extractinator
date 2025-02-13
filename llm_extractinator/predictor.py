import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin
from uuid import UUID

import ollama
import pandas as pd
from langchain.globals import set_debug
from langchain_chroma import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.example_selectors import (
    BaseExampleSelector,
    MaxMarginalRelevanceExampleSelector,
)
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from llm_extractinator.output_parsers import load_parser
from llm_extractinator.utils import extract_json_from_text, handle_failure

# Disable debug mode for LangChain
set_debug(False)


class RandomExampleSelector(BaseExampleSelector):
    """A custom random example selector."""

    def __init__(self, examples: List[Dict[str, Any]], k: int = 3):
        self.examples = examples
        self.k = k

    def add_example(self, example: Dict[str, Any]):
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        return random.sample(self.examples, self.k)


class BatchCallBack(BaseCallbackHandler):
    """A callback handler to manage progress during batch processing."""

    def __init__(self, total: int):
        super().__init__()
        self.count = 0
        self.progress_bar = tqdm(total=total)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.count += 1
        self.progress_bar.update(1)


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
        format: str = "json",
    ) -> None:
        """
        Initialize the Predictor with the provided model, task configuration, and paths.

        Args:
            model
            task_config (Dict): Configuration for the task, including task name, input, and label details.
            examples_path (Path): Path where the generated examples are saved.
            num_examples (int): Number of examples to select for few-shot learning.
        """
        self.model = model
        self.task_config = task_config
        self.num_examples = num_examples
        self.examples_path = examples_path
        self.format = format

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

    def generate_translations(self, data: pd.DataFrame, savepath: Path) -> pd.DataFrame:
        """
        Generate translations for the given data.

        Args:
            data (pd.DataFrame): The data to translate.

        Returns:
            pd.DataFrame: The translated data.
        """
        parser_model = load_parser(task_type="Translation", parser_format=None)
        self.translation_parser = JsonOutputParser(pydantic_object=parser_model)
        self.translation_format_instructions = (
            self.translation_parser.get_format_instructions()
        )

        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("translation/system_prompt")
                + "\n**Format instructions:**\n{format_instructions}",
                input_variables=[],
                partial_variables={
                    "format_instructions": self.translation_format_instructions
                },
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("translation/human_prompt"),
                input_variables=["input"],
            )
        )
        translation_prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt]
        )

        # Chain: prompt -> model -> output parser
        chain = translation_prompt | self.model | self.translation_parser

        # Preprocess data
        data_processed = [
            {"input": row[self.input_field]} for _, row in data.iterrows()
        ]

        # Generate translations
        callbacks = BatchCallBack(len(data_processed))
        results = chain.batch(data_processed, config={"callbacks": [callbacks]})
        results = self.validate_and_fix_results(results, parser_model=parser_model)
        translations = [result["translation"] for result in results]

        # Replace the original text in self.input_field with the translated text
        data[self.input_field] = translations

        # Save translations to file
        data.to_json(savepath, orient="records", indent=4)

    def _create_system_prompt(
        self, template_base: str, format_instructions: str
    ) -> SystemMessagePromptTemplate:
        """
        Helper function to create a system prompt template.

        Args:
            template_base (str): The base template string.
            format_instructions (str): Instructions for formatting output.

        Returns:
            SystemMessagePromptTemplate: A system message prompt template.
        """
        return SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=template_base
                + "\n**Format instructions:**\n{format_instructions}",
                input_variables=[],
                partial_variables={"format_instructions": format_instructions},
            )
        )

    def _create_human_prompt(
        self, template_name: str, input_vars: List[str]
    ) -> HumanMessagePromptTemplate:
        """
        Helper function to create a human prompt template.

        Args:
            template_name (str): The name of the template file to load.
            input_vars (List[str]): List of input variables for the human prompt.

        Returns:
            HumanMessagePromptTemplate: A human message prompt template.
        """
        template = self._load_template(template_name)
        return HumanMessagePromptTemplate(
            prompt=PromptTemplate(template=template, input_variables=input_vars)
        )

    def _load_template(self, template_name: str) -> str:
        """
        Load a template from the 'prompt_templates' directory.

        Args:
            template_name (str): The name of the template file.

        Returns:
            str: The content of the template file.
        """
        template_path = (
            Path(__file__).parent / "prompt_templates" / f"{template_name}.txt"
        )
        with template_path.open() as f:
            return f.read()

    def prepare_prompt_ollama(
        self, model_name: str, examples: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Prepare the system and human prompts for few-shot learning based on provided examples.

        Args:
            examples (Optional[List[Dict[str, Any]]]): The examples to use for the prompt.
        """
        self.parser_model = load_parser(
            task_type=self.type, parser_format=self.parser_format
        )
        self.parser = JsonOutputParser(pydantic_object=self.parser_model)
        self.parser_model_fields = self.parser_model.model_fields
        self.format_instructions = self.parser.get_format_instructions()

        if examples:
            ollama.pull(model_name)
            self.embedding_model = OllamaEmbeddings(model=model_name)
            self.example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                examples, self.embedding_model, Chroma, k=self.num_examples
            )
            self.prompt = self._build_few_shot_prompt()
        else:
            self.prompt = self._build_zero_shot_prompt()

    def _build_few_shot_prompt(self) -> ChatPromptTemplate:
        """
        Build a few-shot prompt with examples.

        Args:
            examples (List[Dict[str, Any]]): The few-shot examples.

        Returns:
            ChatPromptTemplate: A chat prompt template for few-shot learning.
        """
        final_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/system_prompt").format(
                    task=self.task, description=self.description
                )
                + "\n**Format instructions:**\n{format_instructions}",
                input_variables=[],
                partial_variables={"format_instructions": self.format_instructions},
            )
        )
        final_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/human_prompt"),
                input_variables=["input"],
            )
        )
        example_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/human_prompt"),
                input_variables=["input"],
            )
        )
        example_ai_prompt = AIMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/ai_prompt"),
                input_variables=["output"],
            )
        )
        example_prompt = ChatPromptTemplate.from_messages(
            [example_human_prompt, example_ai_prompt]
        )
        final_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self.example_selector,
            input_variables=["input"],
        )
        return ChatPromptTemplate.from_messages(
            [final_system_prompt, final_few_shot_prompt, final_human_prompt]
        )

    def _build_zero_shot_prompt(self) -> ChatPromptTemplate:
        """
        Build a zero-shot prompt without examples.

        Returns:
            ChatPromptTemplate: A chat prompt template for zero-shot learning.
        """
        final_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/system_prompt").format(
                    task=self.task, description=self.description
                )
                + "\n**Format instructions:**\n{format_instructions}",
                input_variables=[],
                partial_variables={"format_instructions": self.format_instructions},
            )
        )
        final_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("data_extraction/human_prompt"),
                input_variables=["input"],
            )
        )
        return ChatPromptTemplate.from_messages(
            [final_system_prompt, final_human_prompt]
        )

    def ollama_prepare_fixing_prompt(self, format_instructions) -> ChatPromptTemplate:
        """Prepare a prompt for fixing incorrectly formatted JSON output."""
        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("output_fixing/system_prompt"),
                input_variables=[],
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=self._load_template("output_fixing/human_prompt"),
                input_variables=["completion"],
                partial_variables={"format_instructions": format_instructions},
            )
        )
        fixing_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        return fixing_prompt

    def validate_and_fix_results(
        self,
        results: List[Dict[str, Any]],
        parser_model: BaseModel,
        max_attempts: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Validate and batch-fix results while keeping metadata separate.

        Args:
            results (List[Dict[str, Any]]): The list of results to validate and fix.
            parser_model (BaseModel): The Pydantic model for validation.
            max_attempts (int, optional): Maximum attempts to fix each result. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: The validated and corrected results.
        """

        parser = JsonOutputParser(pydantic_object=parser_model)
        parser_model_fields = parser_model.model_fields
        format_instructions = parser.get_format_instructions()

        fixing_chain = (
            self.ollama_prepare_fixing_prompt(format_instructions=format_instructions)
            | self.model
            | StrOutputParser
            | extract_json_from_text
        )

        # Store metadata separately
        index_mapping = {i: result for i, result in enumerate(results)}

        # Strip metadata before validation
        stripped_results = [
            {k: v for k, v in result.items() if k in parser_model_fields}
            for result in results
        ]

        attempt = 0
        while attempt < max_attempts:
            # Identify invalid results
            invalid_indices = []
            for i, r in enumerate(stripped_results):
                if isinstance(r, BaseModel):
                    r = r.model_dump()
                try:
                    parser_model.model_validate(r)
                except ValidationError:
                    invalid_indices.append(i)

            if not invalid_indices:
                break  # Stop if all results are valid

            print(
                f"Retry {attempt + 1}: Fixing {len(invalid_indices)} invalid results..."
            )

            invalid_results = [stripped_results[i] for i in invalid_indices]
            fixing_inputs = [{"completion": str(result)} for result in invalid_results]

            try:
                callbacks = BatchCallBack(len(invalid_indices))
                fixed_results = fixing_chain.batch(
                    fixing_inputs, config={"callbacks": [callbacks]}
                )
                callbacks.progress_bar.close()

                for idx, fixed_result in zip(invalid_indices, fixed_results):
                    if self.format != "json":
                        extracted_json = extract_json_from_text(str(fixed_result))
                        if extracted_json:
                            fixed_result = extracted_json

                    try:
                        if isinstance(fixed_result, BaseModel):
                            fixed_result = fixed_result.model_dump()

                        # Strip unnecessary metadata before validation
                        fixed_result = {
                            k: v
                            for k, v in fixed_result.items()
                            if k in parser_model_fields
                        }
                        parser_model.model_validate(fixed_result)

                        stripped_results[idx] = fixed_result
                    except ValidationError:
                        pass  # Leave it in invalid state for now

            except Exception as e:
                print(f"Batch fixing failed at attempt {attempt + 1}: {str(e)}")

            attempt += 1

        # Assign default values for failed results
        for i in invalid_indices:
            stripped_results[i] = {
                key: handle_failure(parser_model_fields[key].annotation)
                for key in parser_model_fields
            }

        # Restore the original order using index_mapping
        final_results = [stripped_results[i] for i in sorted(index_mapping.keys())]

        return final_results

    def predict(self, test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.

        Args:
            test_data (pd.DataFrame): The test data containing text.

        Returns:
            List[Dict[str, Any]]: A list of prediction results.
        """

        chain = self.prompt | self.model | StrOutputParser() | extract_json_from_text

        # Preprocess test data
        test_data_processed = [
            {"input": report} for report in test_data[self.input_field]
        ]
        callbacks = BatchCallBack(len(test_data_processed))
        results = chain.batch(test_data_processed, config={"callbacks": [callbacks]})
        print("DEBUG:", results)
        callbacks.progress_bar.close()

        return self.validate_and_fix_results(results, parser_model=self.parser_model)
