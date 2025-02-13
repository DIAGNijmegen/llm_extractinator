# translator.py
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser
from llm_extractinator.validator import validate_and_fix_results


class Translator:
    """
    A helper class responsible for generating translations using a language model.
    """

    def __init__(
        self,
        model,
        input_field: str,
    ):
        self.model = model
        self.input_field = input_field

        # Load parser for translations
        parser_model = load_parser(task_type="Translation", parser_format=None)
        self.translation_parser = JsonOutputParser(pydantic_object=parser_model)
        self.translation_format_instructions = (
            self.translation_parser.get_format_instructions()
        )
        self.parser_model = parser_model

        self.translation_prompt = self._create_translation_prompt()

    def _create_translation_prompt(self) -> ChatPromptTemplate:
        """
        Create the ChatPromptTemplate used for translations.
        """
        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=(
                    "You're a professional translator. "
                    "Translate the following text from English to Spanish.\n"
                    "**Format instructions:**\n{format_instructions}"
                ),
                input_variables=[],
                partial_variables={
                    "format_instructions": self.translation_format_instructions
                },
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="{input}",
                input_variables=["input"],
            )
        )
        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    def generate_translations(self, data: pd.DataFrame, savepath: Path) -> pd.DataFrame:
        """
        Generate translations for the given data, validate them, and save to disk.
        """
        chain = self.translation_prompt | self.model | self.translation_parser

        data_processed = [
            {"input": row[self.input_field]} for _, row in data.iterrows()
        ]

        callbacks = BatchCallBack(len(data_processed))
        raw_results = chain.batch(data_processed, config={"callbacks": [callbacks]})

        # Validate and fix results
        results = validate_and_fix_results(
            raw_results,
            parser_model=self.parser_model,
            model=self.model,
            output_format="json",
            max_attempts=3,
        )

        # Extract final translations
        translations = [res["translation"] for res in results]
        data[self.input_field] = translations

        data.to_json(savepath, orient="records", indent=4)
        return data
