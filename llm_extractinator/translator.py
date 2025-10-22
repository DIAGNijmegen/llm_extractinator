import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser
from llm_extractinator.prompt_utils import build_translation_prompt
from llm_extractinator.validator import handle_prediction_failure

# Configure logging
logger = logging.getLogger(__name__)


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

    def _prepare_translation_prompt(
        self,
    ) -> None:
        """
        Prepare the translation prompt.
        """
        self.parser_model = load_parser(task_type="Translation", parser_format=None)
        self.prompt = build_translation_prompt()

    def translate(
        self, test_data: pd.DataFrame, savepath: Path
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on the test data.
        """
        logger.info("Starting translation with %d samples.", len(test_data))
        self._prepare_translation_prompt()
        model = self.model.with_structured_output(schema=self.parser_model).with_retry()
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

        logger.info("Saving translations.")
        test_data[self.input_field] = [res["translation"] for res in final_results]
        test_data.to_json(savepath, orient="records", indent=4)
        logger.info("Translation completed successfully.")
