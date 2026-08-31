import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from llm_extractinator.callbacks import BatchCallBack
from llm_extractinator.output_parsers import load_parser
from llm_extractinator.prompt_utils import build_translation_prompt
from llm_extractinator.predictor import RETRY_ATTEMPTS, RETRYABLE_ERRORS
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
        # Same rule as the extraction path: only transport failures are worth
        # re-sending. See predictor.RETRYABLE_ERRORS.
        model = self.model.with_structured_output(
            schema=self.parser_model
        ).with_retry(
            retry_if_exception_type=RETRYABLE_ERRORS,
            stop_after_attempt=RETRY_ATTEMPTS,
            wait_exponential_jitter=True,
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
                final_results.append(result_dict)

        logger.info("Saving translations.")
        # A failed translation keeps the original text. The failure default used
        # to be written straight into the document — an empty string then, None
        # now — which silently replaced the source text with nothing and left the
        # extraction step to run on a blank. Keeping the untranslated original is
        # the honest degradation: the run continues on real content, and the
        # count below says how much of it was never translated.
        failed = sum(1 for res in final_results if res.get("status") == "failure")
        if failed:
            logger.warning(
                "%d of %d rows could not be translated; those keep their original "
                "text and are extracted untranslated.",
                failed,
                len(final_results),
            )
        test_data[self.input_field] = [
            res["translation"] if res.get("status") == "success" else original
            for res, original in zip(final_results, test_data[self.input_field])
        ]
        test_data.to_json(savepath, orient="records", indent=4)
        logger.info("Translation completed successfully.")
