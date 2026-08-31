import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain_ollama import ChatOllama

from llm_extractinator.budget import ContextBudgetError, resolve_budget
from llm_extractinator.ollama_server import (
    ModelInfo,
    measure_prompt_tokens,
    model_capabilities,
)
from llm_extractinator.predictor import Predictor
from llm_extractinator.translator import Translator
from llm_extractinator.run_config import resolve_thinking, resolved_temperature
from llm_extractinator.utils import (
    chunk_files,
    run_folder_name,
    save_json,
    write_failures,
)

# Configure logging
logger = logging.getLogger(__name__)


# Re-exported: it is defined next to the resolver that raises it most often,
# but this module raises it too and callers import it from either.
__all__ = ["ContextBudgetError", "PredictionTask"]


class PredictionTask:
    REQUIRED_PARAMS = {
        "task_id",
        "model_name",
        "embedding_model",
        "output_dir",
        "task_dir",
        "num_examples",
        "n_runs",
        "run_name",
        "temperature",
        "max_context_len",
        "num_predict",
        "data_dir",
        "example_dir",
        "translation_dir",
        "chunk_size",
        "test_run_size",
        "translate",
        "verbose",
        "overwrite",
        "seed",
        "top_k",
        "top_p",
        "reasoning_model",
        "no_reasoning",
        "model_info",
        "budget",
        "max_context_cap",
        "ollama_host",
        "train_path",
        "test_path",
        "input_field",
        "task_name",
        "task_config",
        "data_split",
        "train",
        "test",
    }

    def __init__(self, **kwargs) -> None:
        missing_params = [
            param for param in self.REQUIRED_PARAMS if param not in kwargs
        ]
        if missing_params:
            logger.error("Missing required parameters: %s", ", ".join(missing_params))
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

        for key in self.REQUIRED_PARAMS:
            setattr(self, key, kwargs.get(key, None))

        self.output_path_base = Path(self.output_dir) / Path(self.run_name)

        # What the server says about the model. TaskRunner inspects it once,
        # after the pull and before sizing the window, and passes the answer in;
        # detection here is the fallback for constructing a PredictionTask on
        # its own. Either way the *interpretation* is shared, so the runner and
        # this object cannot disagree about whether a run reasons.
        info = self.model_info
        if info is None:
            info = model_capabilities(self.model_name, host=self.ollama_host)
        self._model_info = info
        _detected = info.supports_thinking

        if self.no_reasoning and self.reasoning_model:
            logger.warning(
                "Both reasoning_model and no_reasoning were set; "
                "no_reasoning wins and reasoning mode stays off."
            )
        self._is_thinking = resolve_thinking(
            detected=_detected,
            reasoning_model=self.reasoning_model,
            no_reasoning=self.no_reasoning,
        )

        # What we actually send Ollama as `think`. The three states are distinct:
        #   True  - think, and return the reasoning separately from the answer
        #   False - do not think (only meaningful, and only accepted, for a model
        #           that advertises the capability: Ollama answers 400 "does not
        #           support thinking" if `think` is sent to a model without it)
        #   None  - say nothing, leave the model on its default
        # So switching reasoning off is None for an ordinary model (there is
        # nothing to disable) and an explicit False for a thinking one.
        if self._is_thinking:
            self._reasoning_param = True
        elif self.no_reasoning and _detected:
            self._reasoning_param = False
            logger.info(
                "Reasoning explicitly disabled for thinking model '%s'.",
                self.model_name,
            )
        else:
            self._reasoning_param = None

        # Temperature follows the same shape: None means "decide for me", and the
        # decision needs _is_thinking, so this is the one place that can make it.
        if self.temperature is None:
            self.temperature = resolved_temperature(None, self._is_thinking)
            logger.info(
                "Temperature not set; using %s for %s model '%s'.",
                self.temperature,
                "thinking" if self._is_thinking else "non-thinking",
                self.model_name,
            )
        elif self._is_thinking and self.temperature == 0:
            logger.warning(
                "Greedy decoding (temperature 0) on thinking model '%s': its own "
                "guidance advises against this — expect degraded output and possible "
                "endless repetition. Pair --seed with a non-zero temperature instead.",
                self.model_name,
            )

        # num_predict is NOT adjusted here. The reasoning allowance is added by
        # TaskRunner._output_budget, before the context window is sized —
        # raising it at this point meant the generation budget could exceed the
        # window that had already been computed to hold it.

        self.model = self.initialize_model()

        self.predictor = Predictor(
            model=self.model,
            task_config=self.task_config,
            examples_path=self.example_dir,
            num_examples=self.num_examples,
            task_dir=self.task_dir,
            ollama_host=self.ollama_host,
        )

    def initialize_model(self) -> ChatOllama:
        self._check_budget()
        return ChatOllama(
            model=self.model_name,
            base_url=self.ollama_host,
            temperature=self.temperature,
            num_predict=self.num_predict,
            num_ctx=self.max_context_len,
            verbose=self.verbose,
            seed=self.seed,
            top_k=self.top_k,
            top_p=self.top_p,
            reasoning=self._reasoning_param,
        )

    def _check_budget(self) -> None:
        """Refuse a generation budget that does not fit in the context window.

        ``num_ctx`` is the whole window: the prompt and the generated answer
        share it. When ``num_predict`` is the larger of the two there is no room
        left for any input at all, and Ollama does not complain — it truncates
        the prompt silently and returns worse output with no indication why
        (ollama/ollama#14259). Failing here is the difference between a
        mystifying result and a fixable message.

        Checked at this boundary rather than only where the window is sized, so
        that anything reaching the model by another route is still caught.

        ``max_context_len`` is skipped when it is not an integer: it carries the
        mode strings ``"max"``/``"split"`` until ``TaskRunner`` resolves them,
        and there is nothing to compare against until it has.
        """
        num_ctx = self.max_context_len
        if not isinstance(num_ctx, int) or isinstance(num_ctx, bool):
            return
        if self.num_predict < num_ctx:
            return

        if self._is_thinking and not self.reasoning_model:
            hint = (
                "This model was auto-detected as a thinking model, so its "
                "reasoning allowance was added to num_predict after the window "
                "had already been sized. Pass --reasoning_model to size the "
                "window with the allowance included, or --no_reasoning to run "
                "without it."
            )
        else:
            hint = (
                "Raise --max_context_len (or --max_context_cap), or lower "
                "--num_predict, so that the window can hold the prompt as well "
                "as the answer."
            )

        raise ContextBudgetError(
            f"num_predict ({self.num_predict}) must be smaller than num_ctx "
            f"({num_ctx}): the context window has to hold the prompt as well as "
            f"the generated answer, and this leaves "
            f"{max(num_ctx - self.num_predict, 0)} tokens for it. {hint}"
        )

    def _translate_task(self) -> None:
        self.translation_path = Path(self.translation_dir) / f"{self.task_id}.json"

        if self.translation_path.exists() and not self.overwrite:
            logger.info("Translation file already exists. Skipping translation.")
            return

        logger.info("Translating Task %s", self.task_id)
        self.translation_path.parent.mkdir(parents=True, exist_ok=True)

        translator = Translator(model=self.model, input_field=self.input_field)
        translator.translate(self.test, self.translation_path)

        with self.translation_path.open("r") as f:
            self.test = pd.read_json(f)

    def run(self) -> List[Path]:
        if self.translate:
            self._translate_task()

        # Validate and prepare examples
        examples = None
        if self.train is not None:
            if isinstance(self.train, list) and len(self.train) > 0:
                examples = self.train
            elif hasattr(self.train, '__len__') and len(self.train) > 0:
                examples = self.train
            else:
                logger.warning("Training data is empty. Using zero-shot prompting.")

        self.predictor.prepare_prompt_ollama(
            embedding_model=self.embedding_model,
            examples=examples,
        )
        self._calibrate_prompt_estimate()

        outpath_list = []
        for run_idx in range(self.n_runs):
            outpath_list.append(self._run_single_prediction(run_idx))

        return outpath_list

    def _longest_input(self) -> Optional[str]:
        """The input this run expects to build its largest prompt from."""
        if self.test is None or not len(self.test):
            return None
        column = self.input_field
        if column not in self.test:
            return None
        if "token_count" in self.test:
            return str(self.test.loc[self.test["token_count"].idxmax(), column])
        return max((str(value) for value in self.test[column]), key=len, default=None)

    def _calibrate_prompt_estimate(self) -> None:
        """Measure the longest real prompt and check the reservation covers it.

        The context estimate is built with tiktoken's ``cl100k_base`` against
        Qwen and Phi models, drifts further on non-English text, and never sees
        the chat template's own role markers at all — so it is an approximation
        with a percentage margin bolted on. This is the first and only point
        where that approximation can be compared with the truth: the model
        exists, the prompt is assembled, and no rows have been spent yet.

        An estimate that came out *over* is fine and merely wasteful. One that
        came out under means every long document silently loses its beginning,
        which is the failure this whole area exists to prevent, so it stops the
        run rather than letting it produce quietly degraded output.
        """
        text = self._longest_input()
        if text is None:
            return
        if not isinstance(self.max_context_len, int):
            return

        try:
            messages = self.predictor.assemble_prompt(text)
        except Exception:
            logger.debug("Could not assemble a prompt to measure.", exc_info=True)
            return

        measured = measure_prompt_tokens(
            self.model_name,
            messages,
            host=self.ollama_host,
            num_ctx=self.max_context_len,
        )
        if measured is None:
            logger.info(
                "Prompt estimate could not be verified against the model; "
                "continuing on the estimate alone."
            )
            return

        reserved = self.max_context_len - self.num_predict
        logger.info(
            "Prompt calibration: reserved %d tokens, the model counted %d "
            "(estimate is %.2fx the real count), %d to spare.",
            reserved,
            measured,
            reserved / measured,
            reserved - measured,
        )

        if measured <= reserved:
            return

        self._correct_window(measured, reserved)

    def _correct_window(self, measured: int, reserved: int) -> None:
        """Re-size the window from the measured prompt, or refuse if it cannot be.

        The estimate came out under. Refusing here would be safe but lazy: the
        measurement is *exact*, so the window can simply be re-resolved from it
        under the same ceilings that produced the original. That turns "your
        estimate was wrong, go and change a flag" into a window that is right.

        It cannot always be done. An explicit ``--max_context_len`` is the user's
        decision and is not overridden; a hardware cap or the model's own limit
        may leave no room to grow. Those cases still stop the run, naming what
        blocked it — that beats Ollama silently discarding the front of every
        long document.
        """
        shortfall = measured - reserved
        if self.budget is None:
            raise ContextBudgetError(
                f"The prompt does not fit the window reserved for it: the model "
                f"counts {measured} tokens where {reserved} were reserved, "
                f"{shortfall} over, and there is no budget to re-resolve from."
            )

        corrected = resolve_budget(
            prompt_tokens=measured,
            output_tokens=self.num_predict,
            requested_ctx=self.budget.requested_ctx,
            hardware_ceiling=self.max_context_cap,
            model_native_max=self._model_info.native_context,
        )

        if measured > corrected.prompt_room:
            blocker = corrected.limited_by or "--max_context_len"
            raise ContextBudgetError(
                f"The prompt does not fit the window reserved for it and the "
                f"window cannot grow: the model counts {measured} tokens, "
                f"{shortfall} more than the {reserved} reserved, and "
                f"{blocker} holds num_ctx at {corrected.num_ctx} which leaves "
                f"only {corrected.prompt_room}. The token estimate uses OpenAI's "
                f"vocabulary against a different model and cannot see the chat "
                f"template, so it drifts — further on non-English text. Raise "
                f"{blocker}, lower --num_predict, or use fewer --num_examples."
            )

        logger.warning(
            "The prompt estimate was %d tokens short of the real count; growing "
            "num_ctx from %d to %d and reloading the model. %s",
            shortfall,
            self.max_context_len,
            corrected.num_ctx,
            corrected.describe(),
        )
        self.max_context_len = corrected.num_ctx
        self.model = self.initialize_model()
        # The predictor builds its chain from ``self.model`` at call time, so
        # replacing the reference is enough — but it has to be replaced, or the
        # run continues against the model built with the too-small window.
        self.predictor.model = self.model

    def _run_single_prediction(self, run_idx: int) -> Path:
        output_path = self.output_path_base / run_folder_name(
            self.task_name, run_idx, self.test_run_size
        )
        output_path.mkdir(parents=True, exist_ok=True)

        prediction_file = output_path / "nlp-predictions-dataset.json"
        if prediction_file.exists() and not self.overwrite:
            logger.warning("Prediction %d already exists. Skipping.", run_idx + 1)
            return prediction_file

        if self.chunk_size is not None:
            logger.info("Processing in chunks of size %d.", self.chunk_size)
            for chunk_idx in range(0, len(self.test), self.chunk_size):
                chunk_output_path = (
                    output_path / f"nlp-predictions-dataset-{chunk_idx}.json"
                )
                if chunk_output_path.exists() and not self.overwrite:
                    logger.warning("Chunk %d already exists. Skipping.", chunk_idx)
                    continue

                samples = self.test.iloc[chunk_idx : chunk_idx + self.chunk_size]
                chunk_results = self.predictor.predict(samples)
                chunk_predictions = [
                    {**sample._asdict(), **result}
                    for sample, result in zip(
                        samples.itertuples(index=False), chunk_results
                    )
                ]
                save_json(
                    chunk_predictions,
                    outpath=output_path,
                    filename=chunk_output_path.name,
                )

            logger.info("Merging chunked predictions.")
            found_chunks = chunk_files(output_path)
            chunk_predictions = []
            for chunk_file in found_chunks:
                with chunk_file.open("r") as f:
                    chunk_predictions.extend(json.load(f))

            filename = (
                f"nlp-predictions-dataset-{self.data_split}.json"
                if self.data_split
                else "nlp-predictions-dataset.json"
            )
            save_json(chunk_predictions, outpath=output_path, filename=filename)
            self._write_failures(chunk_predictions, output_path)
            for chunk_file in found_chunks:
                chunk_file.unlink()

            return output_path / filename
        else:
            logger.info("Running predictions without chunking.")
            results = self.predictor.predict(self.test)
            predictions = [
                {**sample._asdict(), **result}
                for sample, result in zip(self.test.itertuples(index=False), results)
            ]
            filename = (
                f"nlp-predictions-dataset-{self.data_split}.json"
                if self.data_split
                else "nlp-predictions-dataset.json"
            )
            save_json(predictions, outpath=output_path, filename=filename)
            self._write_failures(predictions, output_path)

            return output_path / filename

    def _write_failures(self, rows: List[Dict], output_path: Path) -> None:
        """Write this phase's failures beside its predictions.

        The work lives in ``utils`` because ``split`` writes one file per phase
        and then has to replace them with a single merged one, so the runner
        needs the same routine.
        """
        write_failures(
            rows=rows,
            input_field=self.input_field,
            outpath=output_path,
            data_split=self.data_split,
        )
