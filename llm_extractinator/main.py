import argparse
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    from langchain.globals import set_debug
except Exception:
    from langchain_core.globals import set_debug

from llm_extractinator.budget import (
    REASONING_ALLOWANCE,
    ContextBudgetError,
    resolve_budget,
)
from llm_extractinator.data_loader import DataLoader, TaskLoader
from llm_extractinator.ollama_server import OllamaServerManager, model_capabilities
from llm_extractinator.prediction_task import PredictionTask
from llm_extractinator.output_parsers import resolve_parser_model
from llm_extractinator.prompt_utils import (
    describe_fields,
    extra_instructions,
    prompt_scaffolding_text,
    task_description,
)
from llm_extractinator.run_config import (
    DEFAULT_NUM_PREDICT,
    resolve_thinking,
    suggested_num_predict,
)
from llm_extractinator.utils import save_json, write_failures


class NoHttpRequestsFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request:" not in record.getMessage()


class _TeeStream:
    """Mirrors writes to multiple streams so print() output reaches the log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False


def setup_logging(log_dir: Path, verbose: bool = False):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "task_runner.log"

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    if verbose:
        # LangChain's debug output goes via print() not logging — tee stdout to the log file
        sys.stdout = _TeeStream(sys.__stdout__, open(log_file, "a"))


@dataclass
class TaskConfig:
    # General Task Settings
    task_id: int = 0
    run_name: str = "run"
    n_runs: int = 1
    num_examples: int = 0
    num_predict: Optional[int] = None
    chunk_size: Optional[int] = None
    test_run_size: Optional[int] = None
    test_run_random: bool = False
    overwrite: bool = False
    translate: bool = False
    verbose: bool = False
    reasoning_model: bool = False
    no_reasoning: bool = False

    # Model Configuration
    model_name: str = "phi4"
    embedding_model: str = "nomic-embed-text"
    temperature: Optional[float] = None
    max_context_len: Union[int, str] = "max"
    max_context_cap: Optional[int] = None
    quantile: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    ollama_host: Optional[str] = None

    # File Paths
    output_dir: Optional[Path] = None
    task_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    example_dir: Optional[Path] = None
    translation_dir: Optional[Path] = None

    # Loaded from task file
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    input_field: Optional[str] = None
    task_name: Optional[str] = None
    task_config: Optional[dict] = None
    data_split: Optional[str] = None
    train: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Validation logic for the dataclass attributes."""
        if not (0 <= self.quantile <= 1):
            raise ValueError(f"quantile must be between 0 and 1, got {self.quantile}")

        if self.temperature is not None and self.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )

        if isinstance(self.max_context_len, int) and self.max_context_len <= 0:
            raise ValueError(
                f"max_context_len must be positive, got {self.max_context_len}"
            )

        if self.max_context_cap is not None and self.max_context_cap <= 0:
            raise ValueError(
                f"max_context_cap must be positive, got {self.max_context_cap}"
            )

        # num_ctx is the whole window: prompt and generation share it. A ceiling
        # at or below num_predict leaves nothing for the input, so the run would
        # silently truncate every document to nothing.
        if (
            self.max_context_cap is not None
            and self.num_predict is not None
            and self.max_context_cap <= self.num_predict
        ):
            raise ValueError(
                f"max_context_cap ({self.max_context_cap}) must exceed num_predict "
                f"({self.num_predict}) — the context window has to hold the prompt "
                "as well as the generated answer."
            )

        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")

        if self.test_run_size is not None and self.test_run_size <= 0:
            raise ValueError(
                f"test_run_size must be positive, got {self.test_run_size}"
            )

        self.resolve_paths()

    def resolve_paths(self) -> None:
        """Ensures all paths are Path objects and resolves defaults if not provided."""
        cwd = Path(os.getcwd())
        self.output_dir = Path(self.output_dir) if self.output_dir else cwd / "output"
        self.task_dir = Path(self.task_dir) if self.task_dir else cwd / "tasks"
        self.log_dir = Path(self.log_dir) if self.log_dir else self.output_dir / "logs"
        self.data_dir = Path(self.data_dir) if self.data_dir else cwd / "data"
        self.example_dir = (
            Path(self.example_dir) if self.example_dir else cwd / "examples"
        )
        self.translation_dir = (
            Path(self.translation_dir) if self.translation_dir else cwd / "translations"
        )
        setup_logging(self.log_dir, verbose=self.verbose)


#: Temporary column recording each row's position in the input, so that
#: ``split`` can put its two halves back in the order they arrived. Dropped
#: again before anything is written.
#:
#: The name must not start with an underscore: rows reach the output via
#: ``DataFrame.itertuples``, which renames any field a namedtuple cannot hold —
#: an underscored column silently arrives as a positional ``_5`` instead, so the
#: sort below finds nothing and the reordering quietly does not happen. It is
#: verbose for the same reason ``token_count`` is not: it has to not collide
#: with a column somebody actually has.
_SOURCE_ORDER_COLUMN = "extractinator_source_row"


class TaskRunner:
    """Handles prediction task execution with multiprocessing support."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    def run_tasks(self) -> None:
        """Runs the prediction tasks using a managed Ollama server instance."""
        start_time = time.time()
        set_debug(self.config.verbose)

        self._extract_task_info()
        self._load_data()

        # The model has to exist before it can be asked anything about itself,
        # and what it answers decides the generation budget — which in turn
        # decides the window. So the server comes up first and stays up for the
        # whole run, instead of being started and stopped around each phase.
        # Sizing used to happen above this point, before any of it was knowable.
        manager = OllamaServerManager(self.config.log_dir, host=self.config.ollama_host)
        manager.start_server()
        try:
            manager.pull_model(self.config.model_name)
            self._model_info = model_capabilities(
                self.config.model_name, host=self.config.ollama_host
            )
            self._is_thinking = resolve_thinking(
                detected=self._model_info.supports_thinking,
                reasoning_model=self.config.reasoning_model,
                no_reasoning=self.config.no_reasoning,
            )
            self._output_tokens = self._output_budget()
            self._run_phases()
        finally:
            manager.stop(self.config.model_name)


        total_time = timedelta(seconds=time.time() - start_time)
        logging.info(f"Task execution completed in {total_time}")

    def _scaffolding_tokens(self) -> int:
        """Tokens the fixed prompt framing costs, counted rather than guessed.

        Includes the task description, which is substituted into the system
        prompt and can run to a few hundred tokens on a detailed task.
        """
        try:
            task_config = self.config.task_config or {}
            model = resolve_parser_model(task_config, self.config.task_dir)
            return self.data_loader.count_tokens(
                prompt_scaffolding_text(
                    description=task_description(task_config),
                    fields=describe_fields(model),
                    extra_instructions=extra_instructions(task_config),
                )
            )
        except Exception:
            logging.exception("Could not measure prompt scaffolding; assuming 0")
            return 0

    def _output_budget(self) -> int:
        """How many tokens generation needs, decided before the window is sized.

        The ordering is the whole point. This used to be settled in two places —
        here for an explicit ``--reasoning_model``, and again inside
        ``PredictionTask.__init__`` for an auto-detected thinking model — and the
        second of those ran *after* the window had been sized, so the answer
        could be handed a budget larger than the window meant to contain it. The
        model has now been inspected, so one expression covers both.
        """
        num_predict = self.config.num_predict
        if num_predict is None:
            num_predict = self._num_predict_from_schema()
        if self.config.translate:
            num_predict = self.data_loader.adapt_num_predict(
                df=self._full_test,
                token_column="token_count",
                translate=True,
                reasoning_model=False,
                num_predict=num_predict,
            )
        if self._is_thinking:
            logging.info(
                "Reasoning is on for '%s'; reserving %d extra tokens for the "
                "chain of thought before sizing the window.",
                self.config.model_name,
                REASONING_ALLOWANCE,
            )
            num_predict += REASONING_ALLOWANCE
        return num_predict

    def _num_predict_from_schema(self) -> int:
        """Size the answer from the schema when the user did not say.

        Falls back to the flat default if the schema cannot be read: this is a
        convenience, and failing the run over it would be worse than the number
        it replaces.
        """
        try:
            model = resolve_parser_model(
                self.config.task_config or {}, self.config.task_dir
            )
        except Exception:
            logging.exception(
                "Could not read the schema to size the output budget; "
                "falling back to %d tokens",
                DEFAULT_NUM_PREDICT,
            )
            return DEFAULT_NUM_PREDICT

        chosen = suggested_num_predict(parser_model=model)
        logging.info(
            "--num_predict not set; sized from the schema to %d tokens.", chosen
        )
        return chosen

    def _run_phases(self) -> None:
        """Dispatch on the context mode, then run each phase with its own budget."""
        requested = self.config.max_context_len

        if self.config.test_run_size is not None and requested == "split":
            logging.warning(
                "max_context_len='split' isn't meaningful for a test run; "
                "using 'max' instead."
            )
            requested = "max"

        if requested == "split":
            self._split_data()
            self.short_path = self._run_phase(
                test=self.short_test,
                train=self.short_train,
                sizing_df=self.short_test,
                data_split="short",
            )
            self.long_path = self._run_phase(
                test=self.long_test,
                train=self.long_train,
                sizing_df=self.long_test,
                data_split="long",
            )
            self._combine_results()
        elif requested == "max":
            self._run_phase(test=self.test, train=self.train, sizing_df=self._full_test)
        else:
            self._run_phase(
                test=self.test,
                train=self.train,
                sizing_df=self._full_test,
                requested_ctx=requested,
            )

    def _run_phase(
        self,
        *,
        test,
        train,
        sizing_df: pd.DataFrame,
        data_split: Optional[str] = None,
        requested_ctx: Optional[int] = None,
    ):
        """Resolve this phase's budget, then run it.

        ``sizing_df`` is deliberately separate from ``test``: a test run
        processes a subset but has to be sized from the whole dataset, or a
        three-row check passes with a window the real run never uses.
        """
        prompt_tokens = self.data_loader.estimate_prompt_tokens(
            df=sizing_df,
            num_examples=self.config.num_examples,
            scaffolding_tokens=self._scaffolding_tokens(),
        )
        budget = resolve_budget(
            prompt_tokens=prompt_tokens,
            output_tokens=self._output_tokens,
            requested_ctx=requested_ctx,
            hardware_ceiling=self.config.max_context_cap,
            model_native_max=self._model_info.native_context,
        )
        self._report_budget(budget, sizing_df, prompt_tokens, data_split)

        # Built explicitly rather than by mutating the config, so that the mode
        # ("max"/"split") survives into the next phase and one phase's resolved
        # numbers cannot leak into the next phase's resolution. The shallow copy
        # also avoids asdict()'s deepcopy of the dataframes.
        params = {
            field.name: getattr(self.config, field.name)
            for field in fields(self.config)
        }
        params.update(
            max_context_len=budget.num_ctx,
            num_predict=budget.num_predict,
            data_split=data_split,
            train=train,
            test=test,
            model_info=self._model_info,
            budget=budget,
        )
        return self._run_task(params)

    def _report_budget(
        self,
        budget,
        sizing_df: pd.DataFrame,
        prompt_tokens: int,
        data_split: Optional[str],
    ) -> None:
        """Say where the window went, and what a capped one actually costs.

        "the longest inputs will be truncated" is true but unusable — it does
        not say whether that is three documents or three thousand, which is the
        difference between an acceptable trade and a ruined run.
        """
        where = f" ({data_split} cases)" if data_split else ""
        logging.info("Budget%s: %s", where, budget.describe())
        if not budget.is_clamped:
            return

        longest = int(sizing_df["token_count"].max()) if len(sizing_df) else 0
        overhead = prompt_tokens - longest  # scaffolding + examples + margin
        allowed = budget.prompt_room - overhead
        too_long = (
            int((sizing_df["token_count"] > allowed).sum())
            if allowed > 0
            else len(sizing_df)
        )
        logging.warning(
            "Context window capped at %d by %s, %d short of the %d this data "
            "wants. %d of %d document(s) exceed the %d tokens that leaves for "
            "input and will be truncated.",
            budget.num_ctx,
            budget.limited_by,
            budget.fitted_ctx - budget.num_ctx,
            budget.fitted_ctx,
            too_long,
            len(sizing_df),
            max(allowed, 0),
        )


    def _run_task(self, params: dict) -> bool:
        """Executes a single prediction task in parallel."""
        try:
            task = PredictionTask(**params)
            return task.run()
        except ContextBudgetError:
            # A misconfigured budget is fatal for every row, not just this one.
            # Swallowing it here would let the run report completion having
            # produced nothing.
            raise
        except Exception:
            logging.exception("Error running task")
            return False

    def _extract_task_info(self) -> None:
        """
        Extract task information from the task configuration file.
        """
        task_loader = TaskLoader(
            folder_path=self.config.task_dir, task_id=self.config.task_id
        )
        self.config.task_config = task_loader.find_and_load_task()
        self.example_file = self.config.task_config.get("Example_Path")
        if self.example_file is not None:
            self.config.train_path = (
                self.config.example_dir / self.config.task_config.get("Example_Path")
            )
        else:
            self.config.train_path = None
        self.config.test_path = self.config.data_dir / self.config.task_config.get(
            "Data_Path"
        )
        self.config.input_field = self.config.task_config.get("Input_Field")
        self.config.task_name = task_loader.get_task_name()

    def _load_data(self) -> None:
        """
        Load the training and testing data for the prediction task.
        """
        if self.config.num_examples == 0:
            self.config.train_path = None
        self.data_loader = DataLoader(
            examples_path=self.config.train_path, cases_path=self.config.test_path
        )
        if self.config.num_examples > 0:
            self.train = self.data_loader.load_examples()
        else:
            self.train = None
        self.test = self.data_loader.load_cases(text_column=self.config.input_field)
        # Kept so context sizing can use the whole dataset even when the run
        # itself only processes a subset.
        self._full_test = self.test

        if self.config.test_run_size is not None:
            if self.config.test_run_size >= len(self.test):
                logging.info(
                    "test_run_size >= dataset size; using the full dataset."
                )
            elif self.config.test_run_random:
                self.test = (
                    self.test.sample(
                        n=self.config.test_run_size, random_state=self.config.seed
                    )
                    .sort_index()
                    .reset_index(drop=True)
                )
            else:
                self.test = self.test.head(self.config.test_run_size).reset_index(
                    drop=True
                )

    def _split_data(self) -> None:
        """
        Split the data into short and long examples based on token count.
        """
        if self.train is not None:
            short_df, long_df = self.data_loader.split_data(
                df=self.data_loader.examples_df,
                quantile=self.config.quantile,
            )
            self.short_train = short_df[["input", "output"]].dropna().to_dict(orient="records")
            self.long_train = long_df[["input", "output"]].dropna().to_dict(orient="records")
        else:
            self.short_train, self.long_train = None, None
        # Tag each row with where it came from before splitting: the two halves
        # are processed separately and concatenated, so without this the output
        # comes back ordered by document length rather than by input position,
        # which silently breaks a positional join onto the source data.
        self.test = self.test.assign(
            **{_SOURCE_ORDER_COLUMN: range(len(self.test))}
        )
        self.short_test, self.long_test = self.data_loader.split_data(
            df=self.test,
            quantile=self.config.quantile,
        )

    def _combine_results(self) -> None:
        """
        Combine the results from short and long examples.
        """
        try:
            logging.info("Combining results from short and long cases.")
            if not self.short_path:
                raise ValueError(
                    "No paths found for short cases. Something went wrong."
                )
            if not self.long_path:
                raise ValueError("No paths found for long cases. Something went wrong.")

            for short_path, long_path in zip(self.short_path, self.long_path):
                short_df = pd.read_json(short_path, orient="records")
                long_df = pd.read_json(long_path, orient="records")
                combined_df = pd.concat([short_df, long_df], ignore_index=True)

                if _SOURCE_ORDER_COLUMN in combined_df:
                    combined_df = (
                        combined_df.sort_values(_SOURCE_ORDER_COLUMN)
                        .drop(columns=[_SOURCE_ORDER_COLUMN])
                        .reset_index(drop=True)
                    )

                rows = combined_df.to_dict(orient="records")
                save_json(
                    rows,
                    outpath=short_path.parent,
                    filename="nlp-predictions-dataset.json",
                )
                self._combine_failures(rows, short_path.parent)

            # Remove the individual files
            for short_path, long_path in zip(self.short_path, self.long_path):
                short_path.unlink()
                long_path.unlink()

        except Exception:
            logging.exception("Error combining results")

    def _combine_failures(self, rows: List[dict], output_path: Path) -> None:
        """Replace the per-phase failure files with one covering the merged run.

        The predictions are merged and the per-phase files deleted, so leaving
        ``failures-short.json`` and ``failures-long.json`` behind would mean two
        artefacts whose row numbers refer to files that no longer exist.
        """
        for data_split in ("short", "long"):
            (output_path / f"failures-{data_split}.json").unlink(missing_ok=True)
        write_failures(
            rows=rows, input_field=self.config.input_field, outpath=output_path
        )


def parse_args() -> TaskConfig:
    """Parses command-line arguments and returns a TaskConfig object."""
    parser = argparse.ArgumentParser(
        description="Run prediction tasks for a given model."
    )

    # General Task Settings
    parser.add_argument(
        "--task_id", type=int, default=0, help="Unique identifier for the task."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="Name for the run, used as the folder name for the output files.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of times to repeat the task execution.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples to use in prompts. Must be supplied separately.",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=None,
        help="Maximum number of tokens to generate for a prediction. Left unset it "
        "is sized from the output schema — a schema of thirty free-text fields needs "
        "far more room than one of three enums, and a truncated answer loses the "
        "whole row. A reasoning allowance is added on top for a thinking model.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Size of data chunks for processing; None means full dataset at once.",
    )
    parser.add_argument(
        "--test_run_size",
        type=int,
        default=None,
        help="Run on only the first N rows of the test set (or a random N-row sample "
        "with --test_run_random) instead of the full dataset - a quick sanity check.",
    )
    parser.add_argument(
        "--test_run_random",
        action="store_true",
        help="With --test_run_size, sample randomly (reproducible via --seed) instead "
        "of taking the first N rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, existing files will be overwritten instead of skipped.",
    )
    parser.add_argument(
        "--translate", action="store_true", help="If set, enables translation mode."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints detailed logs and debugging information.",
    )
    parser.add_argument(
        "--reasoning_model",
        action="store_true",
        help="If set, enables a reasoning-based model for task execution (disables JSON mode). "
        "Thinking models are auto-detected, so this is only needed when the model is not yet "
        "pulled or the server cannot be reached.",
    )
    parser.add_argument(
        "--no_reasoning",
        action="store_true",
        help="Force reasoning mode off, overriding auto-detection and --reasoning_model. "
        "Use when a thinking model should answer directly.",
    )

    # Model Configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="phi4",
        help="Name of the model to use. Follows Ollama naming scheme.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="nomic-embed-text",
        help="Name of the embedding model to use for few-shot example selection. Follows Ollama naming scheme.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Left unset it is chosen automatically: 0.0 (greedy) "
        "for an ordinary model, 0.6 for a thinking one, which Qwen's model cards require "
        "-- greedy decoding on a thinking model degrades it and can loop forever. Pass "
        "--temperature 0 to force greedy anyway.",
    )
    parser.add_argument(
        "--max_context_len",
        type=str,
        default="max",
        help="Maximum context length; 'split' splits data into short and long cases and does a run for them seperately (good if your dataset distribution has a tail with long reports and a bulk of short ones), 'max' uses the maximum token length of the dataset, or a number sets a fixed length.",
    )
    parser.add_argument(
        "--max_context_cap",
        type=int,
        default=None,
        help="Upper bound on the context window when --max_context_len is 'max' or "
        "'split'. Fits the window to the data as usual, but never requests more than "
        "this many tokens - use it to keep a run inside a GPU's memory.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="Quantile for splitting data into short and long cases based on token count. Only applicable when max_context_len is 'split'.",
    ),
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter; restricts sampling to the top-k most likely words.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter; restricts sampling to top cumulative probability mass.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=None,
        help=(
            "Base URL of an already-running Ollama server, e.g. 'http://localhost:11500' "
            "or 'http://remote-machine:11434'. If not set, llm_extractinator manages its own "
            "server on the default port (starting/stopping it and pulling models as needed). "
            "When set, the server is treated as externally managed: llm_extractinator will not "
            "start it, pull models onto it, or stop it — it only connects."
        ),
    )

    # File Paths
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where output files will be saved.",
    )
    parser.add_argument(
        "--task_dir",
        type=Path,
        default=None,
        help="Directory containing task files.",
    )
    parser.add_argument(
        "--log_dir", type=Path, default=None, help="Directory for logging output."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing input data files.",
    )
    parser.add_argument(
        "--example_dir",
        type=Path,
        default=None,
        help="Directory containing example files.",
    )
    parser.add_argument(
        "--translation_dir",
        type=Path,
        default=None,
        help="Directory containing translation files.",
    )

    args, unknown = parser.parse_known_args()

    # Convert max_context_len to int if it's a number, otherwise keep as string
    try:
        if (
            args.max_context_len.lower() != "split"
            and args.max_context_len.lower() != "max"
        ):
            args.max_context_len = int(args.max_context_len)
    except ValueError:
        logging.error(
            "max_context_len must be 'split', 'max', or an integer. Exiting..."
        )
        sys.exit(1)

    return TaskConfig(**vars(args))


def extractinate(**kwargs) -> None:
    """Main function that accepts keyword arguments and runs task execution."""
    config = TaskConfig(**kwargs)
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
    task_runner = TaskRunner(config)
    try:
        task_runner.run_tasks()
    except ContextBudgetError:
        logging.error("%s", sys.exc_info()[1])
        raise
    except Exception:
        logging.exception("Error running tasks")


def main():
    config = parse_args()
    extractinate(**asdict(config))


if __name__ == "__main__":
    main()
