import argparse
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from langchain.globals import set_debug

from llm_extractinator.data_loader import DataLoader, TaskLoader
from llm_extractinator.ollama_server import OllamaServerManager
from llm_extractinator.prediction_task import PredictionTask
from llm_extractinator.utils import save_json


class NoHttpRequestsFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request:" not in record.getMessage()


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "task_runner.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@dataclass
class TaskConfig:
    # General Task Settings
    task_id: int = 0
    run_name: str = "run"
    n_runs: int = 5
    num_examples: int = 0
    num_predict: int = 512
    chunk_size: Optional[int] = None
    overwrite: bool = False
    translate: bool = False
    verbose: bool = False
    reasoning_model: bool = False

    # Model Configuration
    model_name: str = "mistral-nemo"
    temperature: float = 0.0
    max_context_len: Union[int, str] = "max"
    quantile: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None

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

        if self.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )

        if isinstance(self.max_context_len, int) and self.max_context_len <= 0:
            raise ValueError(
                f"max_context_len must be positive, got {self.max_context_len}"
            )

        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")

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
        setup_logging(self.log_dir)


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

        if self.config.translate or self.config.reasoning_model:
            self.config.num_predict = self.data_loader.adapt_num_predict(
                df=self.test,
                token_column="token_count",
                translate=self.config.translate,
                reasoning_model=self.config.reasoning_model,
                num_predict=self.config.num_predict,
            )

        if self.config.max_context_len == "split":
            self._split_data()

            self._set_max_context_len("short", self.short_test)
            logging.info(
                f"Running short cases with max_context_len: {self.config.max_context_len}"
            )
            self.short_paths = self._run_with_manager()

            self._set_max_context_len("long", self.long_test)
            logging.info(
                f"Running long cases with max_context_len: {self.config.max_context_len}"
            )
            self.long_paths = self._run_with_manager()

            self._combine_results()
        else:
            dataset = "test" if self.config.max_context_len == "max" else "train"
            self.config.max_context_len = self.data_loader.get_max_input_tokens(
                df=self.test if dataset == "test" else self.train,
                num_predict=self.config.num_predict,
                num_examples=self.config.num_examples,
            )
            logging.info(
                f"Running cases with max_context_len: {self.config.max_context_len}"
            )
            self._run_with_manager()

        total_time = timedelta(seconds=time.time() - start_time)
        logging.info(f"Task execution completed in {total_time}")

    def _run_with_manager(self) -> None:
        """Runs the task with a managed Ollama server."""
        with OllamaServerManager(log_dir=self.config.log_dir) as manager:
            manager.pull_model(self.config.model_name)
            result = self._run_task()
            manager.stop(self.config.model_name)
        return result

    def _run_task(self) -> bool:
        """Executes a single prediction task in parallel with error handling."""
        try:
            task = PredictionTask(**asdict(self.config))
            return task.run()
        except Exception as error:
            logging.error(f"Task execution failed: {error}", exc_info=True)
            raise

    def _extract_task_info(self) -> None:
        """Extract task information from the task configuration file."""
        task_loader = TaskLoader(
            folder_path=self.config.task_dir, task_id=self.config.task_id
        )
        self.config.task_config = task_loader.find_and_load_task()
        self.example_file = self.config.task_config.get("Example_Path")
        self.config.train_path = (
            self.config.example_dir / self.example_file if self.example_file else None
        )
        self.config.test_path = self.config.data_dir / self.config.task_config.get(
            "Data_Path"
        )
        self.config.input_field = self.config.task_config.get("Input_Field")
        self.config.task_name = task_loader.get_task_name()

    def _load_data(self) -> None:
        """Load the training and testing data for the prediction task."""
        self.data_loader = DataLoader(
            train_path=self.config.train_path, test_path=self.config.test_path
        )
        self.train, self.test = self.data_loader.load_data(
            text_column=self.config.input_field
        )

    def _split_data(self) -> None:
        """Split the data into short and long examples based on token count."""
        split_train = (
            self.data_loader.split_data
            if self.train is not None
            else lambda *args, **kwargs: (None, None)
        )

        self.short_train, self.long_train = split_train(
            df=self.train,
            text_column=self.config.input_field,
            quantile=self.config.quantile,
        )
        self.short_test, self.long_test = self.data_loader.split_data(
            df=self.test,
            text_column=self.config.input_field,
            quantile=self.config.quantile,
        )

    def _set_max_context_len(self, data_split: str, df: pd.DataFrame) -> None:
        """Set max_context_len based on the dataset."""
        self.config.max_context_len = self.data_loader.get_max_input_tokens(
            df=df,
            num_predict=self.config.num_predict,
            num_examples=self.config.num_examples,
        )
        self.config.data_split = data_split
        self.config.train = getattr(self, f"{data_split}_train")
        self.config.test = getattr(self, f"{data_split}_test")

    def _combine_results(self) -> None:
        """Combine short and long case results into a single JSON dataset."""
        try:
            logging.info("Combining results from short and long cases.")

            if not (self.short_paths and self.long_paths):
                raise ValueError("Missing results from either short or long cases.")

            for short_path, long_path in zip(self.short_paths, self.long_paths):
                short_df = pd.read_json(short_path, orient="records")
                long_df = pd.read_json(long_path, orient="records")
                combined_df = pd.concat([short_df, long_df], ignore_index=True)

                save_json(
                    combined_df.to_dict(orient="records"),
                    outpath=short_path.parent,
                    filename="nlp-predictions-dataset.json",
                )

                short_path.unlink(missing_ok=True)
                long_path.unlink(missing_ok=True)

        except Exception as error:
            logging.error(f"Error combining results: {error}", exc_info=True)


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
        default=5,
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
        default=512,
        help="Maxinum number of tokens to generate for a prediction.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Size of data chunks for processing; None means full dataset at once.",
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
        help="If set, enables a reasoning-based model for task execution (disables JSON mode).",
    )

    # Model Configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-nemo",
        help="Name of the model to use. Follows Ollama naming scheme.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; higher values lead to more randomness in outputs.",
    )
    parser.add_argument(
        "--max_context_len",
        type=str,
        default="split",
        help="Maximum context length; 'split' splits data into short and long cases and does a run for them seperately (good if your dataset distribution has a tail with long reports and a bulk of short ones), 'max' uses the maximum token length of the dataset, or a number sets a fixed length.",
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
    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
    task_runner = TaskRunner(config)
    try:
        task_runner.run_tasks()
    except Exception as error:
        logging.error(f"Error running tasks: {error}")
        traceback.print_exc()


def main():
    config = parse_args()
    extractinate(**asdict(config))


if __name__ == "__main__":
    main()
