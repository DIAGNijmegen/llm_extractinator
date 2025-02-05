import argparse
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
    max_context_len: Union[int, str] = "split_max"
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

    # Server Configuration
    host: str = "localhost"
    port: int = 28900

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

        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")

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
        if self.config.max_context_len == "split_max":
            self._split_data()

            # Run short cases
            self.config.max_context_len = self.data_loader.get_max_input_tokens(
                df=self.short_test,
                num_predict=self.config.num_predict,
                num_examples=self.config.num_examples,
            )
            self.config.data_split = "short"
            self.config.train = self.short_train
            self.config.test = self.short_test

            print(
                "Running short cases with max_context_len:", self.config.max_context_len
            )

            with OllamaServerManager(
                host=self.config.host, port=self.config.port
            ) as manager:
                manager.pull_model(self.config.model_name)
                self.short_paths = self._run_task()
                manager.stop(self.config.model_name)

            # Run long cases
            self.config.max_context_len = self.data_loader.get_max_input_tokens(
                df=self.long_test,
                num_predict=self.config.num_predict,
                num_examples=self.config.num_examples,
            )
            self.config.data_split = "long"
            self.config.train = self.long_train
            self.config.test = self.long_test

            print(
                "Running long cases with max_context_len:", self.config.max_context_len
            )

            with OllamaServerManager(
                host=self.config.host, port=self.config.port
            ) as manager:
                manager.pull_model(self.config.model_name)
                self.long_paths = self._run_task()
                manager.stop(self.config.model_name)

            self._combine_results()
        elif self.config.max_context_len == "max_data":
            if "token_count" not in self.test.columns:
                self.test = self.data_loader.add_token_count(
                    self.test, self.config.input_field
                )
            self.config.max_context_len = self.data_loader.get_max_input_tokens(
                df=self.test,
                num_predict=self.config.num_predict,
                num_examples=self.config.num_examples,
            )
            self.config.train = self.train
            self.config.test = self.test

            print(
                "Running full cases with max_context_len:", self.config.max_context_len
            )

            with OllamaServerManager(
                host=self.config.host, port=self.config.port
            ) as manager:
                manager.pull_model(self.config.model_name)
                _ = self._run_task()
                manager.stop(self.config.model_name)
        else:
            self.config.train = self.train
            self.config.test = self.test
            with OllamaServerManager(
                host=self.config.host, port=self.config.port
            ) as manager:
                manager.pull_model(self.config.model_name)
                _ = self._run_task()
                manager.stop(self.config.model_name)

        total_time = timedelta(seconds=time.time() - start_time)
        print(f"Total time taken for generating predictions: {total_time}")

    def _run_task(self) -> bool:
        """Executes a single prediction task in parallel."""
        try:
            task = PredictionTask(**asdict(self.config))
            return task.run()
        except Exception as error:
            traceback.print_exc()
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
        self.data_loader = DataLoader(
            train_path=self.config.train_path, test_path=self.config.test_path
        )
        self.train, self.test = self.data_loader.load_data()

    def _split_data(self) -> None:
        """
        Split the data into short and long examples based on token count.
        """
        if self.train is not None:
            self.short_train, self.long_train = self.data_loader.split_data(
                df=self.train,
                text_column=self.config.input_field,
                quantile=self.config.quantile,
            )
        else:
            self.short_train, self.long_train = None, None
        self.short_test, self.long_test = self.data_loader.split_data(
            df=self.test,
            text_column=self.config.input_field,
            quantile=self.config.quantile,
        )

    def _combine_results(self) -> None:
        """
        Combine the results from short and long examples.
        """
        try:
            print("Combining results...")
            for short_path, long_path in zip(self.short_paths, self.long_paths):
                short_df = pd.read_json(short_path, orient="records")
                long_df = pd.read_json(long_path, orient="records")
                combined_df = pd.concat([short_df, long_df], ignore_index=True)

                save_json(
                    combined_df.to_dict(orient="records"),
                    outpath=short_path.parent,
                    filename="nlp-predictions-dataset.json",
                )

            # Remove the individual files
            for short_path, long_path in zip(self.short_paths, self.long_paths):
                short_path.unlink()
                long_path.unlink()

        except Exception as error:
            print(f"Error combining results: {error}")


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
        default="split_max",
        help="Maximum context length; 'auto' to determine automatically.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="Quantile for splitting data into short and long cases based on token count. Only applicable when max_context_len is 'auto'.",
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

    # Server Configuration
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for running the server.",
    )
    parser.add_argument(
        "--port", type=int, default=28900, help="Port number for the server."
    )

    args, unknown = parser.parse_known_args()

    # Convert max_context_len to int if it's a number, otherwise keep as string
    try:
        if (
            args.max_context_len.lower() != "split_max"
            or args.max_context_len.lower() != "max_data"
        ):
            args.max_context_len = int(args.max_context_len)
    except ValueError:
        print(f"ERROR: Invalid max_context_len value: {args.max_context_len}")
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
        print(f"Error running prediction tasks: {error}")
        traceback.print_exc()


def main():
    config = parse_args()
    extractinate(**asdict(config))


if __name__ == "__main__":
    main()
