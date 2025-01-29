import argparse
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from langchain.globals import set_debug

from llm_extractinator.ollama_server import OllamaServerManager
from llm_extractinator.prediction_task import PredictionTask


@dataclass
class TaskConfig:
    model_name: str = "mistral-nemo"
    task_id: int = 0
    num_examples: int = 0
    n_runs: int = 5
    temperature: float = 0.3
    max_context_len: int = 8192
    run_name: str = "run"
    num_predict: int = 1024
    output_dir: Optional[Path] = None
    task_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    example_dir: Optional[Path] = None
    chunk_size: Optional[int] = None
    translate: bool = False
    verbose: bool = False
    overwrite: bool = False
    seed: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def resolve_paths(self) -> None:
        """Resolves default paths if not provided."""
        cwd = Path(os.getcwd())
        self.output_dir = self.output_dir or cwd / "output"
        self.task_dir = self.task_dir or cwd / "tasks"
        self.log_dir = self.log_dir or self.output_dir / "output"
        self.data_dir = self.data_dir or cwd / "data"
        self.example_dir = self.example_dir or cwd / "examples"


class TaskRunner:
    """Handles prediction task execution with multiprocessing support."""

    def __init__(self, config: TaskConfig) -> None:
        config.resolve_paths()
        self.config = config

    def run_tasks(self) -> None:
        """Runs the prediction tasks using multiprocessing."""
        start_time = time.time()

        set_debug(self.config.verbose)

        # Start the Ollama Server
        with OllamaServerManager(
            model_name=self.config.model_name, log_dir=self.config.log_dir
        ):
            self._run_task()

        total_time = timedelta(seconds=time.time() - start_time)
        print(f"Total time taken for generating predictions: {total_time}")

    def _run_task(self) -> bool:
        """Executes a single prediction task in parallel."""
        try:
            task = PredictionTask(**asdict(self.config))
            task.run()
            return True
        except Exception as error:
            import traceback

            traceback.print_exc()
            return False


def parse_args() -> TaskConfig:
    """Parses command-line arguments and returns a TaskConfig object."""
    parser = argparse.ArgumentParser(
        description="Run prediction tasks for a given model."
    )

    # Automatically map dataclass fields to argparse arguments
    for field in TaskConfig.__dataclass_fields__.values():
        arg_name = f"--{field.name.replace('_', '-')}"
        arg_type = (
            field.type if field.type is not Optional else str
        )  # Handle Optional types
        kwargs = {"type": arg_type, "default": field.default}

        if field.type is bool:
            kwargs.pop("type", None)  # Remove 'type' for boolean flags
            kwargs["action"] = "store_true"

        parser.add_argument(arg_name, **kwargs)

    args, unknown = parser.parse_known_args()
    return TaskConfig(**vars(args))


def extractinate(**kwargs) -> None:
    """Main function that accepts keyword arguments and runs task execution."""
    config = TaskConfig(**kwargs)  # Convert kwargs to TaskConfig instance
    # Set seeds if provided
    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
    task_runner = TaskRunner(config)
    task_runner.run_tasks()


def main():
    config = parse_args()
    extractinate(config)


if __name__ == "__main__":
    main()
