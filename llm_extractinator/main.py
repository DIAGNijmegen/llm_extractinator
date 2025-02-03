import argparse
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union, get_origin

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
    temperature: float = 0.0
    max_context_len: Union[int, str] = "auto"
    run_name: str = "run"
    num_predict: int = 512
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
    reasoning_model: bool = False

    def resolve_paths(self) -> None:
        """Ensures all paths are Path objects and resolves defaults if not provided."""
        cwd = Path(os.getcwd())

        # Ensure all paths are Path objects
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
        if field.name == "max_context_len":  # Handle "auto" case
            parser.add_argument(arg_name, type=str, default=field.default)
        else:
            if get_origin(field.type) is Union:
                arg_type = next(
                    (t for t in field.type.__args__ if t is not type(None)), str
                )
            else:
                arg_type = field.type
            kwargs = {"type": arg_type, "default": field.default}

            if field.type is bool:
                kwargs = {"action": "store_true"}

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
    try:
        task_runner.run_tasks()
    except Exception as error:
        print(f"Error running prediction tasks: {error}")


def main():
    config = parse_args()
    extractinate(**asdict(config))


if __name__ == "__main__":
    main()
