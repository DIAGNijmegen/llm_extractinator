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
from langchain.globals import set_debug

from llm_extractinator.ollama_server import OllamaServerManager
from llm_extractinator.prediction_task import PredictionTask


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
    max_context_len: Union[int, str] = "auto"
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
        config.resolve_paths()
        self.config = config

    def run_tasks(self) -> None:
        """Runs the prediction tasks using a managed Ollama server instance."""
        start_time = time.time()
        set_debug(self.config.verbose)

        with OllamaServerManager(
            host=self.config.host, port=self.config.port
        ) as manager:
            manager.pull_model(self.config.model_name)
            self._run_task()
            manager.stop(self.config.model_name)

        total_time = timedelta(seconds=time.time() - start_time)
        print(f"Total time taken for generating predictions: {total_time}")

    def _run_task(self) -> bool:
        """Executes a single prediction task in parallel."""
        try:
            task = PredictionTask(**asdict(self.config))
            task.run()
            return True
        except Exception as error:
            traceback.print_exc()
            return False


def parse_args() -> TaskConfig:
    """Parses command-line arguments and returns a TaskConfig object."""
    parser = argparse.ArgumentParser(
        description="Run prediction tasks for a given model."
    )

    # General Task Settings
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--num_predict", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reasoning_model", action="store_true")

    # Model Configuration
    parser.add_argument("--model_name", type=str, default="mistral-nemo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_context_len", type=str, default="auto")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # File Paths
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--task_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--example_dir", type=Path, default=None)

    # Server Configuration
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=28900)

    args, unknown = parser.parse_known_args()

    # Convert max_context_len to int if it's a number, otherwise keep "auto"
    try:
        if args.max_context_len.lower() != "auto":
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


def main():
    config = parse_args()
    extractinate(**asdict(config))


if __name__ == "__main__":
    main()
