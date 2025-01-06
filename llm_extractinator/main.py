import argparse
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import List

from langchain.globals import set_debug

from llm_extractinator.ollama_server import OllamaServerManager
from llm_extractinator.prediction_task import PredictionTask


class TaskRunner:
    """
    Handles prediction task execution with multiprocessing support.
    """

    def __init__(
        self,
        /,
        model_name: str,
        task_id: int,
        num_examples: int,
        n_runs: int,
        temperature: float,
        max_context_len: int,
        run_name: str,
        output_dir: Path,
        task_dir: Path,
        log_dir: Path,
        num_predict: int,
        translate: bool,
        verbose: bool = False,
        data_dir: Path = Path(__file__).resolve().parents[1] / "data",
        example_dir: Path = Path(__file__).resolve().parents[1] / "examples",
        chunk_size: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.task_id = f"{int(task_id):03}"
        self.num_examples = num_examples
        self.n_runs = n_runs
        self.temperature = temperature
        self.max_context_len = max_context_len
        self.run_name = run_name
        self.output_dir = output_dir
        self.output_path_base = self.output_dir / run_name
        self.task_dir = task_dir
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.example_dir = example_dir
        self.num_predict = num_predict
        self.chunk_size = chunk_size
        self.translate = translate
        self.verbose = verbose

    def run_tasks(self) -> None:
        """
        Runs the prediction tasks using multiprocessing.
        """
        start_time = time.time()

        set_debug(self.verbose)

        # Start the Ollama Server
        with OllamaServerManager(model_name=self.model_name, log_dir=self.log_dir):
            self._run_task()

        total_time = timedelta(seconds=time.time() - start_time)
        print(f"Total time taken for generating predictions: {total_time}")

    def _run_task(self) -> bool:
        """
        Executes a single prediction task in parallel.
        """
        try:
            task = PredictionTask(
                task_id=self.task_id,
                model_name=self.model_name,
                output_path_base=self.output_path_base,
                num_examples=self.num_examples,
                n_runs=self.n_runs,
                temperature=self.temperature,
                max_context_len=self.max_context_len,
                task_dir=self.task_dir,
                num_predict=self.num_predict,
                data_dir=self.data_dir,
                example_dir=self.example_dir,
                chunk_size=self.chunk_size,
                translate=self.translate,
            )
            task.run()
            return True
        except Exception as error:
            import traceback

            traceback.print_exc()
            return False


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run prediction tasks for a given model."
    )
    parser.add_argument(
        "--task_id", type=int, required=True, help="Task ID to generate examples for."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-nemo",
        help="Name of the model for prediction tasks.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples to generate for each task.",
    )
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs.")
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Temperature for generation."
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=8192,
        help="Maximum context length.",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=1024,
        help="Maximum number of tokens to predict.",
    )
    parser.add_argument("--run_name", type=Path, default="run", help="Name of the run.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=f"{Path(__file__).resolve().parents[1]}/output",
        help="Path for output files.",
    )
    parser.add_argument(
        "--task_dir",
        type=Path,
        default=f"{Path(__file__).resolve().parents[1]}/tasks",
        help="Path for task files.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=f"{Path(__file__).resolve().parents[1]}/output",
        help="Path to the directory for the log file for the server.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=f"{Path(__file__).resolve().parents[1]}/data",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--example_dir",
        type=Path,
        default=f"{Path(__file__).resolve().parents[1]}/examples",
        help="Path to the directory for the generated examples.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of examples to generate in a single chunk.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate the generated examples to English.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args()


def extractinate(
    model_name="mistral-nemo",
    task_id=0,
    num_examples=0,
    n_runs=5,
    temperature=0.3,
    max_context_len=8192,
    run_name="run",
    num_predict=1024,
    output_dir=None,
    task_dir=None,
    log_dir=None,
    data_dir=None,
    example_dir=None,
    chunk_size=None,
    translate=False,
    verbose=False,
    **kwargs,
) -> None:
    """
    Main function to initialize and run task execution and evaluation.
    Allows for flexible usage by providing default values for most arguments.
    """

    cwd = Path(os.getcwd())  # Use the current working directory

    output_dir = Path(output_dir) if output_dir else cwd / "output"
    task_dir = Path(task_dir) if task_dir else cwd / "tasks"
    log_dir = Path(log_dir) if log_dir else output_dir / "output"
    data_dir = Path(data_dir) if data_dir else cwd / "data"
    example_dir = Path(example_dir) if example_dir else cwd / "examples"

    # Initialize TaskRunner with the provided or default values
    task_runner = TaskRunner(
        model_name=model_name,
        task_id=task_id,
        num_examples=num_examples,
        n_runs=n_runs,
        temperature=temperature,
        max_context_len=max_context_len,
        run_name=run_name,
        num_predict=num_predict,
        output_dir=output_dir,
        task_dir=task_dir,
        log_dir=log_dir,
        data_dir=data_dir,
        example_dir=example_dir,
        chunk_size=chunk_size,
        translate=translate,
        verbose=verbose,
        **kwargs,
    )

    task_runner.run_tasks()


def main():
    # When the script is executed directly, use command-line arguments
    args = parse_args()
    extractinate(
        model_name=args.model_name,
        task_id=args.task_id,
        num_examples=args.num_examples,
        n_runs=args.n_runs,
        temperature=args.temperature,
        max_context_len=args.max_context_len,
        run_name=args.run_name,
        num_predict=args.num_predict,
        output_dir=args.output_dir,
        task_dir=args.task_dir,
        log_dir=args.log_dir,
        data_dir=args.data_dir,
        example_dir=args.example_dir,
        chunk_size=args.chunk_size,
        translate=args.translate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
