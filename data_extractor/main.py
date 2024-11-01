import argparse
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from dragon_eval import DragonEval

from data_extractor.ollama_server import OllamaServerManager
from data_extractor.prediction_task import PredictionTask

DEBUG = True


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
        max_context_len: Optional[int],
        run_name: str,
        output_dir: Path,
        task_dir: Path,
        log_dir: Path,
        num_predict: int,
        data_dir: Path = Path(__file__).resolve().parents[1] / "data",
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
        self.num_predict = num_predict

    def run_tasks(self) -> None:
        """
        Runs the prediction tasks using multiprocessing.
        """
        start_time = time.time()

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
            )
            task.run()
            return True
        except Exception as error:
            if DEBUG:
                import traceback

                traceback.print_exc()
            else:
                print(f"Error in task {self.task_id}: {error}")
            return False


class PredictionEvaluator:
    """
    Evaluates the results of prediction tasks using DragonEval.
    """

    def __init__(
        self,
        /,
        task_ids: List[int],
        ground_truth_path: Path,
        output_path: Path,
        prediction_path: Path,
    ) -> None:
        self.task_ids = [f"{int(task_id):03}" for task_id in task_ids]
        self.ground_truth_path = ground_truth_path
        self.output_path = output_path
        self.prediction_path = prediction_path

    def evaluate(self) -> None:
        """
        Evaluates the prediction tasks.
        """
        try:
            DragonEval(
                ground_truth_path=self.ground_truth_path,
                predictions_path=self.prediction_path,
                output_file=self.output_path,
                tasks=self.task_ids,
            ).evaluate()
        except Exception as error:
            print(f"Evaluation error: {error}")


def parse_args_extract_data() -> argparse.Namespace:
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
        default=None,
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
    return parser.parse_args()


def parse_args_evaluate() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate prediction tasks.")
    parser.add_argument(
        "--task_ids", nargs="+", type=int, required=True, help="Task IDs to evaluate."
    )
    parser.add_argument(
        "--prediction_path", type=Path, required=True, help="Path to prediction data."
    )
    parser.add_argument(
        "--ground_truth_path",
        type=Path,
        required=True,
        help="Path to ground truth data.",
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path for output file."
    )
    return parser.parse_args()


def extract_data() -> None:
    """
    Main function to initialize and run task execution and evaluation.
    """
    args = parse_args_extract_data()

    task_runner = TaskRunner(
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
    )

    task_runner.run_tasks()


def evaluate() -> None:
    """
    Main function to evaluate the results of prediction tasks.
    """
    args = parse_args_evaluate()

    evaluator = PredictionEvaluator(
        task_ids=args.task_ids,
        ground_truth_path=args.ground_truth_path,
        output_path=args.output_path,
        prediction_path=args.prediction_path,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    extract_data()
