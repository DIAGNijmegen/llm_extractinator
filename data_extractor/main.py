import time
from datetime import timedelta
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from data_extractor.prediction_task import PredictionTask
from data_extractor.ollama_server import OllamaServerManager
from dragon_eval import DragonEval

DEBUG = True

class TaskRunner:
    """
    Handles prediction task execution with multiprocessing support.
    """

    def __init__(self, data_path, example_path, model_name, task_id, num_examples, n_runs, temperature, run_name):
        self.data_path = data_path
        self.example_path = example_path
        self.model_name = model_name
        self.task_id = f"{int(task_id):03}"
        self.num_examples = num_examples
        self.n_runs = n_runs
        self.temperature = temperature
        self.run_name = run_name
        self.homepath = Path(__file__).resolve().parents[1]
        self.output_path_base = self.homepath / f"output/{model_name}/{run_name}"

    def run_tasks(self):
        """
        Runs the prediction tasks using multiprocessing.
        """
        start_time = time.time()

        # Start the Ollama Server
        with OllamaServerManager(self.model_name):
            # Execute tasks in parallel using multiprocessing
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(self._run_task, self.task_id))

        total_time = timedelta(seconds=time.time() - start_time)
        print(f"Total time taken: {total_time}")

    def _run_task(self, task_id):
        """
        Executes a single prediction task in parallel.
        """
        try:
            task = PredictionTask(
                task_id=task_id,
                model_name=self.model_name,
                data_path=self.data_path,
                example_path=self.example_path,
                output_path_base=self.output_path_base,
                num_examples=self.num_examples,
                n_runs=self.n_runs,
                temperature=self.temperature,
            )
            task.run()
            return True
        except Exception as error:
            if DEBUG:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error in task {task_id}: {error}")
            return False


class PredictionEvaluator:
    """
    Evaluates the results of prediction tasks using DragonEval.
    """

    def __init__(self, num_examples, task_ids, ground_truth_path, output_path_base):
        self.num_examples = num_examples
        self.task_ids = task_ids
        self.ground_truth_path = ground_truth_path
        self.output_path_base = output_path_base

    def evaluate(self):
        """
        Evaluates the prediction tasks.
        """
        predictions_path, output_file = self._get_evaluation_paths(self.num_examples)
        try:
            DragonEval(
                ground_truth_path=self.ground_truth_path,
                predictions_path=predictions_path,
                output_file=output_file,
                tasks=self.task_ids
            ).evaluate()
        except Exception as error:
            print(f"Evaluation error for {self.num_examples} examples: {error}")

    def _get_evaluation_paths(self, num_examples):
        """
        Generates paths for evaluation files.
        """
        predictions_path = self.output_path_base / f"{num_examples}_examples"
        output_file = predictions_path / f"metrics_{num_examples}_examples.json"
        return predictions_path, output_file


def parse_args_extract_data():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run prediction tasks for a given model.")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the data directory.")
    parser.add_argument("--example_path", type=Path, help="Path to data for example generation.")
    parser.add_argument("--task_id", type=int, required=True, help="Task ID to generate examples for.")
    parser.add_argument("--model_name", type=str, default="mistral-nemo", help="Name of the model for prediction tasks.")
    parser.add_argument("--num_examples", type=int, default=0, help="Number of examples to generate for each task.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation.")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run.")
    return parser.parse_args()

def parse_args_evaluate():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate prediction tasks.")
    parser.add_argument("--task_ids", nargs="+", type=int, required=True, help="Task IDs to evaluate.")
    parser.add_argument("--ground_truth_path", type=Path, required=True, help="Path to ground truth data.")
    parser.add_argument("--output_path_base", type=Path, required=True, help="Base path for output files.")
    return parser.parse_args()


def extract_data():
    """
    Main function to initialize and run task execution and evaluation.
    """
    args = parse_args_extract_data()

    if not args.example_path and args.num_examples > 0:
        raise ValueError("Example path is required for generating examples.")

    task_runner = TaskRunner(
        data_path=args.data_path,
        example_path=args.example_path,
        model_name=args.model_name,
        task_id=args.task_id,
        num_examples=args.num_examples,
        n_runs=args.n_runs,
        temperature=args.temperature,
        run_name=args.run_name
    )

    task_runner.run_tasks()
    
def evaluate():
    """
    Main function to evaluate the results of prediction tasks.
    """
    args = parse_args_evaluate()

    evaluator = PredictionEvaluator(
        num_examples=args.num_examples,
        task_ids=args.task_ids,
        ground_truth_path=args.ground_truth_path,
        output_path_base=args.output_path_base
    )
    evaluator.evaluate()


if __name__ == "__main__":
    extract_data()
