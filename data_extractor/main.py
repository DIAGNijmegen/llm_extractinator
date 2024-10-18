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

    def __init__(self, datapath, model_name, task_ids, num_examples, n_runs, temperature, run_name):
        self.datapath = datapath
        self.model_name = model_name
        self.task_ids = [f"{int(task_id):03}" for task_id in task_ids]
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
                results = list(executor.map(self._run_task, self.task_ids))

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
                datapath=self.datapath,
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

    def __init__(self, num_examples, task_ids, datapath, ground_truth_path, output_path_base):
        self.num_examples = num_examples
        self.task_ids = task_ids
        self.datapath = datapath
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


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run prediction tasks for a given model.")
    parser.add_argument("--datapath", type=Path, required=True, help="Path to the data directory.")
    parser.add_argument("--example_path", type=Path, help="Path to data for example generation.")
    parser.add_argument("--ground_truth_path", type=Path, help="Path to ground truth data.")
    parser.add_argument("--task_ids", type=int, nargs="+", required=True, help="Task IDs to generate examples for.")
    parser.add_argument("--model_name", type=str, default="mistral-nemo", help="Name of the model for prediction tasks.")
    parser.add_argument("--num_examples", type=int, default=0, help="Number of examples to generate for each task.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation.")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation of the generated examples.")
    return parser.parse_args()


def main():
    """
    Main function to initialize and run task execution and evaluation.
    """
    args = parse_args()

    if not args.skip_evaluation and not args.ground_truth_path:
        raise ValueError("Ground truth data path is required for evaluation.")
    if not args.example_path and args.num_examples > 0:
        raise ValueError("Example path is required for generating examples.")

    task_runner = TaskRunner(
        datapath=args.datapath,
        model_name=args.model_name,
        task_ids=args.task_ids,
        num_examples=args.num_examples,
        n_runs=args.n_runs,
        temperature=args.temperature,
        run_name=args.run_name
    )

    task_runner.run_tasks()

    if not args.skip_evaluation:
        evaluator = PredictionEvaluator(
            num_examples=args.num_examples,
            task_ids=task_runner.task_ids,
            datapath=task_runner.datapath,
            ground_truth_path=args.ground_truth_path,
            output_path_base=task_runner.output_path_base
        )
        evaluator.evaluate()


if __name__ == "__main__":
    main()
