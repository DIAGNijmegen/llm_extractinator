import time
from datetime import timedelta
from itertools import product
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from data_extractor.prediction_task import PredictionTask
from data_extractor.ollama_server import OllamaServerManager
from dragon_eval import DragonEval

debug = True

class TaskRunner:
    """
    A class to handle prediction task execution with multiprocessing support.
    """

    def __init__(self, datapath, model_name, task_ids, num_examples, n_runs, temperatures, example_selectors, run_name):
        self.model_name = model_name
        self.task_ids = [f"{int(task_id):03}" for task_id in task_ids]
        self.num_examples = num_examples
        self.n_runs = n_runs
        self.temperatures = temperatures
        self.example_selectors = example_selectors
        self.run_name = run_name

        self.datapath = datapath
        self.homepath = Path(__file__).resolve().parents[1]
        self.output_path_base = self.homepath / f"output/{model_name}/{run_name}"

    def run_tasks(self):
        """
        Main method to run the prediction tasks using multiprocessing.
        """
        config_base = {
            "model_name": self.model_name,
            "datapath": self.datapath,
            "output_path_base": self.output_path_base,
            "n_runs": self.n_runs
        }

        start_time = time.time()

        # Start the Ollama Server
        with OllamaServerManager(self.model_name):
            configs = [
                {**config_base, **config}
                for config in self._generate_configs()
            ]

            # Execute tasks in parallel using multiprocessing
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(self._run_task, configs))

        total_time = timedelta(seconds=time.time() - start_time)
        print(f"Total time taken: {total_time}")

    def _generate_configs(self):
        """
        Generate configuration dictionaries for each combination of parameters.
        """
        for task_id, num_example, temperature, example_selector in product(
                self.task_ids, self.num_examples, self.temperatures, self.example_selectors):
            yield {
                "task_id": task_id,
                "num_examples": num_example,
                "temperature": temperature,
                "example_selector_name": example_selector,
            }

    @staticmethod
    def _run_task(config):
        """
        Static method to execute a single prediction task.
        This is designed to be run in parallel via multiprocessing.
        """
        try:
            task = PredictionTask(
                task_id=config["task_id"],
                model_name=config["model_name"],
                datapath=config["datapath"],
                output_path_base=config["output_path_base"],
                num_examples=config["num_examples"],
                n_runs=config["n_runs"],
                temperature=config["temperature"],
                example_selector_name=config["example_selector_name"]
            )
            task.run()
            return True
        except Exception as error:
            if debug:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error in task {config['task_id']}: {error}")
                return False


class PredictionEvaluator:
    """
    A class to handle evaluation of prediction results.
    """

    def __init__(self, num_examples, task_ids, datapath, ground_truth_path, output_path_base):
        self.num_examples = num_examples
        self.task_ids = task_ids
        self.datapath = datapath
        self.ground_truth_path = ground_truth_path
        self.output_path_base = output_path_base

    def evaluate(self):
        """
        Evaluate the prediction tasks using DragonEval.
        """
        for num in self.num_examples:
            predictions_path, output_file = self._get_evaluation_paths(num)
            
            try:
                DragonEval(
                    ground_truth_path=self.ground_truth_path,
                    predictions_path=predictions_path,
                    output_file=output_file,
                    tasks=self.task_ids
                ).evaluate()
            except Exception as error:
                print(f"Evaluation error for {num} examples: {error}")

    def _get_evaluation_paths(self, num_examples):
        """
        Helper method to generate paths for evaluation.
        """
        predictions_path = self.output_path_base / f"{num_examples}_examples"
        output_file = predictions_path / f"metrics_{num_examples}_examples.json"
        return predictions_path, output_file


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run prediction tasks for a given model.")
    parser.add_argument("--datapath", type=Path, required=True, help="Path to the data directory.")
    parser.add_argument("--ground_truth_path", type=Path, required=False, help="Path to the ground truth data directory.")
    parser.add_argument("--task_ids", type=int, nargs="+", help="Task IDs to generate examples for.", required=True)
    parser.add_argument("--model_name", type=str, required=False, help="The name of the model to run the prediction tasks for.", default="mistral-nemo")
    parser.add_argument("--num_examples", type=int, nargs="+", help="Number of examples to generate for each task.", required=False, default=[0, 5, 10])
    parser.add_argument("--n_runs", type=int, help="Number of runs.", required=False, default=5)
    parser.add_argument("--temperature", type=float, nargs="+", help="Temperature for generation.", required=False, default=[0.3])
    parser.add_argument("--example_selectors", type=str, nargs="+", help="Example selectors to use for generation.", required=False, default=["MMR"])
    parser.add_argument("--run_name", type=str, help="Name of the run.", required=False, default="run")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation of the generated examples.")
    return parser.parse_args()


def main():
    """
    Main function to initialize and run the task runner and evaluation.
    """
    args = parse_args()
    if not args.skip_evaluation and not args.ground_truth_path:
        raise ValueError("Ground truth data path is required for evaluation.")

    task_runner = TaskRunner(
        datapath=args.datapath,
        model_name=args.model_name,
        task_ids=args.task_ids,
        num_examples=args.num_examples,
        n_runs=args.n_runs,
        temperatures=args.temperature,
        example_selectors=args.example_selectors,
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
