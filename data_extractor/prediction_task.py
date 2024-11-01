import json
from pathlib import Path
from typing import Dict

from langchain_ollama import ChatOllama

from data_extractor.data_loader import DataLoader, TaskLoader
from data_extractor.predictor import Predictor
from data_extractor.utils import save_json


class PredictionTask:
    """
    A class to represent a prediction task that involves loading data, initializing a model,
    running predictions, and saving results.
    """

    def __init__(
        self,
        /,
        task_id: str,
        model_name: str,
        output_path_base: Path,
        task_dir: Path,
        num_examples: int,
        n_runs: int,
        temperature: float,
        max_context_len: int,
        num_predict: int,
        data_dir: Path = Path(__file__).resolve().parents[1] / "data",
    ) -> None:
        """
        Initialize the PredictionTask with the provided parameters.

        Args:
            task_id (str): Identifier for the task.
            model_name (str): The name of the model to be used for predictions.
            output_path_base (Path): Base path for saving the output.
            num_examples (int): Number of examples to generate.
            n_runs (int): Number of runs for the prediction task.
            temperature (float): Temperature setting for the model.
            num_predict (int): Maximum number of tokens to generate per prediction.
            data_dir (Path, optional): Path to the data directory. Defaults to the data directory in the project.
        """
        self.task_id = task_id
        self.model_name = model_name
        self.output_path_base = output_path_base
        self.num_examples = num_examples
        self.n_runs = n_runs
        self.temperature = temperature
        self.max_context_len = max_context_len
        self.data_dir = data_dir
        self.num_predict = num_predict

        # Extract task information such as config, train and test paths
        self.task_dir = task_dir
        self._extract_task_info()

        # Setup output paths
        self.homepath = Path(__file__).resolve().parents[1]
        self.examples_path = self.homepath / f"examples/{self.task_name}_examples.json"

        # Initialize data and model
        self.data_loader = DataLoader(
            train_path=self.train_path, test_path=self.test_path
        )
        self.train, self.test = self.data_loader.load_data()
        self.model = self.initialize_model()
        self.predictor = Predictor(
            model=self.model,
            task_config=self.task_config,
            examples_path=self.examples_path,
            num_examples=self.num_examples,
        )

    def initialize_model(self) -> ChatOllama:
        """
        Initialize the model using the given model name and temperature.

        Returns:
            ChatOllama: The initialized model object.
        """
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            num_ctx=self.max_context_len,
            format="json",
        )

    def _extract_task_info(self) -> None:
        """
        Extract task information from the task configuration file.
        """
        task_loader = TaskLoader(folder_path=self.task_dir, task_id=self.task_id)
        self.task_config = task_loader.find_and_load_task()
        self.train_path = self.data_dir / self.task_config.get("Example_Path")
        self.test_path = self.data_dir / self.task_config.get("Data_Path")
        self.label_field = self.task_config.get("Label_Field")
        self.input_field = self.task_config.get("Input_Field")
        self.task_name = task_loader.get_task_name()

    def _load_examples(self) -> Dict:
        """
        Load examples from a JSON file if it exists, otherwise generate new examples.

        Returns:
            Dict: Loaded or generated examples.
        """
        if self.examples_path.exists():
            with self.examples_path.open("r") as f:
                return json.load(f)
        else:
            # Generate new examples if the file doesn't exist
            self.examples_path.parent.mkdir(parents=True, exist_ok=True)
            return self.predictor.generate_examples(self.train)

    def run(self) -> None:
        """
        Run the prediction task by preparing the model, running predictions, and saving the results.
        """
        # Load or generate examples for the task
        examples = self._load_examples() if self.num_examples > 0 else None

        # Prepare the predictor with the loaded examples
        self.predictor.prepare_prompt_ollama(examples=examples)

        # Run predictions across multiple runs
        for run_idx in range(self.n_runs):
            self._run_single_prediction(run_idx)

    def _run_single_prediction(self, run_idx: int) -> None:
        """
        Run a single prediction iteration and save the results.

        Args:
            run_idx (int): The index of the current run.
        """
        output_path = self.output_path_base / f"{self.task_name}-fold{run_idx}"
        output_path.mkdir(parents=True, exist_ok=True)

        prediction_file = output_path / "nlp-predictions-dataset.json"

        # Skip if predictions already exist
        if prediction_file.exists():
            print(
                f"Prediction {run_idx + 1} of {self.n_runs} already exists. Skipping..."
            )
            return

        print(f"Running prediction {run_idx + 1} of {self.n_runs}...")

        # Get prediction results
        results = self.predictor.predict(self.test)

        predictions = [
            {"uid": uid, **result} for uid, result in zip(self.test["uid"], results)
        ]

        # Save the predictions to a JSON file
        save_json(
            predictions, outpath=output_path, filename="nlp-predictions-dataset.json"
        )
