import json
from pathlib import Path
from typing import Tuple, Dict
from data_extractor.data_loader import DataLoader
from data_extractor.predictor import Predictor
from data_extractor.utils import save_json
from langchain_ollama import ChatOllama


class PredictionTask:
    """
    A class to represent a prediction task that involves loading data, initializing a model,
    running predictions, and saving results.
    """

    def __init__(self, task_id: str, model_name: str, datapath: Path, output_path_base: Path, 
                num_examples: int, n_runs: int, temperature: float, 
                example_selector_name: str) -> None:
        """
        Initialize the PredictionTask with the provided parameters.
        
        Args:
            task_id (str): Identifier for the task.
            model_name (str): The name of the model to be used for predictions.
            datapath (Path): Base path where task folders are located.
            output_path_base (Path): Base path for saving the output.
            num_examples (int): Number of examples to generate.
            n_runs (int): Number of runs for the prediction task.
            temperature (float): Temperature setting for the model.
            example_selector_name (str): The name of the example selector.
        """
        self.task_id = task_id
        self.model_name = model_name
        self.datapath = datapath
        self.output_path_base = output_path_base
        self.num_examples = num_examples
        self.n_runs = n_runs
        self.temperature = temperature
        self.example_selector_name = example_selector_name

        self.homepath = Path(__file__).resolve().parents[1]
        
        # Extract task information such as config, train and test paths
        self.task_config, self.train_path, self.test_path = self._extract_task_info()
        self.task_name = self.task_config['task_name']
        self.input_name = self.task_config['input_name']
        self.label_name = self.task_config['label_name']

        # Setup output paths
        self.output_path_base = self.output_path_base / f"{self.num_examples}_examples"
        self.examples_path = self.homepath / f"examples/{self.task_name}_examples.json"

        # Initialize data and model
        self.data_loader = DataLoader(train_path=self.train_path, test_path=self.test_path)
        self.train, self.test = self.data_loader.load_data()
        self.model = self.initialize_model()
        self.predictor = Predictor(model=self.model, task_config=self.task_config, 
                                   examples_path=self.examples_path, num_examples=self.num_examples)

        # Load task info
        self.task_info = self.predictor.load_tasks()[self.task_name]

    def initialize_model(self) -> ChatOllama:
        """
        Initialize the model using the given model name and temperature.
        
        Returns:
            ChatOllama: The initialized model object.
        """
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=1024,
            format="json",
        )

    def _extract_task_info(self) -> Tuple[Dict, Path, Path]:
        """
        Extract task configuration and find the train and test file paths based on the task ID.
        
        Returns:
            Tuple[Dict, Path, Path]: Task configuration, training data path, and test data path.
        
        Raises:
            ValueError: If no matching folders or files are found for the task_id.
        """
        train_folder = self.datapath / "debug-input"
        test_folder = self.datapath / "debug-test-set"

        train_task_folder = next((folder for folder in train_folder.iterdir() if self.task_id in folder.name), None)
        if not train_task_folder:
            raise ValueError(f"No matching training folder found for task_id: {self.task_id}")
        
        train_path = next((file for file in train_task_folder.iterdir() if "training" in file.name), None)
        test_path = next((file for file in test_folder.iterdir() if self.task_id in file.name), None)

        if not train_path or not test_path:
            raise ValueError(f"No training or test files found for task_id: {self.task_id}")
        
        task_config_path = train_task_folder / "nlp-task-configuration.json"
        with task_config_path.open('r') as f:
            task_config = json.load(f)
        
        return task_config, train_path, test_path

    def _load_examples(self) -> Dict:
        """
        Load examples from a JSON file if it exists, otherwise generate new examples.
        
        Returns:
            Dict: Loaded or generated examples.
        """
        if self.examples_path.exists():
            with self.examples_path.open('r') as f:
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
        self.predictor.prepare_prompt_ollama(examples=examples, example_selector_name=self.example_selector_name)

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
            print(f"Prediction {run_idx + 1} of {self.n_runs} already exists. Skipping...")
            return

        print(f"Running prediction {run_idx + 1} of {self.n_runs}...")

        # Get prediction results
        results = self.predictor.predict(self.test)
        
        predictions = [
            {
                "uid": uid, 
                self.label_name.replace("_target", ""): label, 
                "reasoning": reasoning
            }
            for uid, label, reasoning in zip(
                self.test['uid'], 
                [result['label'] for result in results], 
                [result['reasoning'] for result in results]
            )
        ]
        
        # Save the predictions to a JSON file
        save_json(predictions, outpath=output_path, filename="nlp-predictions-dataset.json")
