import json
from pathlib import Path
from typing import Dict

import pandas as pd
from langchain_ollama import ChatOllama

from llm_extractinator.data_loader import DataLoader, TaskLoader
from llm_extractinator.predictor import Predictor
from llm_extractinator.utils import save_json


class PredictionTask:
    REQUIRED_PARAMS = {
        "task_id",
        "model_name",
        "output_dir",
        "task_dir",
        "num_examples",
        "n_runs",
        "run_name",
        "temperature",
        "max_context_len",
        "num_predict",
        "data_dir",
        "example_dir",
        "chunk_size",
        "translate",
        "verbose",
        "overwrite",
        "seed",
        "top_k",
        "top_p",
        "reasoning_model",
    }

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PredictionTask dynamically by extracting only required parameters.

        Args:
            kwargs: Dictionary of parameters.
        """
        missing_params = [
            param for param in self.REQUIRED_PARAMS if param not in kwargs
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

        for key in self.REQUIRED_PARAMS:
            setattr(self, key, kwargs.get(key, None))

        self._extract_task_info()

        # Setup output paths
        self.homepath = Path(__file__).resolve().parents[1]
        self.translation_path = (
            self.homepath / f"translations/{self.task_name}_translations.json"
        )
        self.output_path_base = Path(self.output_dir) / Path(self.run_name)

        # Initialize data and model
        self.data_loader = DataLoader(
            train_path=self.train_path, test_path=self.test_path
        )
        self.train, self.test = self.data_loader.load_data()

        # Determine max_context_len dynamically
        if self.max_context_len == "auto":
            self.max_context_len = self.data_loader.get_max_input_tokens(
                input_field=self.input_field,
                num_predict=self.num_predict,
                buffer_tokens=1000,
            )

        self.model = self.initialize_model()
        self.predictor = Predictor(
            model=self.model,
            task_config=self.task_config,
            examples_path=self.example_dir,
            num_examples=self.num_examples,
            format=self.format,
        )

    def initialize_model(self) -> ChatOllama:
        """
        Initialize the model using the given model name and temperature.

        Returns:
            ChatOllama: The initialized model object.
        """
        if self.reasoning_model:
            self.format = ""
        else:
            self.format = "json"
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            num_ctx=self.max_context_len,
            format=self.format,
            verbose=self.verbose,
            seed=self.seed,
            top_k=self.top_k,
            top_p=self.top_p,
        )

    def _extract_task_info(self) -> None:
        """
        Extract task information from the task configuration file.
        """
        task_loader = TaskLoader(folder_path=self.task_dir, task_id=self.task_id)
        self.task_config = task_loader.find_and_load_task()
        self.example_file = self.task_config.get("Example_Path")
        if self.example_file is not None:
            self.train_path = self.example_dir / self.task_config.get("Example_Path")
        else:
            self.train_path = None
        self.test_path = self.data_dir / self.task_config.get("Data_Path")
        self.input_field = self.task_config.get("Input_Field")
        self.task_name = task_loader.get_task_name()

    def _load_examples(self) -> Dict:
        """
        Loads the examples for the task from the training data.

        Returns:
            Dict: Loaded examples.
        """
        return self.train[["input", "output"]].to_dict(orient="records")

    def _translate_task(self) -> None:
        """
        Translate the text of the dataset to English.
        """
        if not self.translation_path.exists():
            print(f"Translating Task {self.task_id}...")
            self.translation_path.parent.mkdir(parents=True, exist_ok=True)
            self.predictor.generate_translations(self.test, self.translation_path)

        with self.translation_path.open("r") as f:
            self.test = pd.read_json(f)

    def run(self) -> None:
        """
        Run the prediction task by preparing the model, running predictions, and saving the results.
        """
        if self.translate:
            self._translate_task()

        # Load or generate examples for the task
        examples = self._load_examples() if self.num_examples > 0 else None

        # Prepare the predictor with the loaded examples
        self.predictor.prepare_prompt_ollama(
            model_name="nomic-embed-text", examples=examples
        )

        # Run predictions across multiple runs
        for run_idx in range(self.n_runs):
            self._run_single_prediction(run_idx)

    def _run_single_prediction(self, run_idx: int) -> None:
        """
        Run a single prediction iteration and save the results.

        Args:
            run_idx (int): The index of the current run.
        """
        output_path = self.output_path_base / f"{self.task_name}-run{run_idx}"
        output_path.mkdir(parents=True, exist_ok=True)

        prediction_file = output_path / "nlp-predictions-dataset.json"

        # Skip if predictions already exist
        if prediction_file.exists() and not self.overwrite:
            print(
                f"Prediction {run_idx + 1} of {self.n_runs} already exists. Skipping..."
            )
            return

        print(f"Running prediction {run_idx + 1} of {self.n_runs}...")

        if self.chunk_size is not None:
            for chunk_idx in range(0, len(self.test), self.chunk_size):
                chunk_output_path = (
                    output_path / f"nlp-predictions-dataset-{chunk_idx}.json"
                )
                if chunk_output_path.exists():
                    print(f"Chunk prediction {chunk_idx} already exists. Skipping...")
                    continue

                samples = self.test.iloc[chunk_idx : chunk_idx + self.chunk_size]
                chunk_results = self.predictor.predict(samples)
                chunk_predictions = [
                    {**sample._asdict(), **result}
                    for sample, result in zip(
                        samples.itertuples(index=False), chunk_results
                    )
                ]
                save_json(
                    chunk_predictions,
                    outpath=output_path,
                    filename=chunk_output_path.name,
                )

            # Merge the chunk predictions into a single file
            chunk_files = list(output_path.glob("nlp-predictions-dataset-*.json"))
            chunk_predictions = []
            for chunk_file in chunk_files:
                with chunk_file.open("r") as f:
                    chunk_predictions.extend(json.load(f))

            # Save the predictions to a JSON file
            save_json(
                chunk_predictions,
                outpath=output_path,
                filename="nlp-predictions-dataset.json",
            )

            # Remove the chunk files
            for chunk_file in chunk_files:
                chunk_file.unlink()

        else:
            # Get prediction results
            results = self.predictor.predict(self.test)

            predictions = [
                {**sample._asdict(), **result}
                for sample, result in zip(self.test.itertuples(index=False), results)
            ]

            # Save the predictions to a JSON file
            save_json(
                predictions,
                outpath=output_path,
                filename="nlp-predictions-dataset.json",
            )
