import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_ollama import ChatOllama

from llm_extractinator.data_loader import DataLoader
from llm_extractinator.predictor import Predictor
from llm_extractinator.translator import Translator
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
        "train_path",
        "test_path",
        "input_field",
        "task_name",
        "task_config",
        "data_split",
        "train",
        "test",
    }

    def __init__(self, **kwargs) -> None:
        # Check required parameters
        missing_params = [
            param for param in self.REQUIRED_PARAMS if param not in kwargs
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

        for key in self.REQUIRED_PARAMS:
            setattr(self, key, kwargs.get(key, None))

        # Setup output paths
        self.homepath = Path(__file__).resolve().parents[1]
        self.translation_path = (
            self.homepath / f"translations/{self.task_name}_translations.json"
        )
        self.output_path_base = Path(self.output_dir) / Path(self.run_name)

        # If user selects "auto" for context length, compute it (example)
        if self.max_context_len == "auto":
            self.max_context_len = self.data_loader.get_max_input_tokens(
                input_field=self.input_field,
                num_predict=self.num_predict,
                buffer_tokens=1000,
            )

        # Decide whether we want JSON output or not
        if self.reasoning_model:
            self.format = ""
        else:
            self.format = "json"

        # Initialize the LLM model
        self.model = self.initialize_model()

        # Create a predictor instance (no more 'generate_translations' inside Predictor)
        self.predictor = Predictor(
            model=self.model,
            task_config=self.task_config,
            examples_path=self.example_dir,
            num_examples=self.num_examples,
            output_format=self.format,
        )

    def initialize_model(self) -> ChatOllama:
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

    def _load_examples(self) -> List[Dict]:
        """
        Loads the examples for the task from the training data.
        """
        return self.train[["input", "output"]].to_dict(orient="records")

    def _translate_task(self) -> None:
        """
        Translate the text of the dataset to English.
        """
        # If there's already a translation file and we don't want to overwrite, skip
        if not self.translation_path.exists():
            print(f"Translating Task {self.task_id}...")
            self.translation_path.parent.mkdir(parents=True, exist_ok=True)

            # Use the dedicated translator
            translator = Translator(model=self.model, input_field=self.input_field)
            translator.generate_translations(self.test, self.translation_path)

        with self.translation_path.open("r") as f:
            self.test = pd.read_json(f)

    def run(self) -> List[Path]:
        """
        Run the prediction task by preparing the model, running predictions, and saving the results.
        """
        # Optionally translate first
        if self.translate:
            self._translate_task()

        # Load or generate examples for the task
        examples = self._load_examples() if self.num_examples > 0 else None

        # Prepare the predictor with the loaded examples
        self.predictor.prepare_prompt_ollama(
            model_name="nomic-embed-text",  # or the actual embedding model name you prefer
            examples=examples,
        )

        outpath_list = []

        # Run predictions across multiple runs
        for run_idx in range(self.n_runs):
            outpath_list.append(self._run_single_prediction(run_idx))

        return outpath_list

    def _run_single_prediction(self, run_idx: int) -> Path:
        """
        Run a single prediction iteration and save the results.
        """
        output_path = self.output_path_base / f"{self.task_name}-run{run_idx}"
        output_path.mkdir(parents=True, exist_ok=True)

        prediction_file = output_path / "nlp-predictions-dataset.json"

        # Skip if predictions already exist
        if prediction_file.exists() and not self.overwrite:
            print(
                f"Prediction {run_idx + 1} of {self.n_runs} already exists. Skipping..."
            )
            return prediction_file

        print(f"Running prediction {run_idx + 1} of {self.n_runs}...")

        # If chunking is enabled
        if self.chunk_size is not None:
            for chunk_idx in range(0, len(self.test), self.chunk_size):
                chunk_output_path = (
                    output_path / f"nlp-predictions-dataset-{chunk_idx}.json"
                )
                if chunk_output_path.exists() and not self.overwrite:
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

            # Merge chunked predictions
            chunk_files = list(output_path.glob("nlp-predictions-dataset-*.json"))
            chunk_predictions = []
            for chunk_file in chunk_files:
                with chunk_file.open("r") as f:
                    chunk_predictions.extend(json.load(f))

            # Decide on a final filename for merged predictions
            if self.data_split is not None:
                filename = f"nlp-predictions-dataset-{self.data_split}.json"
            else:
                filename = "nlp-predictions-dataset.json"

            save_json(chunk_predictions, outpath=output_path, filename=filename)

            # Optionally remove the chunk files
            for chunk_file in chunk_files:
                chunk_file.unlink()

            return output_path / filename

        # Otherwise, no chunking
        else:
            results = self.predictor.predict(self.test)
            predictions = [
                {**sample._asdict(), **result}
                for sample, result in zip(self.test.itertuples(index=False), results)
            ]

            if self.data_split is not None:
                filename = f"nlp-predictions-dataset-{self.data_split}.json"
            else:
                filename = "nlp-predictions-dataset.json"

            save_json(predictions, outpath=output_path, filename=filename)
            return output_path / filename
