import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_ollama import ChatOllama

from llm_extractinator.predictor import Predictor
from llm_extractinator.translator import Translator
from llm_extractinator.utils import save_json

# Configure logging
logger = logging.getLogger(__name__)


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
        "translation_dir",
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
        missing_params = [
            param for param in self.REQUIRED_PARAMS if param not in kwargs
        ]
        if missing_params:
            logger.error("Missing required parameters: %s", ", ".join(missing_params))
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

        for key in self.REQUIRED_PARAMS:
            setattr(self, key, kwargs.get(key, None))

        self.output_path_base = Path(self.output_dir) / Path(self.run_name)
        self.format = "" if self.reasoning_model else "json"
        self.model = self.initialize_model()

        self.predictor = Predictor(
            model=self.model,
            task_config=self.task_config,
            examples_path=self.example_dir,
            num_examples=self.num_examples,
            output_format=self.format,
            task_dir=self.task_dir,
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
        logger.info("Loading examples from training data.")
        return self.train[["input", "output"]].to_dict(orient="records")

    def _translate_task(self) -> None:
        self.translation_path = Path(self.translation_dir) / f"{self.task_id}.json"

        if self.translation_path.exists() and not self.overwrite:
            logger.info("Translation file already exists. Skipping translation.")
            return

        logger.info("Translating Task %s", self.task_id)
        self.translation_path.parent.mkdir(parents=True, exist_ok=True)

        translator = Translator(model=self.model, input_field=self.input_field)
        translator.translate(self.test, self.translation_path)

        with self.translation_path.open("r") as f:
            self.test = pd.read_json(f)

    def run(self) -> List[Path]:
        if self.translate:
            self._translate_task()

        examples = self._load_examples() if self.num_examples > 0 else None

        self.predictor.prepare_prompt_ollama(
            embedding_model="nomic-embed-text",
            examples=examples,
        )

        outpath_list = []
        for run_idx in range(self.n_runs):
            outpath_list.append(self._run_single_prediction(run_idx))

        return outpath_list

    def _run_single_prediction(self, run_idx: int) -> Path:
        output_path = self.output_path_base / f"{self.task_name}-run{run_idx}"
        output_path.mkdir(parents=True, exist_ok=True)

        prediction_file = output_path / "nlp-predictions-dataset.json"
        if prediction_file.exists() and not self.overwrite:
            logger.warning("Prediction %d already exists. Skipping.", run_idx + 1)
            return prediction_file

        if self.chunk_size is not None:
            logger.info("Processing in chunks of size %d.", self.chunk_size)
            for chunk_idx in range(0, len(self.test), self.chunk_size):
                chunk_output_path = (
                    output_path / f"nlp-predictions-dataset-{chunk_idx}.json"
                )
                if chunk_output_path.exists() and not self.overwrite:
                    logger.warning("Chunk %d already exists. Skipping.", chunk_idx)
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

            logger.info("Merging chunked predictions.")
            chunk_files = list(output_path.glob("nlp-predictions-dataset-*.json"))
            chunk_predictions = []
            for chunk_file in chunk_files:
                with chunk_file.open("r") as f:
                    chunk_predictions.extend(json.load(f))

            filename = (
                f"nlp-predictions-dataset-{self.data_split}.json"
                if self.data_split
                else "nlp-predictions-dataset.json"
            )
            save_json(chunk_predictions, outpath=output_path, filename=filename)
            for chunk_file in chunk_files:
                chunk_file.unlink()

            return output_path / filename
        else:
            logger.info("Running predictions without chunking.")
            results = self.predictor.predict(self.test)
            predictions = [
                {**sample._asdict(), **result}
                for sample, result in zip(self.test.itertuples(index=False), results)
            ]
            filename = (
                f"nlp-predictions-dataset-{self.data_split}.json"
                if self.data_split
                else "nlp-predictions-dataset.json"
            )
            save_json(predictions, outpath=output_path, filename=filename)

            return output_path / filename
