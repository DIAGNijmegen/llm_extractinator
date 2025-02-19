import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import tiktoken

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self, train_path: Optional[str] = None, test_path: Optional[str] = None
    ) -> None:
        """
        Initializes the DataLoader with optional paths for training and testing datasets.

        Args:
            train_path (Optional[str]): Path to the training data file. Default is None.
            test_path (Optional[str]): Path to the testing data file. Default is None.
        """
        self.train_path = Path(train_path) if train_path else None
        self.test_path = Path(test_path) if test_path else None

        # Store data after loading
        self.train = None
        self.test = None

    def validate_file(self, file_path: Path) -> None:
        """
        Validates if the file exists and is a supported format.

        Args:
            file_path (Path): The path to the file to validate.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        if file_path.suffix.lower() not in {".json", ".csv"}:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. Supported formats are .json, .csv"
            )

    def load_data(
        self, text_column: str = "text"
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Loads the training and testing datasets if paths are provided and valid.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: DataFrames for training and testing data.
        """
        if self.train_path:
            self.validate_file(self.train_path)
            self.train = self._load_file(self.train_path, text_column)

        if self.test_path:
            self.validate_file(self.test_path)
            self.test = self._load_file(self.test_path, text_column)

        return self.train, self.test

    def _load_file(
        self,
        file_path: Path,
        text_column: str = "text",
        token_column: str = "token_count",
    ) -> pd.DataFrame:
        """
        Loads a data file based on its extension.

        Args:
            file_path (Path): The path to the file to load.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        if file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
            return self.add_token_count(df, text_column, token_column)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
            return self.add_token_count(df, text_column, token_column)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def count_tokens(self, text: str, model_name: str = "cl100k_base") -> int:
        """
        Estimate the number of tokens in a given text.

        Args:
            text (str): The text to tokenize.
            model_name (str): The model to use for token estimation.

        Returns:
            int: The estimated token count.
        """
        try:
            encoding = tiktoken.get_encoding(model_name)
            return len(encoding.encode(text))
        except Exception:
            avg_token_ratio = 1.2  # Approximate: Avg token per word
            return int(len(text.split()) * avg_token_ratio)

    def add_token_count(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        token_column: str = "token_count",
    ) -> pd.DataFrame:
        """
        Adds a new column to the DataFrame with the token count for each text.

        Args:
            df (pd.DataFrame): The DataFrame containing the text column.
            text_column (str): The name of the text column.
            token_column (str): The name of the new token count column.

        Returns:
            pd.DataFrame: The DataFrame with the new token count column.
        """
        logger.info("Adding token count column to DataFrame.")
        df[token_column] = df[text_column].apply(self.count_tokens)
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        token_column: str = "token_count",
        quantile: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into two based on the quantile of the token count in the text column.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            text_column (str): The name of the text column.
            quantile (float): The quantile to split on. Default is 0.8.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames split by the quantile.
        """
        logger.info(f"Splitting DataFrame based on token count quantile: {quantile}")

        threshold = df[token_column].quantile(quantile)
        short_df = df[df[token_column] <= threshold]
        long_df = df[df[token_column] > threshold]

        return short_df, long_df

    def get_max_input_tokens(
        self,
        df: pd.DataFrame,
        token_column: str = "token_count",
        num_predict: int = 512,
        buffer_tokens: int = 1000,
        num_examples: int = 0,
    ) -> int:
        """
        Computes the maximum token count for input data, considering a buffer.

        Args:
            df (pd.DataFrame): The DataFrame containing the token count column.
            token_column (str): The name of the token count column.
            num_predict (int): The number of tokens to predict. Default is 512.
            buffer_tokens (int): The buffer tokens to add. Default is 1000.

        Returns:
            int: The maximum token count for input data.
        """
        return df[token_column].max() * (num_examples + 1) + buffer_tokens + num_predict

    def adapt_num_predict(
        self,
        df: pd.DataFrame,
        token_column: str = "token_count",
        buffer_tokens: int = 1000,
        translate: bool = False,
        reasoning_model: bool = False,
        num_predict: int = 512,
    ) -> int:
        """
        Computes the maximum token count for input data, considering a buffer.

        Args:
            df (pd.DataFrame): The DataFrame containing the token count column.
            token_column (str): The name of the token count column.
            buffer_tokens (int): The buffer tokens to add. Default is 1000.

        Returns:
            int: The maximum token count for input data.
        """
        if translate:
            num_predict = df[token_column].max() + buffer_tokens
        if reasoning_model:
            num_predict = num_predict + buffer_tokens
        logging.info(f"Adapting num_predict to: {num_predict}")
        return num_predict


class TaskLoader:
    def __init__(self, folder_path: str, task_id: int):
        """
        Initializes the TaskLoader with the folder path and task ID.

        Args:
            folder_path (str): The path to the folder containing task files.
            task_id (int): The ID of the task to load.
        """
        self.folder_path = Path(folder_path)
        self.task_id = task_id
        self.file_path = None

    def find_and_load_task(self) -> Dict:
        """
        Finds and loads the task file matching the task ID.

        Returns:
            Dict: The loaded task data from the JSON file.

        Raises:
            FileNotFoundError: If no matching file is found.
            RuntimeError: If multiple matching files are found.
        """
        if not self.folder_path.exists():
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")

        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"The path {self.folder_path} is not a directory.")

        # Regex pattern to match files like TaskXXX_name.json ensuring exact match of task ID
        pattern = re.compile(rf"Task{self.task_id:03}(?!\d).*\.json")

        # List all files in the folder that match the pattern
        matching_files = [
            f
            for f in self.folder_path.iterdir()
            if f.is_file() and pattern.match(f.name)
        ]

        # Check for exactly one match
        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"No file found matching Task{self.task_id:03}*.json in {self.folder_path}"
            )
        elif len(matching_files) > 1:
            raise RuntimeError(
                f"Multiple files found matching Task{self.task_id:03}*.json in {self.folder_path}"
            )

        # Load the JSON file
        self.file_path = matching_files[0]
        with self.file_path.open("r") as file:
            data = json.load(file)

        return data

    def get_task_name(self) -> str:
        """
        Extracts the task name from the file name.

        Returns:
            str: The task name without extension.
        """
        if not self.file_path:
            raise ValueError(
                "No task file loaded. Please run find_and_load_task first."
            )

        return self.file_path.stem
