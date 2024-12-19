import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


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

        if file_path.suffix.lower() not in {".json", ".csv", ".parquet"}:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. Supported formats are .json, .csv, and .parquet."
            )

    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Loads the training and testing datasets if paths are provided and valid.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: DataFrames for training and testing data.
        """
        train = None
        test = None

        if self.train_path:
            self.validate_file(self.train_path)
            train = self._load_file(self.train_path)

        if self.test_path:
            self.validate_file(self.test_path)
            test = self._load_file(self.test_path)

        return train, test

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Loads a data file based on its extension.

        Args:
            file_path (Path): The path to the file to load.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        if file_path.suffix.lower() == ".json":
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


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
