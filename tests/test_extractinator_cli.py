import json
import subprocess
import time
import unittest
from pathlib import Path


class TestExtractinatorCLI(unittest.TestCase):
    def setUp(self):
        """Set up paths and test input before running the test."""
        self.basepath = Path(__file__).resolve().parents[1] / "tests"
        self.output_dir = self.basepath / "testoutput"
        self.run_name = "test_run_cli/Task999_example-run0"
        self.expected_output_file = (
            self.output_dir / self.run_name / "nlp-predictions-dataset.json"
        )

        print(
            "Does the expected output file exist before test?",
            self.expected_output_file.exists(),
        )

        # Ensure output directory is clean before test
        try:
            if self.output_dir.exists():
                for file in self.output_dir.iterdir():
                    file.unlink()  # Remove old test outputs
        except:
            print(
                "Error cleaning output directory. Empty manually before running test."
            )

    def test_extractinate_cli_execution(self):
        """Runs extractinate via command line and verifies output."""

        command = [
            "python",
            "-m",
            "llm_extractinator.main",
            "--model_name",
            "qwen2.5:0.5b",
            "--task_id",
            "999",
            "--num_examples",
            "0",
            "--n_runs",
            "1",
            "--temperature",
            "0",
            "--max_context_len",
            "auto",
            "--run_name",
            "test_run_cli",
            "--num_predict",
            "512",
            "--output_dir",
            str(self.output_dir),
            "--task_dir",
            str(self.basepath / "testtasks"),
            "--data_dir",
            str(self.basepath / "testdata"),
            "--translate",
            "False",
            "--verbose",
            "False",
            "--overwrite",
            "True",
            "--seed",
            "42",
        ]

        # Run extractinate as a CLI command
        result = subprocess.run(command, capture_output=True, text=True)

        # Print any errors if the test fails
        if result.returncode != 0:
            print("Command Output:", result.stdout)
            print("Command Error:", result.stderr)

        # Verify that the command executed successfully
        self.assertEqual(result.returncode, 0, "CLI command failed to execute.")

        # Wait for the model to complete execution
        time.sleep(10)

        # Check if the output file was created
        self.assertTrue(
            self.expected_output_file.exists(), "Expected output file was not created."
        )

        # Read the output file
        with open(self.expected_output_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        # Ensure predictions are not empty
        self.assertTrue(len(predictions) > 0, "Output file contains no data.")

        # Check that the required keys are present in the actual output
        for idx, prediction in enumerate(predictions):
            with self.subTest(index=idx):
                # Ensure required keys are present in the prediction
                required_keys = {"HR", "Name", "status", "retry_count"}
                missing_keys = required_keys - prediction.keys()

                self.assertFalse(
                    missing_keys,
                    f"Missing keys at index {idx}: {missing_keys}",
                )


if __name__ == "__main__":
    unittest.main()
