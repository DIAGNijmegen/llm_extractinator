import json
import time
import unittest
from pathlib import Path

from llm_extractinator.main import extractinate


class TestExtractinator(unittest.TestCase):
    def setUp(self):
        """Set up paths and test input before running the test."""
        self.basepath = Path(__file__).resolve().parents[1] / "tests"
        self.output_dir = self.basepath / "testoutput"
        self.run_name = "test_run/Task999_example-run0"
        self.expected_output_file = (
            self.output_dir / self.run_name / "nlp-predictions-dataset.json"
        )
        print(
            "Does the expected output file exist?", self.expected_output_file.exists()
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

    def test_extractinate_execution(self):
        """Runs extractinate and verifies if output matches expected results."""

        # Run extractinate with test input
        extractinate(
            model_name="qwen2.5:0.5b",
            task_id=999,
            num_examples=0,
            n_runs=1,
            temperature=0.3,
            max_context_len=4096,
            run_name="test_run",
            num_predict=512,
            output_dir=self.output_dir,
            task_dir=self.basepath / "testtasks",
            log_dir=self.basepath / "testlogs",
            data_dir=self.basepath / "testdata",
            translate=False,
            verbose=False,
            overwrite=True,
            seed=42,
        )

        # Wait for the model to complete execution
        time.sleep(10)  # Small delay in case of async writing

        # Check if the output file was created
        self.assertTrue(
            self.expected_output_file.exists(), "Expected output file was not created."
        )

        # Read the output file
        with open(self.expected_output_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        # Ensure predictions are not empty
        self.assertTrue(len(predictions) > 0, "Output file contains no data.")

        # Compare actual output with expected results
        for idx, prediction in enumerate(predictions):
            with self.subTest(index=idx):
                expected_output = prediction["expected_output"]
                actual_output = {"a": prediction["a"], "b": prediction["b"]}

                # Verify that the 'a' and 'b' values match the expected output
                self.assertEqual(
                    actual_output,
                    expected_output,
                    f"Mismatch at index {idx}. Expected {expected_output}, got {actual_output}.",
                )

                # Verify that the status is "success"
                self.assertEqual(
                    prediction["status"],
                    "success",
                    f"Status at index {idx} is not 'success'. Found {prediction['status']}.",
                )

                # Ensure retry count is 0
                self.assertEqual(
                    prediction["retry_count"],
                    0,
                    f"Retry count at index {idx} is not 0. Found {prediction['retry_count']}.",
                )


if __name__ == "__main__":
    unittest.main()
