import json
import shutil
import subprocess
import time
import unittest
from pathlib import Path

from llm_extractinator.main import extractinate


def clean_output_dir(output_dir: Path) -> None:
    """Removes all files and subdirectories in the output directory."""
    try:
        if output_dir.exists() and output_dir.is_dir():
            for item in output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    except Exception as e:
        print(f"Error cleaning output directory: {e}")


class TestExtractinator(unittest.TestCase):
    """Test cases for Extractinator with different models"""

    def setUp(self):
        """Set up paths and test input before running the test."""
        self.basepath = Path(__file__).resolve().parents[1] / "tests"
        self.output_dir = self.basepath / "testoutput"

        # Ensure output directory is clean before test
        clean_output_dir(self.output_dir)

    def run_extractinate_test(
        self,
        model_name,
        run_name,
        num_predict,
        reasoning_model,
        n_runs,
        max_context_len,
        translate,
    ):
        """Runs extractinate and verifies if output matches expected results."""

        # Run extractinate with test input
        extractinate(
            model_name=model_name,
            task_id=999,
            num_examples=0,
            n_runs=n_runs,
            temperature=0.0,
            max_context_len=max_context_len,
            run_name=run_name,
            num_predict=num_predict,
            output_dir=self.output_dir,
            task_dir=self.basepath / "testtasks",
            data_dir=self.basepath / "testdata",
            translation_dir=self.basepath / "testtranslations",
            translate=translate,
            verbose=False,
            overwrite=True,
            seed=42,
            reasoning_model=reasoning_model,
        )

        # Wait for the model to complete execution
        time.sleep(10)  # Small delay in case of async writing

        for run_idx in range(n_runs):
            run_output_path = (
                self.output_dir / f"{run_name}/Task999_example-run{run_idx}"
            )
            expected_output_file = run_output_path / "nlp-predictions-dataset.json"

            # Check if the output file was created
            self.assertTrue(
                expected_output_file.exists(), "Expected output file was not created."
            )

            # Read the output file
            with open(expected_output_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)

            # Ensure predictions are not empty
            self.assertTrue(len(predictions) > 0, "Output file contains no data.")

            # Check each prediction
            for idx, prediction in enumerate(predictions):
                with self.subTest(index=idx):
                    required_keys = {"HR", "Name", "status"}
                    missing_keys = required_keys - prediction.keys()
                    self.assertFalse(
                        missing_keys, f"Missing keys at index {idx}: {missing_keys}"
                    )

                    # If status is "success", compare actual vs. expected output
                    if prediction.get("status") == "success":
                        expected_output = prediction.get("expected_output", {})
                        actual_output = {
                            k: prediction[k] for k in ["HR", "Name"] if k in prediction
                        }
                        self.assertEqual(
                            actual_output,
                            expected_output,
                            f"Mismatch in output at index {idx}: expected {expected_output}, got {actual_output}",
                        )

    # Individual test cases for different models
    def test_extractinate_base(self):
        """Test extractinate with Qwen model"""
        self.run_extractinate_test(
            model_name="gemma2",
            run_name="test_run",
            num_predict=256,
            reasoning_model=False,
            n_runs=1,
            max_context_len=512,
            translate=False,
        )

    def test_extractinate_deepseek(self):
        """Test extractinate with DeepSeek model"""
        self.run_extractinate_test(
            model_name="deepseek-r1:1.5b",
            run_name="test_run_deepseek",
            num_predict=1024,
            reasoning_model=True,
            n_runs=1,
            max_context_len="max",
            translate=False,
        )

    def test_extractinate_ctx_auto(self):
        """Test extractinate with max_context_len set to split"""
        self.run_extractinate_test(
            model_name="deepseek-r1:1.5b",
            run_name="test_run_max_context_len",
            num_predict=1024,
            reasoning_model=True,
            n_runs=1,
            max_context_len="split",
            translate=False,
        )

    def test_extractinate_n_runs(self):
        """Test extractinate with multiple runs"""
        self.run_extractinate_test(
            model_name="gemma2",
            run_name="test_run_deepseek_n_runs",
            num_predict=1024,
            reasoning_model=False,
            n_runs=5,
            max_context_len=512,
            translate=False,
        )

    def test_extractinate_n_runs_and_ctx_auto(self):
        """Test extractinate with multiple runs and max_context_len set to auto"""
        self.run_extractinate_test(
            model_name="gemma2",
            run_name="test_run_deepseek_n_runs_auto",
            num_predict=1024,
            reasoning_model=False,
            n_runs=5,
            max_context_len="split",
            translate=False,
        )

    def test_extractinate_translate(self):
        """Test extractinate with translation enabled"""
        self.run_extractinate_test(
            model_name="gemma2",
            run_name="test_run_translate",
            num_predict=1024,
            reasoning_model=False,
            n_runs=1,
            max_context_len=512,
            translate=True,
        )

    def test_extractinate_translate_ctx_auto_n_runs(self):
        """Test extractinate with translation enabled, max_context_len set to auto and multiple runs"""
        self.run_extractinate_test(
            model_name="gemma2",
            run_name="test_run_translate_ctx_auto_n_runs",
            num_predict=1024,
            reasoning_model=False,
            n_runs=5,
            max_context_len="split",
            translate=True,
        )


class TestExtractinator2(unittest.TestCase):
    """Test cases for Extractinator with different models"""

    def setUp(self):
        """Set up paths and test input before running the test."""
        self.basepath = Path(__file__).resolve().parents[1] / "tests"
        self.output_dir = self.basepath / "testoutput"

        # Ensure output directory is clean before test
        clean_output_dir(self.output_dir)

    def run_extractinate_test2(
        self,
        model_name,
        run_name,
        num_predict,
        reasoning_model,
        n_runs,
        max_context_len,
        translate,
    ):
        """Runs extractinate and verifies if output matches expected results."""

        # Run extractinate with test input
        extractinate(
            model_name=model_name,
            task_id=998,
            num_examples=0,
            n_runs=n_runs,
            temperature=0.0,
            max_context_len=max_context_len,
            run_name=run_name,
            num_predict=num_predict,
            output_dir=self.output_dir,
            task_dir=self.basepath / "testtasks",
            data_dir=self.basepath / "testdata",
            translation_dir=self.basepath / "testtranslations",
            translate=translate,
            verbose=False,
            overwrite=True,
            seed=42,
            reasoning_model=reasoning_model,
        )

        # Wait for the model to complete execution
        time.sleep(10)  # Small delay in case of async writing

        for run_idx in range(n_runs):
            run_output_path = (
                self.output_dir / f"{run_name}/Task998_example2-run{run_idx}"
            )
            expected_output_file = run_output_path / "nlp-predictions-dataset.json"

            # Check if the output file was created
            self.assertTrue(
                expected_output_file.exists(), "Expected output file was not created."
            )

            # Read the output file
            with open(expected_output_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)

            # Ensure predictions are not empty
            self.assertTrue(len(predictions) > 0, "Output file contains no data.")

            # Check each prediction
            for idx, prediction in enumerate(predictions):
                with self.subTest(index=idx):
                    self.assertIn(
                        "status", prediction, f"Missing 'status' at index {idx}"
                    )

                    if prediction.get("status") == "success":
                        self.assertIn(
                            "expected_output",
                            prediction,
                            f"Missing 'expected_output' for successful prediction at index {idx}",
                        )

                        expected_output = prediction["expected_output"]
                        actual_products = prediction.get("products")

                        # Check that actual_products is a list of dicts with required keys
                        self.assertIsInstance(
                            actual_products,
                            list,
                            f"'products' should be a list at index {idx}",
                        )
                        for p_idx, product in enumerate(actual_products):
                            self.assertIn(
                                "name",
                                product,
                                f"Missing 'name' in product {p_idx} at index {idx}",
                            )
                            self.assertIn(
                                "price",
                                product,
                                f"Missing 'price' in product {p_idx} at index {idx}",
                            )
                            self.assertIsInstance(
                                product["price"],
                                (float, int),
                                f"'price' must be a number in product {p_idx} at index {idx}",
                            )

                        self.assertEqual(
                            actual_products,
                            expected_output["products"],
                            f"Mismatch in products at index {idx}: expected {expected_output['products']}, got {actual_products}",
                        )

    def test_extractinate_base_2(self):
        """Test extractinate with Qwen model"""
        self.run_extractinate_test2(
            model_name="gemma2",
            run_name="test_run",
            num_predict=256,
            reasoning_model=False,
            n_runs=1,
            max_context_len=512,
            translate=False,
        )


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

        # clean_output_dir(self.output_dir)

    def test_extractinate_cli_execution(self):
        """Runs extractinate via command line and verifies output."""
        command = [
            "python",
            "-m",
            "llm_extractinator.main",
            "--model_name",
            "gemma2",
            "--task_id",
            "999",
            "--num_examples",
            "0",
            "--n_runs",
            "1",
            "--temperature",
            "0",
            "--max_context_len",
            "max",
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

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("Command Output:", result.stdout)
            print("Command Error:", result.stderr)

        self.assertEqual(result.returncode, 0, "CLI command failed to execute.")

        time.sleep(10)

        self.assertTrue(
            self.expected_output_file.exists(), "Expected output file was not created."
        )

        with open(self.expected_output_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        self.assertTrue(len(predictions) > 0, "Output file contains no data.")

        for idx, prediction in enumerate(predictions):
            with self.subTest(index=idx):
                required_keys = {"HR", "Name", "status"}
                missing_keys = required_keys - prediction.keys()
                self.assertFalse(
                    missing_keys, f"Missing keys at index {idx}: {missing_keys}"
                )

            # If status is "success", compare actual vs. expected output
            if prediction.get("status") == "success":
                expected_output = prediction.get("expected_output", {})
                actual_output = {
                    k: prediction[k] for k in ["HR", "Name"] if k in prediction
                }
                self.assertEqual(
                    actual_output,
                    expected_output,
                    f"Mismatch in output at index {idx}: expected {expected_output}, got {actual_output}",
                )


if __name__ == "__main__":
    unittest.main()
