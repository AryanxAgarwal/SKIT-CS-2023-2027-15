import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestModelValidationEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.evaluator = cls.project_root / "evaluation" / "evaluator.py"
        cls.dataset = cls.project_root / "dataset.npz"
        cls.report = cls.project_root / "evaluation_results" / "validation_report.json"

    def run_evaluator(self):
        result = subprocess.run(
            [sys.executable, str(self.evaluator)],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"Evaluator failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        return result

    def load_report(self):
        self.assertTrue(
            self.report.exists(),
            "validation_report.json was not created"
        )

        with open(self.report, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_dataset_exists(self):
        self.assertTrue(
            self.dataset.exists(),
            "dataset.npz must exist"
        )

    def test_evaluator_runs_successfully(self):
        result = self.run_evaluator()

        self.assertIn(
            "MODEL VALIDATION ENGINE",
            result.stdout
        )

    def test_report_is_valid_json(self):
        self.run_evaluator()

        report = self.load_report()

        self.assertIsInstance(report, dict)

    def test_required_report_sections_exist(self):
        self.run_evaluator()

        report = self.load_report()

        required_sections = [
            "dataset",
            "model_input_validation",
            "execution_benchmark",
            "model_status",
            "evaluation_status"
        ]

        for section in required_sections:
            self.assertIn(
                section,
                report,
                f"Missing report section: {section}"
            )

    def test_dataset_information(self):
        self.run_evaluator()

        report = self.load_report()
        dataset = report["dataset"]

        self.assertEqual(dataset["samples"], 383)
        self.assertEqual(dataset["shape"], [383, 30, 7])

        self.assertEqual(
            dataset["labels"]["class_0"],
            92
        )

        self.assertEqual(
            dataset["labels"]["class_1"],
            291
        )

    def test_input_schema_mismatch_is_detected(self):
        self.run_evaluator()

        report = self.load_report()
        validation = report["model_input_validation"]

        self.assertEqual(
            validation["expected_shape"],
            [30, 8]
        )

        self.assertEqual(
            validation["actual_shape"],
            [30, 7]
        )

        self.assertTrue(
            validation["timesteps_match"]
        )

        self.assertFalse(
            validation["features_match"]
        )

        self.assertFalse(
            validation["compatible"]
        )

    def test_execution_benchmark(self):
        self.run_evaluator()

        report = self.load_report()
        benchmark = report["execution_benchmark"]

        self.assertEqual(
            benchmark["samples_processed"],
            383
        )

        self.assertGreaterEqual(
            benchmark["elapsed_seconds"],
            0
        )

        self.assertGreater(
            benchmark["samples_per_second"],
            0
        )

    def test_model_status_is_reported_honestly(self):
        self.run_evaluator()

        report = self.load_report()
        status = report["model_status"]

        self.assertFalse(
            status["trained_checkpoint"]
        )

        self.assertFalse(
            status["tensorflow_available"]
        )

        self.assertFalse(
            status["inference_evaluation"]
        )

    def test_evaluation_status(self):
        self.run_evaluator()

        report = self.load_report()
        status = report["evaluation_status"]

        self.assertTrue(
            status["dataset_validation"]
        )

        self.assertFalse(
            status["input_schema_validation"]
        )

        self.assertFalse(
            status["model_metrics_available"]
        )

        self.assertFalse(
            status["loss_curve_available"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)