
import csv
import tempfile
import unittest
from pathlib import Path

from frame_ingest_validator import validate_ingest


class TestFrameIngestValidator(unittest.TestCase):

    def create_valid_dataset(self, temp_dir):
        root = Path(temp_dir)
        frames_dir = root / "frames"
        frames_dir.mkdir()

        metadata_csv = root / "metadata.csv"

        frame_names = [
            "frame_000001.jpg",
            "frame_000002.jpg",
        ]

        for name in frame_names:
            (frames_dir / name).touch()

        with metadata_csv.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "frame_id",
                    "filename",
                    "timestamp",
                    "width",
                    "height",
                    "fps_at_capture",
                ],
            )

            writer.writeheader()

            for index, name in enumerate(
                frame_names,
                start=1
            ):
                writer.writerow({
                    "frame_id": index,
                    "filename": name,
                    "timestamp": (
                        "2026-09-18T10:00:00"
                    ),
                    "width": 640,
                    "height": 480,
                    "fps_at_capture": 30.0,
                })

        return frames_dir, metadata_csv


class TestValidDataset(TestFrameIngestValidator):

    def test_valid_dataset_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir, metadata_csv = (
                self.create_valid_dataset(temp_dir)
            )

            errors = validate_ingest(
                frames_dir,
                metadata_csv
            )

            self.assertEqual(errors, [])


class TestInvalidDataset(TestFrameIngestValidator):

    def test_missing_frame_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir, metadata_csv = (
                self.create_valid_dataset(temp_dir)
            )

            (frames_dir / "frame_000002.jpg").unlink()

            errors = validate_ingest(
                frames_dir,
                metadata_csv
            )

            self.assertTrue(
                any(
                    "Missing frame files" in error
                    for error in errors
                )
            )

    def test_extra_frame_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir, metadata_csv = (
                self.create_valid_dataset(temp_dir)
            )

            (frames_dir / "frame_000003.jpg").touch()

            errors = validate_ingest(
                frames_dir,
                metadata_csv
            )

            self.assertTrue(
                any(
                    "Frames without metadata" in error
                    for error in errors
                )
            )


if __name__ == "__main__":
    unittest.main()
