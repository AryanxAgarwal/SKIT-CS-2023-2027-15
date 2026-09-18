import csv
import re
from pathlib import Path
from datetime import datetime


REQUIRED_COLUMNS = {
    "frame_id",
    "filename",
    "timestamp",
    "width",
    "height",
    "fps_at_capture",
}

FRAME_NAME_PATTERN = re.compile(
    r"^frame_\d{6}\.jpg$"
)


def validate_ingest(frames_dir, metadata_csv):
    """
    Validate captured frames and their metadata.

    Returns:
        list[str]: Validation errors.
        Empty list means all checks passed.
    """

    frames_dir = Path(frames_dir)
    metadata_csv = Path(metadata_csv)

    errors = []

    if not frames_dir.exists():
        errors.append(
            f"Frames directory not found: {frames_dir}"
        )
        return errors

    if not metadata_csv.exists():
        errors.append(
            f"Metadata CSV not found: {metadata_csv}"
        )
        return errors

    frame_files = sorted(frames_dir.glob("*.jpg"))

    try:
        with metadata_csv.open(
            newline="",
            encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                errors.append("CSV has no header.")
                return errors

            missing_columns = (
                REQUIRED_COLUMNS
                - set(reader.fieldnames)
            )

            if missing_columns:
                errors.append(
                    "Missing columns: "
                    + ", ".join(sorted(missing_columns))
                )
                return errors

            rows = list(reader)

    except Exception as exc:
        errors.append(
            f"Could not read metadata CSV: {exc}"
        )
        return errors

    metadata_filenames = []

    for row_number, row in enumerate(rows, start=2):
        filename = row["filename"]
        metadata_filenames.append(filename)

        if not FRAME_NAME_PATTERN.match(
            Path(filename).name
        ):
            errors.append(
                f"Row {row_number}: invalid filename "
                f"format: {filename}"
            )

        try:
            int(row["frame_id"])
        except ValueError:
            errors.append(
                f"Row {row_number}: frame_id must be integer."
            )

        try:
            datetime.fromisoformat(
                row["timestamp"].replace("Z", "+00:00")
            )
        except ValueError:
            errors.append(
                f"Row {row_number}: invalid timestamp."
            )

        try:
            if int(row["width"]) <= 0:
                raise ValueError

            if int(row["height"]) <= 0:
                raise ValueError

            if float(row["fps_at_capture"]) <= 0:
                raise ValueError

        except ValueError:
            errors.append(
                f"Row {row_number}: invalid dimensions or FPS."
            )

    actual_filenames = [
        frame.name for frame in frame_files
    ]

    missing_frames = set(metadata_filenames) - set(
        actual_filenames
    )

    extra_frames = set(actual_filenames) - set(
        metadata_filenames
    )

    if missing_frames:
        errors.append(
            "Missing frame files: "
            + ", ".join(sorted(missing_frames))
        )

    if extra_frames:
        errors.append(
            "Frames without metadata: "
            + ", ".join(sorted(extra_frames))
        )

    if len(metadata_filenames) != len(
        set(metadata_filenames)
    ):
        errors.append(
            "Duplicate filenames found in metadata."
        )

    return errors


if __name__ == "__main__":
    errors = validate_ingest(
        "dataset/frames",
        "dataset/metadata.csv"
    )

    if errors:
        print("Frame ingest validation FAILED")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("Frame ingest validation PASSED")
