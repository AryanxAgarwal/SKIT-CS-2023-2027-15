import os
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_features.csv"
)

BATCH_SIZE = 128


# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------

def data_loader(dataset_path, batch_size):
    """
    Load the dataset in batches instead of processing
    the complete dataset at once.
    """

    for chunk in pd.read_csv(
        dataset_path,
        chunksize=batch_size
    ):
        yield chunk


# ---------------------------------------------------------
# Streaming Validation
# ---------------------------------------------------------

def test_data_loader_streaming():

    assert os.path.exists(DATASET_PATH), (
        f"Dataset not found: {DATASET_PATH}"
    )

    total_rows = 0
    batch_count = 0

    for batch in data_loader(
        DATASET_PATH,
        BATCH_SIZE
    ):

        batch_count += 1
        total_rows += len(batch)

        # Verify that every batch contains data
        assert len(batch) > 0, (
            "Data loader returned an empty batch."
        )

        # Verify required columns
        required_columns = [
            "source_clip",
            "frame",
            "time_sec",
            "face_detected",
            "ear",
            "mar",
        ]

        for column in required_columns:
            assert column in batch.columns, (
                f"Missing column: {column}"
            )

        print(
            f"Batch {batch_count}: "
            f"{len(batch)} rows"
        )

    # Verify that the complete dataset was streamed
    original_rows = len(pd.read_csv(DATASET_PATH))

    assert total_rows == original_rows, (
        "Data loader did not process the complete dataset."
    )
    assert batch_count > 0, (
        "No batches were loaded."
    )

    print("\nTotal batches:", batch_count)
    print("Total rows streamed:", total_rows)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("========================================")
    print("DATA LOADER STREAMING VALIDATION")
    print("========================================")

    test_data_loader_streaming()

    print("\n========================================")
    print("DATA LOADER STREAMING TEST PASSED")
    print("========================================")