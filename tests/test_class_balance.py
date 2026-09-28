import os
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_features.csv"
)


# ---------------------------------------------------------
# Class Balance Validation
# ---------------------------------------------------------

def test_class_balance():

    assert os.path.exists(DATASET_PATH), (
        f"Dataset not found: {DATASET_PATH}"
    )

    df = pd.read_csv(DATASET_PATH)

    # Count samples belonging to each source class
    class_counts = df["source_clip"].value_counts()

    # Calculate class percentages
    class_percentages = (
        df["source_clip"]
        .value_counts(normalize=True)
        .mul(100)
    )

    print("\n===== CLASS BALANCE =====")

    for class_name in class_counts.index:
        print(
            f"{class_name}: "
            f"{class_counts[class_name]} samples "
            f"({class_percentages[class_name]:.2f}%)"
        )

    # Verify that at least two classes are present
    assert len(class_counts) >= 2, (
        "Dataset contains fewer than two classes."
    )

    # Verify that every class has samples
    assert (class_counts > 0).all(), (
        "One or more classes contain zero samples."
    )

    print("\nClass balance validation completed successfully.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("========================================")
    print("CLASS BALANCE VALIDATION")
    print("========================================")

    test_class_balance()

    print("\n========================================")
    print("CLASS BALANCE TEST PASSED")
    print("========================================")