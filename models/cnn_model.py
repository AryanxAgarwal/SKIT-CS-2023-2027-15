
"""
1D CNN for binary driver drowsiness classification.

Input shape: (30 frames, 8 features per frame)
Output: probability of drowsiness (0 to 1)
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


SEQUENCE_LENGTH = 30

FEATURE_COLUMNS = [
    "ear",
    "mar",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "gaze_vertical",
    "gaze_horizontal",
    "face_detected",
]

FEATURE_COUNT = len(FEATURE_COLUMNS)


def build_cnn_model(
    timesteps=SEQUENCE_LENGTH,
    features=FEATURE_COUNT,
    learning_rate=0.001,
):
    """Build and compile the CNN model."""

    if not isinstance(timesteps, int) or timesteps < 4:
        raise ValueError(
            "timesteps must be an integer of at least 4."
        )

    if features != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features per frame, "
            f"but received {features}."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    model = Sequential([
        Input(shape=(timesteps, features)),

        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same",
        ),
        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
        ),
        MaxPooling1D(pool_size=2),

        GlobalAveragePooling1D(),

        Dense(64, activation="relu"),
        Dropout(0.5),

        Dense(32, activation="relu"),
        Dropout(0.25),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()

    print("\nCNN architecture validation:")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Features per frame: {FEATURE_COUNT}")
    print(f"Expected input shape: (None, {SEQUENCE_LENGTH}, {FEATURE_COUNT})")
    print("Output: one drowsiness probability per sequence")
