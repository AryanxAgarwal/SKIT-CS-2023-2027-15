
"""
CNN model for binary driver drowsiness classification.
Input: 30 time steps x 7 features.
Output: probability of class 1.
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


def build_cnn_model(
    timesteps=30,
    features=7,
    learning_rate=0.001,
):
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
