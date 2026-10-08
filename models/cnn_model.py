"""
Basic CNN architecture for driver drowsiness detection.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D


def build_cnn_model(timesteps=30, features=8):
    model = Sequential([
        Input(shape=(timesteps, features)),

        # First convolution block
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        # Second convolution block
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),
    ])

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()


"""
Commit 2: Add dense fully connected layers.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
)


def build_cnn_model(timesteps=30, features=8):
    model = Sequential([
        # Input layer
        Input(shape=(timesteps, features)),

        # Convolutional layers
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        # Feature aggregation
        GlobalAveragePooling1D(),

        # Fully connected layers
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
    ])

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()


"""
Commit 3: Add dropout regularization.
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


def build_cnn_model(timesteps=30, features=8):
    model = Sequential([
        # Input layer
        Input(shape=(timesteps, features)),

        # Convolutional layers
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        # Feature aggregation
        GlobalAveragePooling1D(),

        # Fully connected layers
        Dense(64, activation="relu"),
        Dropout(0.5),

        Dense(32, activation="relu"),
        Dropout(0.25),
    ])

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()


"""
Commit 4: Add output layer, compile, and validate model.
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
    features=8,
    learning_rate=0.001,
):
    model = Sequential([
        # Input layer
        Input(shape=(timesteps, features)),

        # Convolutional layers
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        MaxPooling1D(pool_size=2),

        # Feature aggregation
        GlobalAveragePooling1D(),

        # Fully connected layers
        Dense(64, activation="relu"),
        Dropout(0.5),

        Dense(32, activation="relu"),
        Dropout(0.25),

        # Binary classification output
        Dense(1, activation="sigmoid"),
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()