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