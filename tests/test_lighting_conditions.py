import os
import cv2
import numpy as np
import pandas as pd

from landmarks import LandmarkExtractor


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VIDEO_PATH = os.path.join(
    "ayush-labelling",
    "..",
    "normal.mp4"
)

OUTPUT_CSV = os.path.join(
    "tests",
    "lighting_test_results.csv"
)


# ---------------------------------------------------------
# Lighting transformations
# ---------------------------------------------------------

def normal_lighting(frame):
    return frame.copy()


def low_light(frame):
    """
    Simulates low-light conditions.
    """
    return cv2.convertScaleAbs(
        frame,
        alpha=0.45,
        beta=-10
    )


def bright_light(frame):
    """
    Simulates strong/bright illumination.
    """
    return cv2.convertScaleAbs(
        frame,
        alpha=1.4,
        beta=35
    )


def left_side_light(frame):
    """
    Simulates stronger illumination from the left side.
    """
    h, w = frame.shape[:2]

    gradient = np.tile(
        np.linspace(0.55, 1.20, w),
        (h, 1)
    )

    result = frame.astype(np.float32)

    for channel in range(3):
        result[:, :, channel] *= gradient

    return np.clip(result, 0, 255).astype(np.uint8)


def right_side_light(frame):
    """
    Simulates stronger illumination from the right side.
    """
    h, w = frame.shape[:2]

    gradient = np.tile(
        np.linspace(1.20, 0.55, w),
        (h, 1)
    )

    result = frame.astype(np.float32)

    for channel in range(3):
        result[:, :, channel] *= gradient

    return np.clip(result, 0, 255).astype(np.uint8)


def top_light(frame):
    """
    Simulates stronger illumination from the top.
    """
    h, w = frame.shape[:2]

    gradient = np.linspace(
        1.20,
        0.60,
        h
    ).reshape(h, 1)

    result = frame.astype(np.float32)

    for channel in range(3):
        result[:, :, channel] *= gradient

    return np.clip(result, 0, 255).astype(np.uint8)