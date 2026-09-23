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

# ---------------------------------------------------------
# Lighting conditions
# ---------------------------------------------------------

LIGHTING_CONDITIONS = {
    "normal": normal_lighting,
    "low_light": low_light,
    "bright_light": bright_light,
    "left_side_light": left_side_light,
    "right_side_light": right_side_light,
    "top_light": top_light,
}


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def run_lighting_test():

    if not os.path.exists(VIDEO_PATH):
        print("Video not found:")
        print(VIDEO_PATH)
        print()
        print("Change VIDEO_PATH to the location of one of your videos.")
        return

    extractor = LandmarkExtractor()

    if not extractor.available:
        print("Face landmark model is not available.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video:")
        print(VIDEO_PATH)
        extractor.close()
        return

    results = []

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Test every 5th frame to reduce processing time
        if frame_number % 5 != 0:
            frame_number += 1
            continue

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        for condition, transformation in LIGHTING_CONDITIONS.items():

            modified_frame = transformation(frame_rgb)

            features = extractor.process(
                modified_frame
            )

            if features is None:

                results.append({
                    "frame": frame_number,
                    "lighting_condition": condition,
                    "face_detected": 0,
                    "ear": np.nan,
                    "mar": np.nan,
                    "head_pitch": np.nan,
                    "head_yaw": np.nan,
                })

            else:

                results.append({
                    "frame": frame_number,
                    "lighting_condition": condition,
                    "face_detected": 1,
                    "ear": features["ear"],
                    "mar": features["mar"],
                    "head_pitch": features["head_pitch"],
                    "head_yaw": features["head_yaw"],
                })

        frame_number += 1

    cap.release()
    extractor.close()

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nLighting test completed.")
    print("Results saved to:")
    print(OUTPUT_CSV)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n===== LIGHTING TEST SUMMARY =====")

    summary = df.groupby(
        "lighting_condition"
    ).agg(
        frames=("frame", "count"),
        face_detection_rate=("face_detected", "mean"),
        mean_ear=("ear", "mean"),
        std_ear=("ear", "std"),
        mean_mar=("mar", "mean"),
        std_mar=("mar", "std"),
    )

    summary["face_detection_rate"] *= 100

    print(
        summary.round(4)
    )


if __name__ == "__main__":
    run_lighting_test()