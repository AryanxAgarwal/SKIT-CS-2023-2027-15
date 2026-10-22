
import argparse
from pathlib import Path

import pandas as pd


CLIP_LABELS = {
    "normal.mp4": ("alert", 0),
    "eyes_closed.mp4": ("microsleep", 1),
    "yawn.mp4": ("yawn", 1),
    "head_nod.mp4": ("head_nod", 1),
    "phone_glance.mp4": ("phone_glance", 1),
    "head_tilt_left.mp4": ("head_tilt", 1),
    "head_tilt_right.mp4": ("head_tilt", 1),
}


def normalize_face_detected(value):
    """Convert common CSV representations to True or False."""
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no", ""}:
        return False

    raise ValueError(
        f"Invalid face_detected value: {value!r}"
    )


def label_features(input_path, output_path):
    df = pd.read_csv(input_path)

    required_columns = {"source_clip", "face_detected"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Normalize video filenames and face-detection values.
    df["source_clip"] = (
        df["source_clip"].astype(str).str.strip().str.replace(
            "\\", "/", regex=False
        ).str.rsplit("/", n=1).str[-1]
    )

    df["face_detected"] = df["face_detected"].map(
        normalize_face_detected
    )

    # Labels come from the annotated source clip.
    # Face visibility is a separate property.
    df["behavior_label"] = df["source_clip"].map(
        lambda clip: CLIP_LABELS.get(
            clip, ("unknown", None)
        )[0]
    )

    df["drowsy"] = df["source_clip"].map(
        lambda clip: CLIP_LABELS.get(
            clip, ("unknown", None)
        )[1]
    )

    df["visibility_label"] = df["face_detected"].map(
        {
            True: "face_detected",
            False: "face_not_detected",
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output file: {output_path}")
    print("\nBehavior-label counts:")
    print(df["behavior_label"].value_counts(dropna=False))
    print("\nDrowsiness-label counts:")
    print(df["drowsy"].value_counts(dropna=False))
    print("\nFace-visibility counts:")
    print(df["visibility_label"].value_counts(dropna=False))

    unknown_clips = sorted(
        set(df.loc[df["drowsy"].isna(), "source_clip"])
    )

    if unknown_clips:
        print("\nWARNING: These clips need labels before training:")
        for clip in unknown_clips:
            print(f"  - {clip}")


def main():
    parser = argparse.ArgumentParser(
        description="Map source clips to behavior and drowsiness labels."
    )
    parser.add_argument(
        "--in", dest="input_path",
        default="raw_features.csv",
        help="Input feature CSV"
    )
    parser.add_argument(
        "--out", dest="output_path",
        default="ayush-labelling/labeled_features.csv",
        help="Output labeled CSV"
    )

    args = parser.parse_args()
    label_features(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
