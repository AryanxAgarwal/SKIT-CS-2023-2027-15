"""
Training-time data augmentation for the windowed dataset.

Reads dataset.npz (built by build_dataset.py) and writes a NEW file,
dataset_augmented.npz. dataset.npz itself is never modified.

What it does, in this order:
  1. Splits the windows chronologically per clip into train / validation
     (same rule as train_classifier.py: first 75% train, a gap of 3 windows,
     then validation). The split happens BEFORE augmenting, so no augmented
     copy of a training window can ever leak into validation.
  2. Augments ONLY the training windows:
       - mirror : negate head_yaw and head_roll, flip gaze_horizontal to
                  1 - value. A left tilt becomes a right tilt (and back).
       - jitter : small Gaussian noise on every feature; half of the jittered
                  copies are also mirrored. Optionally (--pitch-offset N) one
                  random pitch shift of +/-N degrees per window. The default is
                  0 because the camera is always mounted in the same place
                  (behind the steering wheel): a pitch shift would only teach
                  the model to ignore head dips, which is what we want to detect.
  3. Leaves validation windows untouched (real data only).

Frames that are all zeros (the "no face detected" sentinel) are never
mirrored, shifted or noised -- they stay exact zeros.

This runs ONCE, offline, before training. It is not part of deployment: in
the car the model sees real, unmodified features.

Usage:
    python augment_dataset.py
    python augment_dataset.py --jitter-copies 3 --seed 7
    python augment_dataset.py --pitch-offset 3      # only if you want some pitch tolerance
Output:
    dataset_augmented.npz with X_train, y_train, clip_train, aug_type,
    X_val, y_val, clip_val, features
"""

import argparse
from collections import Counter

import numpy as np

FEATURES = ["ear", "mar", "head_pitch", "head_yaw", "head_roll", "gaze_vertical", "gaze_horizontal"]
IDX = {name: i for i, name in enumerate(FEATURES)}

# Same sane bounds as build_dataset.py -- augmented values are clipped back in.
FEATURE_BOUNDS = {
    "ear": (0.0, 0.5),
    "mar": (0.0, 1.2),
    "head_pitch": (-60.0, 60.0),
    "head_yaw": (-60.0, 60.0),
    "head_roll": (-60.0, 60.0),
    "gaze_vertical": (0.0, 1.0),
    "gaze_horizontal": (0.0, 1.0),
}

# Per-feature noise (standard deviation), chosen to be about the size of
# natural frame-to-frame jitter in your recordings -- small on purpose.
NOISE_STD = {
    "ear": 0.01, "mar": 0.01, "head_pitch": 1.0, "head_yaw": 1.0,
    "head_roll": 1.0, "gaze_vertical": 0.02, "gaze_horizontal": 0.02,
}

PITCH_OFFSET_DEG = 0.0   # set from --pitch-offset; 0 = off (fixed camera position)


def chronological_split(clip_source, train_frac=0.75, gap=3):
    """Per-clip split by time order. Must match train_classifier.chronological_split."""
    clip_source = np.asarray(clip_source)
    train_mask = np.zeros(len(clip_source), dtype=bool)
    val_mask = np.zeros(len(clip_source), dtype=bool)
    for clip in np.unique(clip_source):
        idx = np.where(clip_source == clip)[0]  # already in temporal order
        n_train = int(len(idx) * train_frac)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train + gap:]] = True
    return train_mask, val_mask


def _real_frames(window):
    """Boolean mask (n_frames,) of frames that have real measurements,
    i.e. NOT the all-zero 'no face' sentinel."""
    return np.abs(window).sum(axis=1) > 0


def mirror(window):
    """Left <-> right. yaw and roll change sign; gaze_horizontal flips to 1 - g."""
    out = window.copy()
    real = _real_frames(window)
    out[real, IDX["head_yaw"]] *= -1
    out[real, IDX["head_roll"]] *= -1
    out[real, IDX["gaze_horizontal"]] = 1.0 - out[real, IDX["gaze_horizontal"]]
    return out


def jitter(window, rng):
    """Small noise on every feature + one random pitch offset for the window."""
    out = window.copy()
    real = _real_frames(window)
    n_real = int(real.sum())
    if n_real == 0:
        return out
    for name, std in NOISE_STD.items():
        out[real, IDX[name]] += rng.normal(0.0, std, n_real)
    out[real, IDX["head_pitch"]] += rng.uniform(-PITCH_OFFSET_DEG, PITCH_OFFSET_DEG)
    for name, (lo, hi) in FEATURE_BOUNDS.items():
        out[real, IDX[name]] = np.clip(out[real, IDX[name]], lo, hi)
    return out


def augment_training_set(X, y, clips, rng, jitter_copies=2):
    """Returns (X_aug, y_aug, clips_aug, aug_type). Includes the originals."""
    Xs, ys, cs, kinds = [], [], [], []
    for w, label, clip in zip(X, y, clips):
        Xs.append(w); ys.append(label); cs.append(clip); kinds.append("original")

        Xs.append(mirror(w)); ys.append(label); cs.append(clip); kinds.append("mirror")

        for k in range(jitter_copies):
            aug = jitter(w, rng)
            if k % 2 == 1:           # every second jittered copy is also mirrored
                aug = mirror(aug)
            Xs.append(aug); ys.append(label); cs.append(clip); kinds.append("jitter")

    return (np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64),
            np.array(cs), np.array(kinds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="dataset.npz")
    parser.add_argument("--out", dest="out_path", default="dataset_augmented.npz")
    parser.add_argument("--jitter-copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pitch-offset", type=float, default=0.0,
                        help="random pitch shift in degrees (+/-); keep 0 for a fixed camera mount")
    args = parser.parse_args()

    global PITCH_OFFSET_DEG
    PITCH_OFFSET_DEG = args.pitch_offset

    d = np.load(args.in_path, allow_pickle=True)
    X, y, clips = d["X"], d["y"], np.asarray(d["clip_source"])
    assert list(d["features"]) == FEATURES, "feature order in dataset.npz differs from this script"

    train_mask, val_mask = chronological_split(clips)
    rng = np.random.default_rng(args.seed)

    X_tr, y_tr, c_tr, kinds = augment_training_set(
        X[train_mask], y[train_mask], clips[train_mask], rng, args.jitter_copies)

    np.savez(args.out_path,
             X_train=X_tr, y_train=y_tr, clip_train=c_tr, aug_type=kinds,
             X_val=X[val_mask], y_val=y[val_mask], clip_val=clips[val_mask],
             features=np.array(FEATURES))

    n_orig_train = int(train_mask.sum())
    print(f"Original windows: {len(X)}  ->  train {n_orig_train}, validation {int(val_mask.sum())} "
          f"(+{len(X) - n_orig_train - int(val_mask.sum())} gap windows dropped to prevent overlap)")
    print(f"Training set after augmentation: {len(X_tr)} windows "
          f"({len(X_tr) / n_orig_train:.1f}x)  {dict(Counter(kinds))}")
    print(f"Class balance (train): alert={int((y_tr == 0).sum())}  not_alert={int((y_tr == 1).sum())}")
    print(f"Validation windows left untouched: {int(val_mask.sum())}")
    print(f"Wrote {args.out_path}  (original {args.in_path} not modified)")


if __name__ == "__main__":
    main()
