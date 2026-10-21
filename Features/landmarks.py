"""
Face landmark extraction using MediaPipe's FaceLandmarker (Tasks API).

Note: older MediaPipe tutorials use `mp.solutions.face_mesh` -- that API
was removed from the current PyPI package. This uses the current
Tasks API instead. Identical on laptop and Pi/Jetson; only difference
is the one-time model file download (see config.FACE_LANDMARKER_MODEL_URL).
"""

import math
import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config

# Landmark indices (same numbering as the old 468/478-point face mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33

# For gaze (iris-based) -- needs the eye top/bottom points plus the iris
# center, which is only present because refine/iris landmarks are included
# by default in the current FaceLandmarker (indices 468-477).
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 159, 145
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 386, 374
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473


def _ensure_model(path: str, url: str) -> bool:
    """Downloads the .task model file if missing. Returns True if the file
    is available afterward. Needs internet the first time only; safe to
    pre-download and copy onto an offline device (e.g. Pi with no wifi)."""
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        print(f"Downloading face landmark model to {path} (one-time, ~4MB)...")
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"Could not download face landmark model ({e}). "
              f"Face detection will be disabled -- download {url} manually "
              f"and place it at {path}, then re-run.")
        return False


class LandmarkExtractor:
    def __init__(self, model_path: str = config.FACE_LANDMARKER_MODEL_PATH,
                 model_url: str = config.FACE_LANDMARKER_MODEL_URL):
        self.available = _ensure_model(model_path, model_url)
        self.detector = None

        if self.available:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_rgb):
        """Returns a dict of features for the first detected face, or None
        (no face detected, or model unavailable)."""
        if not self.available:
            return None

        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm])

        ear = (self._eye_aspect_ratio(pts, LEFT_EYE) + self._eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
        mar = self._mouth_aspect_ratio(pts, MOUTH)
        pitch, yaw = self._head_pose(pts)
        roll = self._head_roll(pts)
        gaze_v, gaze_h = self._gaze(pts)

        return {
            "ear": ear, "mar": mar,
            "head_pitch": pitch, "head_yaw": yaw, "head_roll": roll,
            "gaze_vertical": gaze_v, "gaze_horizontal": gaze_h,
            "landmarks": pts,
        }

    @staticmethod
    def _eye_aspect_ratio(pts, idx):
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = 2.0 * np.linalg.norm(p1 - p4)
        return vertical / horizontal if horizontal > 0 else 0.0

    @staticmethod
    def _mouth_aspect_ratio(pts, idx):
        p1, p2, p3, p4, p5, p6, p7, p8 = [pts[i] for i in idx]
        vertical = np.linalg.norm(p2 - p8) + np.linalg.norm(p3 - p7) + np.linalg.norm(p4 - p6)
        horizontal = 3.0 * np.linalg.norm(p1 - p5)
        return vertical / horizontal if horizontal > 0 else 0.0

    @staticmethod
    def _head_pose(pts):
        """Lightweight pitch/yaw proxy from nose position relative to the
        eye/chin line (skips full solvePnP + camera calibration, which is
        overkill for a nod/turn signal)."""
        nose = pts[NOSE_TIP]
        chin = pts[CHIN]
        left_eye = pts[LEFT_EYE_CORNER]
        right_eye = pts[RIGHT_EYE_CORNER]

        eye_mid = (left_eye + right_eye) / 2.0
        dx, dy = nose - eye_mid
        face_height = np.linalg.norm(chin - eye_mid) or 1.0

        pitch = math.degrees(math.atan2(dy, face_height))
        yaw = math.degrees(math.atan2(dx, face_height))
        return pitch, yaw

    @staticmethod
    def _head_roll(pts):
        """Head tilt (sideways lean) -- angle of the line between the two
        eye corners relative to horizontal. 0 = level, positive/negative =
        tilted toward one shoulder (e.g. looking away from the road sideways
        while head stays roughly forward)."""
        left_eye = pts[LEFT_EYE_CORNER]
        right_eye = pts[RIGHT_EYE_CORNER]
        dx, dy = left_eye - right_eye
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _gaze(pts):
        """Where the iris sits inside the eye socket, independent of head
        pose -- catches 'head straight but eyes looking down at phone',
        which head_pitch alone would miss.
        gaze_vertical:   ~0.5 = centered, >0.5 = looking down, <0.5 = looking up
        gaze_horizontal: ~0.5 = centered, >0.5 = looking toward one side
        """
        r_top, r_bot = pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM]
        l_top, l_bot = pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM]
        r_iris, l_iris = pts[RIGHT_IRIS_CENTER], pts[LEFT_IRIS_CENTER]

        r_v = (r_iris[1] - r_top[1]) / (r_bot[1] - r_top[1] + 1e-6)
        l_v = (l_iris[1] - l_top[1]) / (l_bot[1] - l_top[1] + 1e-6)
        gaze_vertical = float(np.clip((r_v + l_v) / 2.0, 0.0, 1.0))

        r_eye_l, r_eye_r = pts[RIGHT_EYE_CORNER], pts[133]   # right eye corners
        l_eye_l, l_eye_r = pts[362], pts[LEFT_EYE_CORNER]     # left eye corners
        r_h = (r_iris[0] - r_eye_l[0]) / (r_eye_r[0] - r_eye_l[0] + 1e-6)
        l_h = (l_iris[0] - l_eye_l[0]) / (l_eye_r[0] - l_eye_l[0] + 1e-6)
        gaze_horizontal = float(np.clip((r_h + l_h) / 2.0, 0.0, 1.0))

        return gaze_vertical, gaze_horizontal

    def close(self):
        if self.detector:
            self.detector.close()