import os

# Face Landmarker model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_LANDMARKER_MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_landmarker.task"
)

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)