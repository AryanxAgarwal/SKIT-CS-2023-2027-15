"""
Live Driver Drowsiness and Behaviour Monitoring  (v2)

Run from the project root:  python live_drowsiness.py
Keys: Q / ESC = quit,  C = recalibrate

What changed compared with v1
- Per-driver calibration of the eye-closure threshold (first 5 s, eyes open).
- Rules use SECONDS, not frame counts, so they work at any webcam FPS.
- PERCLOS (share of the last 60 s with eyes closed), the standard fatigue metric.
- Yawn events are counted over the last 10 minutes (one yawn alone != drowsy).
- Overall state: ALERT / CAUTION / DROWSY, with an audible alarm.
- "Driver not visible" warning when the face is lost for too long.
- CNN is optional and shown as a secondary signal (see notes on its validity).
- Events are written to logs/events.csv.

Prototype only - not safety-certified.
"""
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import winsound  # Windows only
except ImportError:
    winsound = None

try:
    from landmarks import LandmarkExtractor
except ImportError as exc:
    raise ImportError(
        "Could not import LandmarkExtractor from landmarks.py. "
        "Run this script from the project root."
    ) from exc

# ----------------------------------------------------------------- settings
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "cnn_drowsiness.keras"
LOG_PATH = PROJECT_DIR / "logs" / "events.csv"

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

USE_CNN = True              # set False to run rule-based only
SEQUENCE_LENGTH = 30
CNN_EVERY_N_FRAMES = 3      # run the CNN less often to save CPU
CNN_SMOOTHING = 5
CNN_DROWSY_THRESHOLD = 0.50

# Must match training order exactly.
FEATURE_NAMES = ("ear", "mar", "head_pitch", "head_yaw", "head_roll",
                 "gaze_vertical", "gaze_horizontal", "face_detected")

# Calibration
CALIB_SECONDS = 5.0
CALIB_MIN_SAMPLES = 30
EAR_CLOSED_RATIO = 0.72     # closed if EAR < 72 % of your open-eye median
EAR_CLAMP = (0.12, 0.25)
DEFAULT_EAR_CLOSED = 0.19   # used if calibration fails

# Rules (seconds)
MAR_YAWN_THRESHOLD = 0.50
EYE_CLOSED_SECONDS = 0.8    # sustained closure = microsleep warning
YAWN_SECONDS = 1.5
NO_FACE_SECONDS = 3.0
PERCLOS_WINDOW = 60.0
PERCLOS_MIN_COVERAGE = 20.0
PERCLOS_CAUTION = 0.12
PERCLOS_DROWSY = 0.20
YAWN_WINDOW = 600.0
YAWNS_FOR_CAUTION = 3

COL_OK, COL_WARN, COL_BAD = (0, 200, 0), (0, 165, 255), (0, 0, 255)
COL_TXT = (230, 230, 230)


# ------------------------------------------------------------------ helpers
def safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def features_to_vector(face_features):
    values = [safe_float(face_features.get(n, 0.0)) for n in FEATURE_NAMES[:-1]]
    values.append(1.0)
    return np.asarray(values, dtype=np.float32)


def draw_text(frame, text, pos, color=COL_TXT, scale=0.55, thick=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def cleanup_extractor(extractor):
    for name in ("close", "release"):
        method = getattr(extractor, name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            break


class Alarm:
    """Non-blocking beep (Windows) or terminal bell elsewhere."""

    def __init__(self):
        self._busy = False

    def beep(self):
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            if winsound:
                winsound.Beep(1800, 350)
            else:
                print("\a", end="", flush=True)
        finally:
            self._busy = False


class EventLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if new:
            self._writer.writerow(["time", "event", "ear", "mar", "perclos"])

    def write(self, event, ear, mar, perclos):
        self._writer.writerow([datetime.now().isoformat(timespec="seconds"),
                               event, f"{ear:.3f}", f"{mar:.3f}", f"{perclos:.3f}"])
        self._file.flush()

    def close(self):
        self._file.close()


def load_cnn():
    if not USE_CNN:
        return None
    if not MODEL_PATH.is_file():
        print(f"NOTE: CNN model not found ({MODEL_PATH}); running rule-based only.")
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        print(f"NOTE: could not load CNN ({exc}); running rule-based only.")
        return None
    shape = getattr(model, "input_shape", None)
    if shape is not None and len(shape) == 3:
        if shape[1] not in (None, SEQUENCE_LENGTH) or shape[2] not in (None, len(FEATURE_NAMES)):
            print(f"NOTE: CNN input shape {shape} does not match script; CNN disabled.")
            return None
    return model


# --------------------------------------------------------------------- main
def main():
    model = load_cnn()

    print("Loading facial landmark detector...")
    try:
        extractor = LandmarkExtractor()
    except Exception as exc:
        print(f"ERROR: Could not initialize LandmarkExtractor: {exc}")
        return 1

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    camera = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("ERROR: Could not open webcam. Close other camera apps or change CAMERA_INDEX.")
        cleanup_extractor(extractor)
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    alarm = Alarm()
    log = EventLog(LOG_PATH)

    # calibration state
    ear_closed_thr = DEFAULT_EAR_CLOSED
    calibrated = False
    calib_start = None
    calib_ears = []

    # runtime state
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    cnn_probs = deque(maxlen=CNN_SMOOTHING)
    perclos_win = deque()          # (time, closed_bool)
    yawn_times = deque()
    closed_since = None
    yawn_since = None
    yawn_counted = False
    no_face_since = None
    last_state = "ALERT"
    frame_idx = 0
    fps, prev_t = 0.0, time.time()

    print("Monitoring started. Q/ESC = quit, C = recalibrate.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame_idx += 1
            now = time.time()
            frame = cv2.flip(frame, 1)
            feats = extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            ear = mar = 0.0
            eye_warning = yawn_active = face_lost_warning = False

            if feats is None:
                sequence.append(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
                cnn_probs.clear()
                closed_since = yawn_since = None
                yawn_counted = False
                if no_face_since is None:
                    no_face_since = now
                face_lost_warning = (now - no_face_since) >= NO_FACE_SECONDS
            else:
                no_face_since = None
                ear = safe_float(feats.get("ear"))
                mar = safe_float(feats.get("mar"))
                sequence.append(features_to_vector(feats))

                # ---- calibration
                if not calibrated:
                    if calib_start is None:
                        calib_start = now
                    calib_ears.append(ear)
                    if now - calib_start >= CALIB_SECONDS:
                        if len(calib_ears) >= CALIB_MIN_SAMPLES:
                            median = float(np.median(calib_ears))
                            ear_closed_thr = float(np.clip(median * EAR_CLOSED_RATIO, *EAR_CLAMP))
                            print(f"Calibrated: open-eye EAR {median:.3f} -> closed threshold {ear_closed_thr:.3f}")
                        else:
                            print("Calibration had too few samples; using default threshold.")
                        calibrated = True
                else:
                    # ---- eye closure (time-based)
                    is_closed = ear < ear_closed_thr
                    perclos_win.append((now, is_closed))
                    if is_closed:
                        closed_since = closed_since or now
                        eye_warning = (now - closed_since) >= EYE_CLOSED_SECONDS
                    else:
                        closed_since = None

                    # ---- yawn (time-based, counted once per yawn)
                    if mar > MAR_YAWN_THRESHOLD:
                        yawn_since = yawn_since or now
                        if (now - yawn_since) >= YAWN_SECONDS:
                            yawn_active = True
                            if not yawn_counted:
                                yawn_times.append(now)
                                yawn_counted = True
                                log.write("YAWN", ear, mar, 0.0)
                    else:
                        yawn_since, yawn_counted = None, False

                # ---- CNN (secondary signal)
                if model is not None and len(sequence) == SEQUENCE_LENGTH \
                        and frame_idx % CNN_EVERY_N_FRAMES == 0:
                    x = np.expand_dims(np.asarray(sequence, dtype=np.float32), 0)
                    p = safe_float(np.asarray(model(x, training=False)).reshape(-1)[0])
                    cnn_probs.append(p)

            # ---- trim windows, compute PERCLOS
            while perclos_win and now - perclos_win[0][0] > PERCLOS_WINDOW:
                perclos_win.popleft()
            while yawn_times and now - yawn_times[0] > YAWN_WINDOW:
                yawn_times.popleft()
            perclos, perclos_ready = 0.0, False
            if len(perclos_win) > 10:
                coverage = perclos_win[-1][0] - perclos_win[0][0]
                perclos_ready = coverage >= PERCLOS_MIN_COVERAGE
                perclos = sum(c for _, c in perclos_win) / len(perclos_win)

            cnn_smoothed = float(np.mean(cnn_probs)) if cnn_probs else None
            cnn_drowsy = cnn_smoothed is not None and cnn_smoothed >= CNN_DROWSY_THRESHOLD

            # ---- overall state
            reasons = []
            state = "ALERT"
            if eye_warning:
                state = "DROWSY"; reasons.append("eyes closed")
            if perclos_ready and perclos >= PERCLOS_DROWSY:
                state = "DROWSY"; reasons.append("high PERCLOS")
            if face_lost_warning:
                state = "DROWSY"; reasons.append("driver not visible")
            if state != "DROWSY":
                if yawn_active:
                    state = "CAUTION"; reasons.append("yawning")
                if len(yawn_times) >= YAWNS_FOR_CAUTION:
                    state = "CAUTION"; reasons.append("frequent yawns")
                if perclos_ready and perclos >= PERCLOS_CAUTION:
                    state = "CAUTION"; reasons.append("elevated PERCLOS")
                if cnn_drowsy:
                    state = "CAUTION"; reasons.append("CNN drowsy")

            if state == "DROWSY":
                alarm.beep()
            if state != last_state:
                log.write(f"STATE {state}: {', '.join(reasons) or 'recovered'}", ear, mar, perclos)
                last_state = state

            # ---- FPS
            dt = now - prev_t
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            prev_t = now

            # ---- overlay
            color = {"ALERT": COL_OK, "CAUTION": COL_WARN, "DROWSY": COL_BAD}[state]
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 40), (0, 0, 0), -1)
            draw_text(frame, f"STATE: {state}", (12, 28), color, 0.8, 2)
            if reasons:
                draw_text(frame, ", ".join(reasons), (230, 28), color, 0.5)

            if not calibrated and feats is not None:
                left = max(0.0, CALIB_SECONDS - (now - (calib_start or now)))
                draw_text(frame, f"CALIBRATING {left:.0f}s - eyes open, look ahead",
                          (12, 75), (255, 255, 0), 0.6, 2)
            elif feats is None:
                draw_text(frame, "FACE NOT DETECTED", (12, 75), COL_WARN, 0.7, 2)
            else:
                draw_text(frame, f"EAR {ear:.3f} (closed < {ear_closed_thr:.3f})", (12, 70))
                draw_text(frame, f"MAR {mar:.3f} (yawn > {MAR_YAWN_THRESHOLD:.2f})", (12, 92))
                held = (now - closed_since) if closed_since else 0.0
                draw_text(frame, f"Eyes closed: {held:.1f}s / {EYE_CLOSED_SECONDS:.1f}s", (12, 114))
                draw_text(frame, f"Yawns (10 min): {len(yawn_times)}", (12, 136))
                p_txt = f"{perclos * 100:.0f}%" if perclos_ready else "warming up"
                draw_text(frame, f"PERCLOS (60 s): {p_txt}", (12, 158))
                if model is not None:
                    if cnn_smoothed is None:
                        draw_text(frame, f"CNN: collecting {len(sequence)}/{SEQUENCE_LENGTH}", (12, 180), (255, 255, 0))
                    else:
                        draw_text(frame, f"CNN: {'DROWSY' if cnn_drowsy else 'ALERT'} ({cnn_smoothed:.2f})",
                                  (12, 180), COL_BAD if cnn_drowsy else COL_OK)
            draw_text(frame, f"FPS {fps:.0f}", (12, frame.shape[0] - 14), (200, 200, 200), 0.45)

            cv2.imshow("Driver Drowsiness Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("c"), ord("C")):
                calibrated, calib_start, calib_ears = False, None, []
                perclos_win.clear()
                closed_since = None
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        cleanup_extractor(extractor)
        log.close()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())