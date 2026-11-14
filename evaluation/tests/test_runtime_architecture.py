
import unittest
import threading
import time
from unittest.mock import patch

import live_drowsiness as runtime


class TestRuntimeHelpers(unittest.TestCase):

    def test_safe_float_valid_value(self):
        self.assertEqual(runtime.safe_float(0.75), 0.75)

    def test_safe_float_invalid_value(self):
        self.assertEqual(runtime.safe_float("invalid"), 0.0)

    def test_safe_float_non_finite_value(self):
        self.assertEqual(runtime.safe_float(float("nan")), 0.0)

    def test_feature_vector_has_eight_features(self):
        features = {
            "ear": 0.25,
            "mar": 0.40,
            "head_pitch": 1.0,
            "head_yaw": 2.0,
            "head_roll": 3.0,
            "gaze_vertical": 0.1,
            "gaze_horizontal": 0.2
        }

        vector = runtime.features_to_vector(features)

        self.assertEqual(vector.shape, (8,))
        self.assertEqual(vector.dtype.name, "float32")
        self.assertEqual(vector[-1], 1.0)


class TestAlarmThread(unittest.TestCase):

    def test_alarm_runs_in_background_and_prevents_duplicates(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        class FakeSound:
            @staticmethod
            def Beep(frequency, duration):
                calls.append((frequency, duration))
                started.set()
                release.wait(timeout=3)

        alarm = runtime.Alarm()

        try:
            with patch.object(runtime, "winsound", FakeSound):
                start = time.perf_counter()
                alarm.beep()
                elapsed = time.perf_counter() - start

                self.assertLess(elapsed, 0.5)
                self.assertTrue(started.wait(timeout=2))

                alarm.beep()
                self.assertEqual(len(calls), 1)

        finally:
            release.set()

        deadline = time.monotonic() + 3
        while alarm._busy and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(alarm._busy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
