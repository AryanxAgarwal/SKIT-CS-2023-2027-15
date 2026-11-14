
import threading
import time
import unittest

from evaluation.runtime_threads import RuntimePipeline


class TestRuntimePipeline(unittest.TestCase):

    def test_capture_and_inference_run_on_separate_threads(self):
        pipeline = RuntimePipeline()

        capture_thread_ids = []
        inference_thread_ids = []
        received = threading.Event()

        def capture():
            capture_thread_ids.append(threading.get_ident())
            time.sleep(0.01)
            return "frame"

        def inference(frame):
            inference_thread_ids.append(threading.get_ident())
            received.set()
            return "prediction"

        pipeline.start(capture, inference)

        try:
            self.assertTrue(received.wait(timeout=2))
            self.assertNotEqual(
                capture_thread_ids[0],
                inference_thread_ids[0]
            )
        finally:
            self.assertTrue(pipeline.stop())

    def test_result_is_available(self):
        pipeline = RuntimePipeline()
        processed = threading.Event()

        def capture():
            time.sleep(0.02)
            return "frame"

        def inference(frame):
            processed.set()
            return frame + "_processed"

        pipeline.start(capture, inference)

        try:
            self.assertTrue(processed.wait(timeout=2))
            result = pipeline.get_result(timeout=2)
            self.assertEqual(result, "frame_processed")
        finally:
            pipeline.stop()

    def test_capture_error_is_reported(self):
        pipeline = RuntimePipeline()

        def capture():
            raise RuntimeError("Camera failure")

        pipeline.start(capture, lambda frame: frame)

        try:
            deadline = time.monotonic() + 2

            while not pipeline.stop_event.is_set():
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)

            errors = pipeline.get_errors()

            self.assertTrue(errors)
            self.assertEqual(errors[0]["stage"], "capture")
            self.assertIn("Camera failure", errors[0]["error"])
        finally:
            pipeline.stop()

    def test_inference_error_is_reported(self):
        pipeline = RuntimePipeline()
        frame_ready = threading.Event()

        def capture():
            frame_ready.set()
            return "frame"

        def inference(frame):
            raise RuntimeError("Inference failure")

        pipeline.start(capture, inference)

        try:
            deadline = time.monotonic() + 2

            while not pipeline.stop_event.is_set():
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)

            errors = pipeline.get_errors()

            self.assertTrue(errors)
            self.assertEqual(errors[0]["stage"], "inference")
            self.assertIn("Inference failure", errors[0]["error"])
        finally:
            pipeline.stop()

    def test_none_frame_does_not_busy_spin(self):
        pipeline = RuntimePipeline()
        capture_calls = []
        first_call = threading.Event()

        def capture():
            capture_calls.append(time.monotonic())
            first_call.set()
            return None

        pipeline.start(capture, lambda frame: frame)

        try:
            self.assertTrue(first_call.wait(timeout=2))
            time.sleep(0.12)

            self.assertLessEqual(
                len(capture_calls), 20,
                "Capture worker may be busy-spinning on None frames"
            )
        finally:
            self.assertTrue(pipeline.stop())

    def test_invalid_queue_size_rejected(self):
        with self.assertRaises(ValueError):
            RuntimePipeline(frame_queue_size=0)

    def test_invalid_worker_rejected(self):
        pipeline = RuntimePipeline()

        with self.assertRaises(ValueError):
            pipeline.start("not a function", lambda frame: frame)

    def test_pipeline_cannot_start_twice(self):
        pipeline = RuntimePipeline()

        def capture():
            time.sleep(0.02)
            return None

        pipeline.start(capture, lambda frame: frame)

        try:
            with self.assertRaises(RuntimeError):
                pipeline.start(capture, lambda frame: frame)
        finally:
            pipeline.stop()

    def test_clean_shutdown(self):
        pipeline = RuntimePipeline()

        def capture():
            time.sleep(0.01)
            return "frame"

        pipeline.start(capture, lambda frame: frame)

        self.assertTrue(pipeline.stop())
        self.assertFalse(pipeline.capture_thread.is_alive())
        self.assertFalse(pipeline.inference_thread.is_alive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
