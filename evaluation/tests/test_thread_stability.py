
import threading
import time
import unittest

from evaluation.runtime_threads import RuntimePipeline


class TestThreadStability(unittest.TestCase):

    def test_high_load_queue_remains_bounded(self):
        pipeline = RuntimePipeline(
            frame_queue_size=3,
            result_queue_size=3
        )

        produced = 0
        counter_lock = threading.Lock()
        producer_done = threading.Event()
        processed = threading.Event()

        def capture():
            nonlocal produced

            with counter_lock:
                if produced >= 500:
                    producer_done.set()
                    pipeline.stop_event.wait(timeout=5)
                    return None

                produced += 1
                return produced

        def inference(frame):
            time.sleep(0.001)
            processed.set()
            return frame * 2

        pipeline.start(capture, inference)

        try:
            self.assertTrue(
                producer_done.wait(timeout=5),
                "Capture did not produce 500 frames"
            )

            self.assertTrue(
                processed.wait(timeout=2),
                "Inference did not process any frame"
            )

            self.assertLessEqual(
                pipeline.frame_queue.qsize(), 3
            )

            self.assertLessEqual(
                pipeline.result_queue.qsize(), 3
            )

            self.assertEqual(produced, 500)

        finally:
            pipeline.stop()

    def test_repeated_start_stop_cycles(self):
        for cycle in range(10):
            pipeline = RuntimePipeline()

            def capture():
                time.sleep(0.005)
                return cycle

            pipeline.start(capture, lambda frame: frame)

            self.assertTrue(pipeline.stop())

            self.assertFalse(pipeline.capture_thread.is_alive())
            self.assertFalse(pipeline.inference_thread.is_alive())

    def test_worker_exception_does_not_leave_pipeline_running(self):
        pipeline = RuntimePipeline()

        def capture():
            raise RuntimeError("Stress test failure")

        pipeline.start(capture, lambda frame: frame)

        try:
            deadline = time.monotonic() + 2

            while not pipeline.stop_event.is_set():
                if time.monotonic() >= deadline:
                    self.fail("Worker did not report its failure")
                time.sleep(0.01)

            self.assertTrue(pipeline.get_errors())

        finally:
            self.assertTrue(pipeline.stop())

    def test_queue_size_never_exceeds_capacity(self):
        pipeline = RuntimePipeline(
            frame_queue_size=2,
            result_queue_size=2
        )

        for value in range(100):
            pipeline._put_latest(pipeline.frame_queue, value)

        self.assertLessEqual(pipeline.frame_queue.qsize(), 2)

        self.assertEqual(pipeline.frame_queue.get_nowait(), 98)
        self.assertEqual(pipeline.frame_queue.get_nowait(), 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
