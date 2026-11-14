
import threading
import time
import unittest

from evaluation.runtime_threads import RuntimePipeline
from evaluation.runtime_state_machine import (
    RuntimeStateMachine,
    SystemState,
)


class TestSystemIntegration(unittest.TestCase):

    def test_complete_frame_to_alert_workflow(self):
        pipeline = RuntimePipeline(
            frame_queue_size=5,
            result_queue_size=5
        )

        state_machine = RuntimeStateMachine()

        frame_count = 0
        processed_results = []
        result_received = threading.Event()

        def capture():
            nonlocal frame_count

            if frame_count >= 10:
                return None

            frame_count += 1

            return {
                "frame_id": frame_count,
                "eye_status": "closed"
            }

        def inference(frame):
            # Simulated inference result
            return {
                "frame_id": frame["frame_id"],
                "prediction": "DROWSY"
                if frame["eye_status"] == "closed"
                else "NORMAL"
            }

        state_machine.transition(SystemState.MONITORING)

        pipeline.start(capture, inference)

        try:
            deadline = time.monotonic() + 5

            while time.monotonic() < deadline:
                result = pipeline.get_result(timeout=0.1)

                if result is None:
                    if frame_count >= 10:
                        break
                    continue

                processed_results.append(result)

                if result["prediction"] == "DROWSY":
                    if state_machine.get_state() == "MONITORING":
                        state_machine.transition(SystemState.ALERT)

                result_received.set()

                if len(processed_results) >= 3:
                    break

            self.assertTrue(
                result_received.is_set(),
                "No inference result was received"
            )

            self.assertGreaterEqual(len(processed_results), 1)

            self.assertEqual(
                state_machine.get_state(),
                "ALERT"
            )

            self.assertTrue(
                all(
                    result["prediction"] == "DROWSY"
                    for result in processed_results
                )
            )

        finally:
            self.assertTrue(pipeline.stop())

        self.assertFalse(pipeline.capture_thread.is_alive())
        self.assertFalse(pipeline.inference_thread.is_alive())

    def test_error_state_on_inference_failure(self):
        pipeline = RuntimePipeline()
        state_machine = RuntimeStateMachine()

        state_machine.transition(SystemState.MONITORING)

        def capture():
            return {"frame_id": 1}

        def inference(frame):
            raise RuntimeError("Model inference failed")

        pipeline.start(capture, inference)

        try:
            deadline = time.monotonic() + 3

            while not pipeline.stop_event.is_set():
                if time.monotonic() >= deadline:
                    self.fail("Inference failure was not reported")
                time.sleep(0.01)

            errors = pipeline.get_errors()

            self.assertTrue(errors)
            self.assertEqual(errors[0]["stage"], "inference")

            state_machine.transition(SystemState.ERROR)

            self.assertEqual(
                state_machine.get_state(),
                "ERROR"
            )

        finally:
            pipeline.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
