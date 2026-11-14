
import queue
import threading
import logging


class RuntimePipeline:
    """
    Independent producer-consumer pipeline for:
    1. Video frame capture
    2. Model inference
    3. UI result consumption

    Camera and UI hardware are supplied by the application.
    UI results are consumed by the main application thread.
    """

    def __init__(self, frame_queue_size=5, result_queue_size=5):
        if frame_queue_size < 1 or result_queue_size < 1:
            raise ValueError("Queue sizes must be positive")

        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.result_queue = queue.Queue(maxsize=result_queue_size)

        self.stop_event = threading.Event()
        self.error_queue = queue.Queue()

        self.capture_thread = None
        self.inference_thread = None

        self.logger = logging.getLogger(__name__)

    def _report_error(self, stage, error):
        self.logger.error("%s worker failed: %s", stage, error)

        self.error_queue.put({
            "stage": stage,
            "error": str(error)
        })

        self.stop_event.set()

    def _put_latest(self, target_queue, item):
        """Keep the newest item when a bounded queue is full."""
        try:
            target_queue.put_nowait(item)

        except queue.Full:
            try:
                target_queue.get_nowait()
                target_queue.task_done()
            except queue.Empty:
                pass

            try:
                target_queue.put_nowait(item)
            except queue.Full:
                self.logger.warning("Queue remained full; item dropped")

    def _capture_worker(self, capture_function):
        try:
            while not self.stop_event.is_set():
                frame = capture_function()

                if frame is None:
                    # Avoid busy-spinning when no frame is available.
                    self.stop_event.wait(0.01)
                    continue

                self._put_latest(self.frame_queue, frame)

        except Exception as error:
            self._report_error("capture", error)

    def _inference_worker(self, inference_function):
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)

                except queue.Empty:
                    continue

                try:
                    result = inference_function(frame)

                    if result is not None:
                        self._put_latest(self.result_queue, result)

                finally:
                    self.frame_queue.task_done()

        except Exception as error:
            self._report_error("inference", error)

    def start(self, capture_function, inference_function):
        if self.capture_thread or self.inference_thread:
            raise RuntimeError("Pipeline has already been started")

        if not callable(capture_function) or not callable(inference_function):
            raise ValueError("Worker functions must be callable")

        self.stop_event.clear()

        self.capture_thread = threading.Thread(
            target=self._capture_worker,
            args=(capture_function,),
            name="VideoCaptureWorker",
            daemon=True
        )

        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            args=(inference_function,),
            name="ModelInferenceWorker",
            daemon=True
        )

        self.capture_thread.start()
        self.inference_thread.start()

    def get_result(self, timeout=0.1):
        try:
            result = self.result_queue.get(timeout=timeout)
            self.result_queue.task_done()
            return result

        except queue.Empty:
            return None

    def stop(self, timeout=2.0):
        self.stop_event.set()

        for worker in (self.capture_thread, self.inference_thread):
            if worker and worker.is_alive():
                worker.join(timeout=timeout)

        return not any(
            worker and worker.is_alive()
            for worker in (self.capture_thread, self.inference_thread)
        )

    def get_errors(self):
        errors = []

        while True:
            try:
                errors.append(self.error_queue.get_nowait())

            except queue.Empty:
                break

        return errors
