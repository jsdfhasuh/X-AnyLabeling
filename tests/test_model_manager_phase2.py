import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
from PyQt5 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.prediction_job import (
    PredictionRequest,
)
from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_until(predicate, timeout_ms=2000):
    app = _application()
    deadline = time.monotonic() + (timeout_ms / 1000)
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    return predicate()


class _Shape:
    def __init__(self, label="person"):
        self.label = label

    def to_dict(self):
        return {
            "label": self.label,
            "points": [[1, 2], [3, 4]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }


class _FakeModel:
    def __init__(self, result=None, error=None, release=None):
        self.result = result or AutoLabelingResult([_Shape()], True, "model")
        self.error = error
        self.release = release
        self.calls = []
        self.unload_calls = 0
        self.model_ids = []
        self.tasks = []

    def predict_shapes(self, image, filename=None, **kwargs):
        self.calls.append((image, filename, kwargs))
        if self.release is not None:
            self.release.wait(1)
        if self.error is not None:
            raise self.error
        return self.result

    def unload(self):
        self.unload_calls += 1

    def set_model_id(self, model_id):
        self.model_ids.append(model_id)

    def set_task(self, task_id):
        self.tasks.append(task_id)


class _DownloadState:
    def __init__(self, running=True):
        self.running = running

    def isRunning(self):
        return self.running


class ModelManagerPhase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _application()

    def setUp(self):
        with mock.patch.object(ModelManager, "load_model_configs"):
            self.manager = ModelManager()
        self.model = _FakeModel()
        self.manager.loaded_model_config = {
            "type": "remote_server",
            "model": self.model,
        }

    def tearDown(self):
        thread = self.manager.model_execution_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(2000)

    def test_private_core_requires_active_token_and_returns_plain_payload(
        self,
    ):
        with self.assertRaisesRegex(RuntimeError, "inference_lease_invalid"):
            self.manager._predict_payload_unleased(None, object())
        token = self.manager.inference_lease.acquire("TEST", "owner")
        payload = self.manager._predict_payload_unleased(token, object())
        self.assertEqual(payload.shapes[0]["label"], "person")
        self.assertIs(payload.replace, True)
        token.release()

    def test_blocking_single_vl_and_runner_share_one_lease(self):
        runner_token = self.manager.inference_lease.acquire(
            "PREDICTION_RUNNER", "runner"
        )
        self.assertFalse(self.manager.predict_shapes(object(), batch=True))
        self.assertFalse(
            self.manager.predict_shapes(
                object(), text_prompt="describe", batch=True
            )
        )
        self.assertEqual(self.model.calls, [])
        runner_token.release()

        result = self.manager.predict_shapes(object(), batch=True)
        self.assertIsInstance(result, AutoLabelingResult)
        self.assertEqual(len(self.model.calls), 1)
        self.assertFalse(self.manager.inference_lease.is_active)

    def test_blocking_model_exception_is_not_an_empty_success(self):
        self.model.error = RuntimeError("model boom")
        with self.assertRaisesRegex(RuntimeError, "model boom"):
            self.manager.predict_shapes(object(), batch=True)
        self.assertFalse(self.manager.inference_lease.is_active)

    def test_legacy_empty_and_vl_results_keep_signal_contract(self):
        self.model.result = AutoLabelingResult([], False, "empty")
        results = []
        started = []
        finished = []
        self.manager.new_auto_labeling_result.connect(results.append)
        self.manager.prediction_started.connect(lambda: started.append(True))
        self.manager.prediction_finished.connect(lambda: finished.append(True))

        self.assertTrue(
            self.manager.predict_shapes_threading(
                object(), "target.jpg", text_prompt="describe"
            )
        )
        self.assertTrue(_wait_until(lambda: len(finished) == 1))
        self.assertEqual(
            (len(started), len(results), len(finished)), (1, 1, 1)
        )
        self.assertEqual(results[0].shapes, [])
        self.assertIs(results[0].replace, False)
        self.assertEqual(self.model.calls[0][2], {"text_prompt": "describe"})

    def test_threaded_legacy_signals_are_exactly_once_and_busy_has_no_start(
        self,
    ):
        release = threading.Event()
        self.model.release = release
        started = []
        finished = []
        results = []
        order = []
        self.manager.prediction_started.connect(lambda: started.append(True))
        self.manager.prediction_finished.connect(
            lambda: (finished.append(True), order.append("finished"))
        )
        self.manager.new_auto_labeling_result.connect(
            lambda result: (results.append(result), order.append("result"))
        )

        self.assertTrue(self.manager.predict_shapes_threading(object()))
        self.assertTrue(_wait_until(lambda: len(self.model.calls) == 1))
        self.assertEqual(len(started), 1)
        self.assertFalse(self.manager.predict_shapes_threading(object()))
        self.assertEqual(len(started), 1)
        self.assertEqual(finished, [])
        release.set()

        self.assertTrue(_wait_until(lambda: len(finished) == 1))
        self.assertEqual(len(results), 1)
        self.assertEqual(order, ["result", "finished"])
        self.assertFalse(self.manager.inference_lease.is_active)
        self.assertIsNone(self.manager.model_execution_thread)

    def test_threaded_model_failure_releases_lease_without_canvas_result(self):
        self.model.error = RuntimeError("model boom")
        started = []
        finished = []
        results = []
        self.manager.prediction_started.connect(lambda: started.append(True))
        self.manager.prediction_finished.connect(lambda: finished.append(True))
        self.manager.new_auto_labeling_result.connect(results.append)

        self.assertTrue(self.manager.predict_shapes_threading(object()))
        self.assertTrue(_wait_until(lambda: len(finished) == 1))
        self.assertEqual(
            (len(started), len(finished), len(results)), (1, 1, 0)
        )
        self.assertFalse(self.manager.inference_lease.is_active)

    def test_active_lease_blocks_unload_remote_switch_and_task_switch(self):
        before = self.manager.loaded_model_config
        token = self.manager.inference_lease.acquire("TEST", "owner")
        self.assertFalse(self.manager.unload_model())
        self.assertFalse(self.manager.set_remote_server_model("new-model"))
        self.assertFalse(self.manager.set_task("new-task"))
        self.assertIs(self.manager.loaded_model_config, before)
        self.assertEqual(self.model.unload_calls, 0)
        self.assertEqual(self.model.model_ids, [])
        self.assertEqual(self.model.tasks, [])
        token.release()

        self.assertTrue(self.manager.set_remote_server_model("new-model"))
        self.assertTrue(self.manager.set_task("new-task"))

    def test_active_lease_blocks_integrated_and_custom_model_load(self):
        self.manager.model_configs = [
            {
                "config_file": "next-model.yaml",
                "display_name": "Next",
            }
        ]
        token = self.manager.inference_lease.acquire("TEST", "owner")
        with mock.patch.object(self.manager, "load_model_configs") as reload:
            self.assertFalse(self.manager.load_model("next-model.yaml"))
            self.assertFalse(self.manager.load_custom_model("missing.yaml"))
        reload.assert_not_called()
        self.assertIsNone(self.manager.model_download_thread)
        self.assertIs(self.manager.loaded_model_config["model"], self.model)
        token.release()

    def test_active_download_blocks_every_new_prediction_entry(self):
        self.manager.model_download_thread = _DownloadState()
        self.assertFalse(self.manager.predict_shapes(object(), batch=True))
        self.assertFalse(self.manager.predict_shapes_threading(object()))

        runner = PredictionRunner(self.manager)
        runner.start()
        request = PredictionRequest(
            run_id="run-a",
            job_id="job-a",
            attempt_id="attempt-a",
            image_id="image-a",
            canonical_image_path=os.path.abspath(__file__),
            generation=runner.generation,
            parameter_snapshot={},
            existing_shapes_input=[],
            delivery_mode="RETURN_ONLY",
        )
        self.assertFalse(runner.submit(request))
        self.assertEqual(self.model.calls, [])
        self.assertFalse(self.manager.inference_lease.is_active)
        runner.dispose_after_idle()
        self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_stale_download_cleanup_cannot_clear_a_new_generation(self):
        old_thread = mock.Mock()
        old_worker = mock.Mock()
        new_thread = mock.Mock()
        new_worker = mock.Mock()
        self.manager._model_load_generation = 2
        self.manager.model_download_thread = new_thread
        self.manager.model_download_worker = new_worker

        self.manager._on_model_download_thread_finished(
            1, old_thread, old_worker
        )

        self.assertIs(self.manager.model_download_thread, new_thread)
        self.assertIs(self.manager.model_download_worker, new_worker)
        old_thread.deleteLater.assert_called_once_with()
        new_thread.deleteLater.assert_not_called()

    def test_safe_shutdown_is_deferred_and_waits_for_download_and_lease(self):
        safe = []
        self.manager.safe_to_close.connect(safe.append)
        download = _DownloadState()
        self.manager.model_download_thread = download
        token = self.manager.inference_lease.acquire("TEST", "owner")
        generation = self.manager.request_safe_shutdown("window_close")
        self.assertEqual(safe, [])
        token.release()
        self.manager.on_inference_idle()
        self.assertEqual(safe, [])
        download.running = False
        self.manager.on_inference_idle()
        self.assertTrue(_wait_until(lambda: safe == [generation]))

    def test_idle_safe_shutdown_is_never_emitted_synchronously(self):
        safe = []
        self.manager.safe_to_close.connect(safe.append)
        generation = self.manager.request_safe_shutdown("window_close")
        self.assertEqual(safe, [])
        self.assertTrue(_wait_until(lambda: safe == [generation]))
        self.assertFalse(self.manager.unload_model())

    def test_default_runner_decodes_request_path_and_returns_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            Image.new("RGB", (2, 3), color=(10, 20, 30)).save(image_path)
            runner = PredictionRunner(self.manager)
            runner.start()
            outcomes = []
            legacy = []
            runner.outcome_ready.connect(outcomes.append)
            self.manager.new_auto_labeling_result.connect(legacy.append)
            request = PredictionRequest(
                run_id="run-a",
                job_id="job-a",
                attempt_id="attempt-a",
                image_id="image-a",
                canonical_image_path=os.path.abspath(image_path),
                generation=runner.generation,
                parameter_snapshot={
                    "image_input_source": "STANDALONE_FILE_LIST"
                },
                existing_shapes_input=[],
                delivery_mode="RETURN_ONLY",
            )
            self.assertTrue(runner.submit(request))
            self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
            self.assertEqual(outcomes[0].status, "succeeded")
            self.assertEqual(legacy, [])
            qimage = self.model.calls[0][0]
            self.assertEqual((qimage.width(), qimage.height()), (2, 3))
            runner.dispose_after_idle()
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_runner_model_conversion_uses_snapshot_not_a_second_file_read(
        self,
    ):
        class _SnapshotModel(_FakeModel):
            def __init__(self, image_path):
                super().__init__()
                self.image_path = image_path
                self.converted_pixel = None

            def predict_shapes(self, image, filename=None, **kwargs):
                Image.new("RGB", (2, 3), color=(200, 210, 220)).save(
                    self.image_path
                )
                converted = qt_img_to_rgb_cv_img(image, filename)
                self.converted_pixel = converted[0, 0].tolist()
                return super().predict_shapes(image, filename, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            Image.new("RGB", (2, 3), color=(10, 20, 30)).save(image_path)
            self.model = _SnapshotModel(image_path)
            self.manager.loaded_model_config["model"] = self.model
            runner = PredictionRunner(self.manager)
            runner.start()
            outcomes = []
            runner.outcome_ready.connect(outcomes.append)
            request = PredictionRequest(
                run_id="run-a",
                job_id="job-a",
                attempt_id="attempt-a",
                image_id="image-a",
                canonical_image_path=os.path.abspath(image_path),
                generation=runner.generation,
                parameter_snapshot={
                    "image_input_source": "STANDALONE_FILE_LIST"
                },
                existing_shapes_input=[],
                delivery_mode="RETURN_ONLY",
            )

            self.assertTrue(runner.submit(request))
            self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
            self.assertEqual(outcomes[0].status, "succeeded")
            self.assertEqual(self.model.converted_pixel, [10, 20, 30])
            runner.dispose_after_idle()
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_runner_preserves_snapshot_conflict_code_without_calling_model(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            Image.new("RGB", (2, 3), color=(10, 20, 30)).save(image_path)
            runner = PredictionRunner(self.manager)
            runner.start()
            outcomes = []
            runner.outcome_ready.connect(outcomes.append)
            request = PredictionRequest(
                run_id="run-a",
                job_id="job-a",
                attempt_id="attempt-a",
                image_id="image-a",
                canonical_image_path=os.path.abspath(image_path),
                generation=runner.generation,
                parameter_snapshot={
                    "image_input_source": "STANDALONE_FILE_LIST",
                    "expected_sha256": "0" * 64,
                },
                existing_shapes_input=[],
                delivery_mode="RETURN_ONLY",
            )
            self.assertTrue(runner.submit(request))
            self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
            self.assertEqual(outcomes[0].status, "failed")
            self.assertEqual(outcomes[0].error_code, "conflict_image_changed")
            self.assertEqual(self.model.calls, [])
            runner.dispose_after_idle()
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_fast_request_reapplies_frozen_model_parameters(self):
        class FrozenModel(_FakeModel):
            class Meta:
                output_modes = {
                    "rectangle": "Rectangle",
                    "polygon": "Polygon",
                }

            def __init__(self):
                super().__init__()
                self.conf_thres = 0.9
                self.iou_thres = 0.9
                self.kpt_thres = 0.9
                self.output_mode = "polygon"
                self.replace = True
                self.observed = []

            def set_output_mode(self, mode):
                self.output_mode = mode

            def predict_shapes(self, image, filename=None, **kwargs):
                self.observed.append(
                    (
                        self.conf_thres,
                        self.iou_thres,
                        self.kpt_thres,
                        self.output_mode,
                        self.replace,
                    )
                )
                return AutoLabelingResult([], self.replace, "")

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            Image.new("RGB", (2, 3), color=(10, 20, 30)).save(image_path)
            model = FrozenModel()
            self.manager.loaded_model_config = {
                "type": "yolov8_pose",
                "model": model,
            }
            parameters = {
                "model_type": "yolov8_pose",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "keypoint_threshold": 0.1,
                "output_mode": "rectangle",
                "preserve_existing_annotations": True,
                "replace": False,
                "skip_detection": False,
                "cropping_mode": None,
                "mask_fineness": None,
                "image_input_source": "STANDALONE_FILE_LIST",
            }
            token = self.manager.inference_lease.acquire("TEST", "fast-run")
            try:
                for index in range(2):
                    request = PredictionRequest(
                        run_id="run-a",
                        job_id=f"job-{index}",
                        attempt_id=f"attempt-{index}",
                        image_id=f"image-{index}",
                        canonical_image_path=os.path.abspath(image_path),
                        generation=1,
                        parameter_snapshot=parameters,
                        existing_shapes_input=[],
                        delivery_mode="RETURN_ONLY",
                    )
                    outcome = self.manager.execute_prediction_request_unleased(
                        request, token
                    )
                    self.assertEqual(outcome.status, "succeeded")
                    self.assertFalse(outcome.result.replace)
                    model.conf_thres = 0.99
                    model.iou_thres = 0.99
                    model.kpt_thres = 0.99
                    model.output_mode = "polygon"
                    model.replace = True
            finally:
                token.release()

            self.assertEqual(
                model.observed,
                [
                    (0.25, 0.45, 0.1, "rectangle", False),
                    (0.25, 0.45, 0.1, "rectangle", False),
                ],
            )

    def test_fast_request_rejects_changed_model_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "target.png"
            Image.new("RGB", (2, 3)).save(image_path)
            self.manager.loaded_model_config = {
                "type": "yolo11",
                "model": self.model,
            }
            request = PredictionRequest(
                run_id="run-a",
                job_id="job-a",
                attempt_id="attempt-a",
                image_id="image-a",
                canonical_image_path=os.path.abspath(image_path),
                generation=1,
                parameter_snapshot={
                    "model_type": "yolov8",
                    "image_input_source": "STANDALONE_FILE_LIST",
                },
                existing_shapes_input=[],
                delivery_mode="RETURN_ONLY",
            )
            token = self.manager.inference_lease.acquire("TEST", "fast-run")
            try:
                outcome = self.manager.execute_prediction_request_unleased(
                    request, token
                )
            finally:
                token.release()

            self.assertEqual(outcome.status, "failed")
            self.assertEqual(outcome.error_code, "prediction_model_changed")
            self.assertEqual(self.model.calls, [])


if __name__ == "__main__":
    unittest.main()
