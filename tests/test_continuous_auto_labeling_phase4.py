import hashlib
import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from PIL import Image
from PyQt5 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.inference_lease import (
    InferenceLeaseRegistry,
)
from anylabeling.services.auto_labeling.prediction_job import (
    AutoLabelingPayload,
    PredictionOutcome,
)
from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    FastRunOptionsV1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    FastAutoLabelingController,
    FastCanvasStateGuard,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    commit_label_for_image_v1,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_until(predicate, timeout_ms=10000):
    if predicate():
        return True
    loop = QtCore.QEventLoop()
    poll = QtCore.QTimer()
    poll.setInterval(5)

    def check():
        if predicate():
            poll.stop()
            loop.quit()

    poll.timeout.connect(check)
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    poll.start()
    timeout.start(timeout_ms)
    loop.exec_()
    return predicate()


class _Manager(QtCore.QObject):
    safe_to_close = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.inference_lease = InferenceLeaseRegistry()
        self._shutdown_generation = 0
        self.statuses = []

    def acquire_inference_lease(self, owner_kind, owner_id):
        token = self.inference_lease.acquire(owner_kind, owner_id)
        if token is None:
            self.statuses.append("Another model is being executed")
        return token

    def on_inference_idle(self):
        pass

    def request_safe_shutdown(self, _reason):
        self._shutdown_generation += 1
        generation = self._shutdown_generation
        QtCore.QTimer.singleShot(
            0, lambda: self.safe_to_close.emit(generation)
        )
        return generation


def _fingerprint():
    return {
        "fingerprint_schema_version": 1,
        "model_type": "yolov8",
        "config_digest": "a" * 64,
        "artifact_digests": [
            {"field": "model_path", "sha256": "b" * 64, "size": 1}
        ],
        "remote_endpoint_identity": None,
        "adapter_version": "phase4-yolo-v1",
        "resume_supported": True,
    }


def _options(**changes):
    values = {
        "delay_seconds": 0.0,
        "range": "ALL_IMAGES",
        "filter": "ALL",
        "write_policy": "FORCE_REPLACE",
        "current_anchor_image_id": None,
        "workset_source": "CURRENT_FILE_LIST_SNAPSHOT",
        "model_fingerprint": _fingerprint(),
        "parameter_snapshot": {"confidence_threshold": 0.25},
    }
    values.update(changes)
    return FastRunOptionsV1(**values)


def _images(root, count):
    images = root / "images"
    labels = root / "labels"
    images.mkdir()
    labels.mkdir()
    paths = []
    for index in range(count):
        path = images / f"image-{index:04d}.png"
        Image.new("RGB", (4, 3), color=(index % 255, 20, 30)).save(path)
        paths.append(str(path.resolve()))
    return paths, labels


def _valid_empty(image_name):
    return {
        "version": "3.3.7",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": 3,
        "imageWidth": 4,
        "description": "",
    }


def _success(request, target_count=0):
    shapes = []
    if target_count:
        shapes = [
            {
                "label": "object",
                "points": [[0, 0], [2, 2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        ]
    digest = hashlib.sha256(request.image_id.encode()).hexdigest()
    payload = AutoLabelingPayload(
        shapes,
        True,
        "",
        input_width=4,
        input_height=3,
        source_image_digest=digest,
    )
    return PredictionOutcome.succeeded(request, payload)


class FastControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def _run(self, paths, labels, executor, options=None, timeout=15000):
        manager = _Manager()
        runner = PredictionRunner(manager, request_executor=executor)
        controller = FastAutoLabelingController(
            runner,
            options or _options(),
            standalone_image_paths=paths,
            standalone_output_dir=str(labels),
        )
        final = []
        controller.finished.connect(final.append)
        self.assertTrue(controller.start())
        self.assertTrue(_wait_until(lambda: bool(final), timeout))
        self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
        return controller, runner, manager, final[0]

    def test_zero_eligible_items_finishes_without_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, labels = _images(root, 2)
            for path in paths:
                target = labels / (Path(path).stem + ".json")
                target.write_text(
                    json.dumps(_valid_empty(Path(path).name)),
                    encoding="utf-8",
                )
            calls = []
            controller, _runner, _manager, summary = self._run(
                paths,
                labels,
                lambda request, _lease: calls.append(request),
                _options(filter="ONLY_WITHOUT_VALID_ANNOTATION"),
            )
            self.assertEqual(calls, [])
            self.assertEqual(summary["processing_status"], "COMPLETED")
            self.assertEqual(summary["skipped_existing"], 2)
            self.assertEqual(summary["pending_review"], 0)
            self.assertEqual(summary["review_progress"], "COMPLETED")
            self.assertFalse(controller.modified_image_ids)

    def test_one_two_and_one_hundred_are_serial_exactly_once(self):
        for count in (1, 2, 100):
            with (
                self.subTest(count=count),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                paths, labels = _images(root, count)
                calls = []
                thread_ids = []
                thread_objects = []

                def execute(request, _lease):
                    calls.append(request.image_id)
                    thread_ids.append(int(QtCore.QThread.currentThreadId()))
                    thread_objects.append(QtCore.QThread.currentThread())
                    return _success(request)

                controller, runner, manager, summary = self._run(
                    paths, labels, execute, timeout=30000
                )
                self.assertEqual(len(calls), count)
                self.assertEqual(len(set(calls)), count)
                self.assertEqual(len(set(thread_ids)), 1)
                self.assertTrue(
                    all(
                        thread is thread_objects[0]
                        for thread in thread_objects
                    )
                )
                self.assertEqual(summary["succeeded"], count)
                self.assertEqual(summary["zero_target"], count)
                self.assertEqual(summary["processing_status"], "COMPLETED")
                self.assertEqual(summary["pending_review"], count)
                self.assertEqual(summary["review_progress"], "NOT_STARTED")
                self.assertEqual(manager.statuses, [])
                self.assertEqual(
                    runner.completed_attempt_count, min(count, 128)
                )
                self.assertEqual(len(controller.modified_image_ids), count)
                for path in paths:
                    document = json.loads(
                        (labels / (Path(path).stem + ".json")).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(document["shapes"], [])

    def test_each_later_admission_is_observed_only_after_runner_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 5)
            manager = _Manager()
            runner = PredictionRunner(
                manager,
                request_executor=lambda request, _lease: _success(request),
            )
            idle_count = [0]
            admission_idle_counts = []
            runner.runner_idle.connect(
                lambda _generation: idle_count.__setitem__(
                    0,
                    idle_count[0] + 1,
                )
            )
            runner.request_started.connect(
                lambda _request: admission_idle_counts.append(idle_count[0])
            )
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(admission_idle_counts, [0, 1, 2, 3, 4])

    def test_failed_input_auto_skips_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                if len(calls) == 1:
                    return PredictionOutcome.failed(
                        request, "failed_input", "decode failed"
                    )
                return _success(request, target_count=1)

            controller, _runner, _manager, summary = self._run(
                paths, labels, execute
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["failed_input"], 1)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["processing_status"], "COMPLETED")
            items = controller.run_store.list_items(controller.run_id)
            self.assertEqual(items[0]["failure_resolution"], "auto_skip")

    def test_nonzero_success_has_distinct_zero_target_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            _controller, _runner, _manager, summary = self._run(
                paths,
                labels,
                lambda request, _lease: _success(request, target_count=1),
            )

            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["zero_target"], 0)
            self.assertEqual(summary["pending_review"], 1)
            self.assertEqual(summary["processing_status"], "COMPLETED")
            self.assertEqual(summary["review_progress"], "NOT_STARTED")

    def test_active_filename_is_reported_before_inference_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            entered = threading.Event()
            release = threading.Event()

            def execute(request, _lease):
                entered.set()
                release.wait(5)
                return _success(request)

            manager = _Manager()
            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            progress = []
            final = []
            controller.progress_changed.connect(progress.append)
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(entered.wait(2))
            self.assertTrue(progress)
            self.assertEqual(
                progress[0]["current_filename"], Path(paths[0]).name
            )
            self.assertEqual(progress[0]["processed"], 0)
            self.assertEqual(progress[0]["total"], 1)
            self.assertEqual(final, [])
            release.set()
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_duplicate_and_stale_outcomes_do_not_repeat_commit_or_statistics(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            entered = threading.Event()
            release = threading.Event()
            requests = []

            def execute(request, _lease):
                requests.append(request)
                entered.set()
                release.wait(5)
                return _success(request)

            manager = _Manager()
            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            with mock.patch(
                "anylabeling.views.labeling.utils.continuous_auto_labeling."
                "commit_label_for_image_v1",
                wraps=commit_label_for_image_v1,
            ) as commit:
                self.assertTrue(controller.start())
                self.assertTrue(entered.wait(2))
                request = requests[0]
                controller._on_outcome_ready(
                    _success(
                        replace(request, generation=request.generation - 1)
                    )
                )
                controller._on_outcome_ready(_success(request))
                controller._on_outcome_ready(_success(request))
                self.assertEqual(commit.call_count, 1)
                release.set()
                self.assertTrue(_wait_until(lambda: bool(final)))
                self.assertEqual(commit.call_count, 1)
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(final[0]["succeeded"], 1)
            self.assertEqual(final[0]["zero_target"], 1)
            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(len(item["prediction_attempts"]), 1)

    def test_model_error_retry_uses_new_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            manager = _Manager()
            attempts = []

            def execute(request, _lease):
                attempts.append(request.attempt_id)
                if len(attempts) == 1:
                    return PredictionOutcome.failed(
                        request, "model_prediction_failed", "out of memory"
                    )
                return _success(request)

            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(
                _wait_until(
                    lambda: controller.phase == "WAITING_ERROR"
                    and runner.is_idle()
                )
            )
            self.assertTrue(controller.retry_current())
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertEqual(len(attempts), 2)
            self.assertNotEqual(attempts[0], attempts[1])
            self.assertEqual(final[0]["processing_status"], "COMPLETED")
            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(len(item["prediction_attempts"]), 2)
            self.assertIsNone(item["result_summary"]["error_code"])
            self.assertIsNone(item["result_summary"]["error_message"])

    def test_late_previous_attempt_and_generation_are_ignored_during_retry(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            manager = _Manager()
            requests = []
            second_entered = threading.Event()
            release_second = threading.Event()

            def execute(request, _lease):
                requests.append(request)
                if len(requests) == 1:
                    return PredictionOutcome.failed(
                        request,
                        "model_prediction_failed",
                        "retry me",
                    )
                second_entered.set()
                release_second.wait(5)
                return _success(request)

            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(
                _wait_until(
                    lambda: controller.phase == "WAITING_ERROR"
                    and runner.is_idle()
                )
            )
            self.assertTrue(controller.retry_current())
            self.assertTrue(second_entered.wait(2))

            controller._on_outcome_ready(_success(requests[0]))
            controller._on_outcome_ready(
                _success(
                    replace(
                        requests[1],
                        generation=requests[1].generation - 1,
                    )
                )
            )
            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(item["execution_status"], "running")
            self.assertEqual(item["latest_attempt_id"], requests[1].attempt_id)
            self.assertFalse(any(labels.iterdir()))

            release_second.set()
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(final[0]["succeeded"], 1)
            self.assertEqual(len(requests), 2)

    def test_pause_finishes_current_then_resume_and_stop_is_not_forceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            manager = _Manager()
            entered = threading.Event()
            release = threading.Event()
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                if len(calls) == 1:
                    entered.set()
                    release.wait(5)
                return _success(request)

            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(entered.wait(2))
            self.assertTrue(controller.request_pause())
            release.set()
            self.assertTrue(_wait_until(lambda: controller.phase == "PAUSED"))
            self.assertEqual(len(calls), 1)
            self.assertTrue(runner.is_idle())
            self.assertIsNone(controller.active_request)
            paused_state = controller.run_store.read_state(controller.run_id)
            self.assertFalse(controller.resume("KEEP_PAUSED"))
            self.assertEqual(
                controller.run_store.read_state(controller.run_id),
                paused_state,
            )
            self.assertEqual(controller.phase, "PAUSED")
            self.assertEqual(len(calls), 1)
            self.assertTrue(controller.resume())
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertEqual(len(calls), 2)
            self.assertEqual(final[0]["processing_status"], "COMPLETED")

        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            manager = _Manager()
            entered = threading.Event()
            release = threading.Event()
            calls = []

            def execute_stop(request, _lease):
                calls.append(request.image_id)
                entered.set()
                release.wait(5)
                return _success(request)

            runner = PredictionRunner(manager, request_executor=execute_stop)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            controller.start()
            self.assertTrue(entered.wait(2))
            controller.request_stop()
            self.assertFalse(final)
            release.set()
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(final[0]["processing_status"], "PARTIAL")

    def test_repeated_controls_are_idempotent_and_preserve_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            entered = threading.Event()
            release = threading.Event()

            def execute(request, _lease):
                entered.set()
                release.wait(5)
                return _success(request)

            manager = _Manager()
            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(entered.wait(2))

            self.assertTrue(controller.request_pause())
            pause_revision = controller.run_store.read_state(
                controller.run_id
            )["revision"]
            self.assertTrue(controller.request_pause())
            self.assertEqual(
                controller.run_store.read_state(controller.run_id)["revision"],
                pause_revision,
            )
            self.assertTrue(controller.request_stop())
            stop_revision = controller.run_store.read_state(controller.run_id)[
                "revision"
            ]
            self.assertTrue(controller.request_stop())
            self.assertEqual(
                controller.run_store.read_state(controller.run_id)["revision"],
                stop_revision,
            )
            self.assertFalse(controller.request_pause())
            self.assertEqual(controller.control_intent, "STOP")
            self.assertEqual(
                controller.run_store.read_state(controller.run_id)["revision"],
                stop_revision,
            )

            generation = controller.request_close()
            close_revision = controller.run_store.read_state(
                controller.run_id
            )["revision"]
            self.assertEqual(controller.request_close(), generation)
            self.assertFalse(controller.request_stop())
            self.assertFalse(controller.request_pause())
            self.assertEqual(controller.control_intent, "CLOSE")
            self.assertEqual(
                controller.run_store.read_state(controller.run_id)["revision"],
                close_revision,
            )
            release.set()
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_stop_counts_an_existing_annotation_as_a_safe_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            (labels / "image-0000.json").write_text(
                json.dumps(_valid_empty("image-0000.png")),
                encoding="utf-8",
            )
            manager = _Manager()

            def execute(request, _lease):
                return PredictionOutcome.failed(
                    request, "model_prediction_failed", "failed"
                )

            runner = PredictionRunner(manager, request_executor=execute)
            controller = FastAutoLabelingController(
                runner,
                _options(filter="ONLY_WITHOUT_VALID_ANNOTATION"),
                standalone_image_paths=paths,
                standalone_output_dir=str(labels),
            )
            final = []
            controller.finished.connect(final.append)
            self.assertTrue(controller.start())
            self.assertTrue(
                _wait_until(lambda: controller.phase == "WAITING_ERROR")
            )
            self.assertTrue(controller.request_stop())
            self.assertTrue(_wait_until(lambda: bool(final)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(final[0]["skipped_existing"], 1)
            self.assertEqual(final[0]["model_failed_unresolved"], 1)
            self.assertEqual(final[0]["processing_status"], "PARTIAL")


class CanvasStateGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_background_work_never_switches_and_current_reload_is_at_most_once(
        self,
    ):
        class _Value:
            def __init__(self, value):
                self.current = value

            def value(self):
                return self.current

            def setValue(self, value):
                self.current = value

        class _FileList(_Value):
            def __init__(self, row):
                super().__init__(row)
                self.enabled = True

            def currentRow(self):
                return self.current

            def setCurrentRow(self, row):
                self.current = row

            def isEnabled(self):
                return self.enabled

            def setEnabled(self, enabled):
                self.enabled = enabled

        class _Control:
            def __init__(self, enabled=True):
                self.enabled = enabled

            def isEnabled(self):
                return self.enabled

            def setEnabled(self, enabled):
                self.enabled = enabled

        image = os.path.abspath("current.jpg")
        record = type(
            "Record",
            (),
            {"image_id": "current-id"},
        )()
        widget = type("Widget", (), {})()
        widget.filename = image
        widget.file_list_widget = _FileList(7)
        widget.zoom_mode = 2
        widget.zoom_widget = _Value(135)
        widget.scroll_bars = {
            QtCore.Qt.Vertical: _Value(22),
            QtCore.Qt.Horizontal: _Value(33),
        }
        widget.auto_labeling_host_context = type(
            "Context",
            (),
            {"image_records_by_path": {_canonical(image): record}},
        )()
        widget.canvas = type(
            "Canvas",
            (),
            {
                "selected_shapes": [object()],
                "sequence_edit_locked": False,
                "set_sequence_edit_locked": lambda self, locked: setattr(
                    self, "sequence_edit_locked", bool(locked)
                ),
            },
        )()
        widget.loads = []
        widget.auto_labeling_widget = _Control()
        widget.label_list = _Control()
        widget.actions = type(
            "Actions",
            (),
            {
                "save": _Control(),
                "delete": _Control(),
                "open_next_image": _Control(),
            },
        )()
        widget.load_file = widget.loads.append
        widget.set_zoom = widget.zoom_widget.setValue
        widget.open_next_image = mock.Mock()
        widget.import_image_folder = mock.Mock()

        guard = FastCanvasStateGuard(widget)
        guard.set_running(True)
        self.assertFalse(widget.file_list_widget.enabled)
        self.assertTrue(widget.canvas.sequence_edit_locked)
        self.assertTrue(widget.fast_auto_labeling_edit_locked)
        self.assertFalse(widget.auto_labeling_widget.enabled)
        self.assertFalse(widget.label_list.enabled)
        self.assertFalse(widget.actions.save.enabled)
        self.assertFalse(widget.actions.delete.enabled)
        self.assertFalse(widget.actions.open_next_image.enabled)
        guard.set_paused(True)
        self.assertTrue(widget.file_list_widget.enabled)
        self.assertFalse(widget.canvas.sequence_edit_locked)
        self.assertFalse(widget.fast_auto_labeling_edit_locked)
        self.assertFalse(widget.auto_labeling_widget.enabled)
        self.assertTrue(widget.label_list.enabled)
        self.assertTrue(widget.actions.save.enabled)
        self.assertTrue(widget.actions.delete.enabled)
        self.assertTrue(widget.actions.open_next_image.enabled)
        guard.restore(set())
        self.assertEqual(widget.loads, [])
        widget.open_next_image.assert_not_called()
        widget.import_image_folder.assert_not_called()

        guard = FastCanvasStateGuard(widget)
        guard.set_running(True)
        guard.restore({"current-id"})
        self.assertEqual(widget.loads, [image])
        self.assertEqual(widget.file_list_widget.currentRow(), 7)
        self.assertEqual(widget.zoom_widget.value(), 135)
        self.assertEqual(widget.scroll_bars[QtCore.Qt.Vertical].value(), 22)
        self.assertEqual(widget.scroll_bars[QtCore.Qt.Horizontal].value(), 33)
        self.assertEqual(widget.canvas.selected_shapes, [])

        widget.loads.clear()
        guard = FastCanvasStateGuard(widget)
        guard.set_running(True)
        guard.release_for_close()
        self.assertEqual(widget.loads, [])
        self.assertFalse(widget.fast_auto_labeling_active)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


if __name__ == "__main__":
    unittest.main()
