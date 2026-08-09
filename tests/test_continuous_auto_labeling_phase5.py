import copy
import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt5 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.prediction_job import (
    PredictionOutcome,
)
from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils import batch
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    atomic_write_label_document,
    resolve_existing_label,
)
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    FastRunOptionsV1,
    FastSequenceContractError,
    SequenceRunOptionsV1,
    resolve_sequence_capabilities,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    ContinuousAutoLabelingController,
    FastAutoLabelingController,
    FastControllerError,
    LabelingWidgetSequencePresenter,
    VisibleCanvasStateGuard,
)
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)
from anylabeling.views.labeling.widgets.auto_labeling_run_dialog import (
    ContinuousRunSetupDialog,
    FastRunProgressDialog,
    continuous_auto_labeling_settings_v1,
)
from tests.test_continuous_auto_labeling_phase4 import (
    _Manager,
    _fingerprint,
    _images,
    _options,
    _success,
    _valid_empty,
    _wait_until,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sequence_options(delay=2.0, **changes):
    values = {
        "delay_seconds": delay,
        "range": "ALL_IMAGES",
        "filter": "ALL",
        "write_policy": "FORCE_REPLACE",
        "current_anchor_image_id": None,
        "workset_source": "CURRENT_FILE_LIST_SNAPSHOT",
        "model_fingerprint": _fingerprint(),
        "parameter_snapshot": {"confidence_threshold": 0.25},
    }
    values.update(changes)
    return SequenceRunOptionsV1(**values)


class _MonotonicClock:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._value

    def advance(self, seconds):
        with self._lock:
            self._value += int(seconds * 1_000_000_000)


class _ManualPresentationTimer:
    def __init__(self):
        self.active = False
        self.milliseconds = None
        self.epoch = None
        self.callback = None
        self.history = []

    def start(self, milliseconds, epoch, callback):
        self.stop()
        self.active = True
        self.milliseconds = milliseconds
        self.epoch = epoch
        self.callback = callback
        self.history.append((milliseconds, epoch, callback))

    def stop(self):
        self.active = False
        self.milliseconds = None
        self.epoch = None
        self.callback = None

    def is_active(self):
        return self.active

    def fire(self):
        if not self.active:
            raise AssertionError("presentation timer is not active")
        epoch = self.epoch
        callback = self.callback
        self.active = False
        callback(epoch)


class _DiskPresenter:
    def __init__(self, clock=None, load_seconds=0.0, present_seconds=0.0):
        self.clock = clock
        self.load_seconds = load_seconds
        self.present_seconds = present_seconds
        self.token = None
        self.records = {}
        self.events = []
        self.presented_image_ids = []
        self.presented_documents = []
        self.presented_item_revisions = []
        self.deactivated = 0

    def activate(self, token, records):
        self.token = token
        self.records = dict(records)
        self.events.append(("activate", None))

    def load(self, token, epoch, image_id, _expected_digest):
        if token != self.token:
            raise FastControllerError("presentation_token_mismatch")
        if self.clock is not None:
            self.clock.advance(self.load_seconds)
        self.events.append(("load", image_id, epoch))

    def present(
        self,
        token,
        epoch,
        image_id,
        item,
        _expected_digest,
    ):
        if token != self.token:
            raise FastControllerError("presentation_token_mismatch")
        record = self.records[image_id]
        label_path = record.canonical_session_label_path
        resolved = resolve_existing_label(label_path)
        digests = item["digests"]
        if resolved.document_digest != digests["staged_document_digest"]:
            raise FastControllerError("presentation_document_digest_mismatch")
        if resolved.semantic_digest != digests["staged_annotation_digest"]:
            raise FastControllerError("presentation_semantic_digest_mismatch")
        if self.clock is not None:
            self.clock.advance(self.present_seconds)
        self.events.append(("present", image_id, epoch))
        self.presented_image_ids.append(image_id)
        self.presented_documents.append(copy.deepcopy(resolved.document))
        self.presented_item_revisions.append(item["item_revision"])

    def revalidate(self, token, image_id, item, expected_digest):
        record = self.records[image_id]
        resolved = resolve_existing_label(record.canonical_session_label_path)
        if (
            token != self.token
            or resolved.document_digest
            != item["digests"]["staged_document_digest"]
        ):
            raise FastControllerError("presentation_document_digest_mismatch")
        self.events.append(("revalidate", image_id, expected_digest))

    def deactivate(self, token):
        if token != self.token:
            return False
        self.deactivated += 1
        self.events.append(("deactivate", None))
        self.token = None
        return True


def _visible_controller(
    paths,
    labels,
    executor,
    *,
    delay=2.0,
    options=None,
    presenter=None,
    timer=None,
    clock=None,
):
    manager = _Manager()
    runner = PredictionRunner(manager, request_executor=executor)
    clock = clock or _MonotonicClock()
    timer = timer or _ManualPresentationTimer()
    presenter = presenter or _DiskPresenter()
    controller = ContinuousAutoLabelingController(
        runner,
        options or _sequence_options(delay),
        standalone_image_paths=paths,
        standalone_output_dir=str(labels),
        presentation_adapter=presenter,
        presentation_timer=timer,
        monotonic_ns=clock,
    )
    return controller, runner, manager, presenter, timer, clock


def _fire_full_delay(controller, timer, clock):
    if not _wait_until(lambda: timer.active or controller._finished):
        raise AssertionError("presentation timer was not armed")
    if controller._finished:
        return
    clock.advance(controller.remaining_seconds)
    timer.fire()


class SequenceOptionsAndDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_delay_domain_mode_and_fast_compatibility(self):
        for delay, mode in (
            (0, "FAST"),
            (0.5, "VISIBLE"),
            (2.0, "VISIBLE"),
            (60, "VISIBLE"),
        ):
            with self.subTest(delay=delay):
                options = _sequence_options(delay)
                self.assertEqual(options.delay_seconds, float(delay))
                self.assertEqual(options.execution_mode, mode)

        for delay, code in (
            (-0.5, "invalid_sequence_delay_seconds"),
            (60.5, "invalid_sequence_delay_seconds"),
            (math.nan, "invalid_sequence_delay_seconds"),
            (math.inf, "invalid_sequence_delay_seconds"),
            (False, "invalid_sequence_delay_seconds"),
            (0.25, "invalid_sequence_delay_step"),
        ):
            with self.subTest(delay=delay):
                with self.assertRaises(FastSequenceContractError) as raised:
                    _sequence_options(delay)
                self.assertEqual(raised.exception.code, code)

        with self.assertRaises(FastSequenceContractError) as raised:
            FastRunOptionsV1(
                **{
                    **_sequence_options(0.0).__dict__,
                    "delay_seconds": 0.5,
                }
            )
        self.assertEqual(
            raised.exception.code, "visible_mode_not_available_phase4"
        )

    def test_yolo_capability_keeps_phase4_adapter_identity(self):
        capability = resolve_sequence_capabilities(
            {"type": "yolov8_pose", "model": object()}
        )
        self.assertTrue(capability.supports_fast_sequence)
        self.assertTrue(capability.supports_visible_sequence)
        self.assertTrue(capability.single_image_independent)
        self.assertFalse(capability.stateful_across_images)
        self.assertEqual(capability.adapter_version, "phase4-yolo-v1")

    def test_settings_defaults_invalid_fallback_and_dynamic_copy(self):
        self.assertEqual(
            continuous_auto_labeling_settings_v1({}),
            {
                "delay_seconds": 2.0,
                "range": "CURRENT_TO_END",
                "filter": "ALL",
                "write_policy": "INHERIT_MODEL_POLICY",
            },
        )
        settings = continuous_auto_labeling_settings_v1(
            {
                "unrelated": {"kept": True},
                "continuous_auto_labeling": {
                    "delay_seconds": 0.25,
                    "range": "bad",
                    "filter": "bad",
                    "write_policy": "bad",
                },
            }
        )
        self.assertEqual(settings["delay_seconds"], 2.0)
        self.assertEqual(settings["range"], "CURRENT_TO_END")

        dialog = ContinuousRunSetupDialog(
            {
                "workset_total": 3,
                "existing_annotation": 1,
                "host_prepare_failed": 0,
            },
            "YOLO",
            current_anchor_available=True,
        )
        try:
            self.assertEqual(dialog.delay_spin.minimum(), 0.0)
            self.assertEqual(dialog.delay_spin.maximum(), 60.0)
            self.assertEqual(dialog.delay_spin.singleStep(), 0.5)
            self.assertEqual(dialog.delay_spin.decimals(), 1)
            self.assertEqual(dialog.delay_spin.value(), 2.0)
            self.assertEqual(
                dialog.note_label.text(),
                auto_labeling_text_v1("visible_description"),
            )
            self.assertEqual(
                dialog.start_button.text(),
                auto_labeling_text_v1("start_visible"),
            )
            dialog.delay_spin.setValue(0.0)
            self.assertEqual(
                dialog.note_label.text(),
                auto_labeling_text_v1("fast_description"),
            )
            self.assertEqual(
                dialog.start_button.text(),
                auto_labeling_text_v1("start_fast"),
            )
        finally:
            dialog.close()

    def test_button_and_ctrl_b_use_one_entry_with_distinct_initial_delay(self):
        opener = mock.Mock(return_value="opened")
        button_owner = SimpleNamespace(open_continuous_auto_labeling=opener)
        self.assertEqual(
            AutoLabelingWidget.run_continuous_auto_labeling(button_owner),
            "opened",
        )
        opener.assert_called_once_with(initial_delay=None)

        opener.reset_mock()
        manager = SimpleNamespace(
            loaded_model_config={"type": "yolov8", "model": object()},
            new_model_status=SimpleNamespace(emit=mock.Mock()),
        )
        labeling_widget = SimpleNamespace(
            auto_labeling_widget=SimpleNamespace(
                model_manager=manager,
                open_continuous_auto_labeling=opener,
            ),
            tr=lambda value: value,
        )
        self.assertEqual(batch.run_all_images(labeling_widget), "opened")
        opener.assert_called_once_with()

    def test_progress_exposes_presenting_and_paused_remaining(self):
        dialog = FastRunProgressDialog()
        try:
            dialog.set_phase("PRESENTING")
            dialog.update_progress(
                {
                    "total": 3,
                    "processed": 1,
                    "current_filename": "image.png",
                    "delay_seconds": 2.0,
                    "remaining_seconds": 1.25,
                }
            )
            self.assertEqual(
                dialog.state_label.text(),
                auto_labeling_text_v1("phase_presenting"),
            )
            self.assertEqual(
                dialog.configured_delay_label.text(),
                f"2.0{auto_labeling_text_v1('seconds_suffix')}",
            )
            self.assertEqual(
                dialog.remaining_delay_label.text(),
                f"1.2{auto_labeling_text_v1('seconds_suffix')}",
            )
            dialog.set_phase("PAUSED")
            dialog.update_progress(
                {"delay_seconds": 2.0, "remaining_seconds": 1.25}
            )
            self.assertIn("1.2", dialog.state_label.text())
        finally:
            dialog.close()


class VisibleLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_one_two_and_one_hundred_commit_then_present_exactly_once(self):
        for count in (1, 2, 100):
            with (
                self.subTest(count=count),
                tempfile.TemporaryDirectory() as tmp,
            ):
                paths, labels = _images(Path(tmp), count)
                calls = []

                def execute(request, _lease):
                    calls.append(request.image_id)
                    return _success(request, target_count=1)

                (
                    controller,
                    runner,
                    _manager,
                    presenter,
                    timer,
                    clock,
                ) = _visible_controller(paths, labels, execute, delay=0.5)
                finished = []
                ready = []
                controller.finished.connect(finished.append)
                controller.presentation_ready.connect(ready.append)
                self.assertTrue(controller.start())
                for _index in range(count):
                    self.assertTrue(_wait_until(lambda: timer.active))
                    self.assertTrue(controller.runner_idle_seen)
                    _fire_full_delay(controller, timer, clock)
                self.assertTrue(_wait_until(lambda: bool(finished), 30000))
                self.assertTrue(
                    _wait_until(lambda: runner.worker_thread is None)
                )

                self.assertEqual(len(calls), count)
                self.assertEqual(len(set(calls)), count)
                self.assertEqual(presenter.presented_image_ids, calls)
                self.assertEqual(len(ready), count)
                self.assertEqual(len(timer.history), count)
                self.assertEqual(finished[0]["succeeded"], count)
                self.assertEqual(finished[0]["processing_status"], "COMPLETED")
                self.assertEqual(presenter.deactivated, 1)
                for document in presenter.presented_documents:
                    self.assertEqual(len(document["shapes"]), 1)
                for image_id, presented_revision in zip(
                    calls,
                    presenter.presented_item_revisions,
                ):
                    self.assertEqual(
                        controller.run_store.read_item(
                            controller.run_id,
                            image_id,
                        )["item_revision"],
                        presented_revision,
                    )

    def test_deadline_starts_after_load_infer_commit_and_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            clock = _MonotonicClock()
            presenter = _DiskPresenter(
                clock,
                load_seconds=3.0,
                present_seconds=7.0,
            )

            def execute(request, _lease):
                clock.advance(11.0)
                return _success(request)

            controller, runner, _manager, _presenter, timer, _clock = (
                _visible_controller(
                    paths,
                    labels,
                    execute,
                    delay=2.0,
                    presenter=presenter,
                    clock=clock,
                )
            )
            finished = []
            controller.finished.connect(finished.append)
            self.assertTrue(controller.start())
            self.assertTrue(_wait_until(lambda: timer.active))
            self.assertEqual(clock(), 21_000_000_000)
            self.assertEqual(
                controller.deadline_monotonic - clock(),
                2_000_000_000,
            )
            self.assertEqual(timer.milliseconds, 2000)
            clock.advance(2.0)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_timer_and_runner_idle_are_an_order_independent_double_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                return _success(request)

            controller, runner, _manager, _presenter, timer, clock = (
                _visible_controller(paths, labels, execute, delay=0.5)
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(lambda: timer.active))
            first_epoch = timer.epoch
            first_timeout = timer.callback

            controller.runner_idle_seen = False
            clock.advance(0.5)
            timer.fire()
            self.assertTrue(controller.timer_expired)
            self.assertEqual(len(calls), 1)
            controller._on_runner_idle(runner.generation)
            self.assertTrue(_wait_until(lambda: timer.active))
            self.assertEqual(len(calls), 2)
            second_image = controller.current_presentation_image_id
            second_epoch = timer.epoch

            first_timeout(first_epoch)
            self.assertEqual(
                controller.current_presentation_image_id, second_image
            )
            self.assertEqual(timer.epoch, second_epoch)
            self.assertTrue(timer.active)

            clock.advance(0.5)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_presenting_pause_freezes_remaining_and_rejects_old_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            controller, runner, _manager, _presenter, timer, clock = (
                _visible_controller(
                    paths,
                    labels,
                    lambda request, _lease: _success(request),
                    delay=2.0,
                )
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(lambda: timer.active))
            stale_epoch = timer.epoch
            stale_timeout = timer.callback
            clock.advance(0.75)
            self.assertTrue(controller.request_pause())
            self.assertEqual(controller.phase, "PAUSED")
            self.assertAlmostEqual(
                controller.remaining_seconds, 1.25, places=6
            )
            self.assertFalse(timer.active)

            stale_timeout(stale_epoch)
            self.assertEqual(controller.phase, "PAUSED")
            self.assertAlmostEqual(
                controller.remaining_seconds, 1.25, places=6
            )
            self.assertFalse(controller.resume("KEEP_PAUSED"))
            self.assertTrue(controller.resume("CLEAN"))
            self.assertTrue(_wait_until(lambda: timer.active))
            self.assertEqual(timer.milliseconds, 1250)
            clock.advance(1.25)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_zero_remaining_resume_defers_completion_to_next_gui_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            controller, runner, _manager, _presenter, timer, clock = (
                _visible_controller(
                    paths,
                    labels,
                    lambda request, _lease: _success(request),
                    delay=0.5,
                )
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(lambda: timer.active))
            clock.advance(0.5)
            self.assertTrue(controller.request_pause())
            self.assertEqual(controller.phase, "PAUSED")
            self.assertEqual(controller.remaining_seconds, 0.0)

            self.assertTrue(controller.resume("CLEAN"))
            self.assertEqual(finished, [])
            self.assertTrue(_wait_until(lambda: timer.active))
            self.assertEqual(timer.milliseconds, 0)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_pause_during_loading_requeues_same_item_before_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            calls = []
            controller, runner, _manager, _presenter, timer, clock = (
                _visible_controller(
                    paths,
                    labels,
                    lambda request, _lease: (
                        calls.append(request.image_id) or _success(request)
                    ),
                    delay=0.5,
                )
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertEqual(controller.phase, "LOADING")
            self.assertTrue(controller.request_pause())
            self.assertTrue(_wait_until(lambda: controller.phase == "PAUSED"))
            self.assertEqual(calls, [])
            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(item["execution_status"], "queued")
            self.assertEqual(
                item["prediction_attempts"][-1]["status"],
                "cancelled_before_submit",
            )

            self.assertTrue(controller.resume("CLEAN"))
            self.assertTrue(_wait_until(lambda: timer.active))
            self.assertEqual(len(calls), 1)
            clock.advance(0.5)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_pause_during_inference_commits_and_presents_before_pausing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            entered = threading.Event()
            release = threading.Event()

            def execute(request, _lease):
                entered.set()
                release.wait(5)
                return _success(request)

            controller, runner, _manager, presenter, timer, clock = (
                _visible_controller(paths, labels, execute, delay=0.5)
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(entered.is_set))
            self.assertTrue(controller.request_pause())
            release.set()
            self.assertTrue(_wait_until(lambda: controller.phase == "PAUSED"))
            self.assertEqual(len(presenter.presented_image_ids), 1)
            self.assertAlmostEqual(controller.remaining_seconds, 0.5)
            self.assertFalse(timer.active)

            self.assertTrue(controller.resume("DISCARDED"))
            self.assertTrue(_wait_until(lambda: timer.active))
            clock.advance(0.5)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_presenting_stop_and_close_cancel_timer_without_next_item(self):
        for control in ("stop", "close"):
            with (
                self.subTest(control=control),
                tempfile.TemporaryDirectory() as tmp,
            ):
                paths, labels = _images(Path(tmp), 2)
                calls = []
                controller, runner, _manager, presenter, timer, _clock = (
                    _visible_controller(
                        paths,
                        labels,
                        lambda request, _lease: (
                            calls.append(request.image_id) or _success(request)
                        ),
                        delay=2.0,
                    )
                )
                finished = []
                safe = []
                controller.finished.connect(finished.append)
                controller.safe_to_close.connect(safe.append)
                controller.start()
                self.assertTrue(_wait_until(lambda: timer.active))
                if control == "stop":
                    self.assertTrue(controller.request_stop())
                else:
                    generation = controller.request_close()
                    self.assertTrue(_wait_until(lambda: safe == [generation]))
                self.assertTrue(_wait_until(lambda: bool(finished)))
                self.assertTrue(
                    _wait_until(lambda: runner.worker_thread is None)
                )
                self.assertFalse(timer.active)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(presenter.presented_image_ids), 1)
                self.assertEqual(finished[0]["processing_status"], "PARTIAL")

    def test_model_error_has_no_timer_and_retry_uses_same_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                if len(calls) == 1:
                    return PredictionOutcome.failed(
                        request,
                        "model_prediction_failed",
                        "retry",
                    )
                return _success(request)

            controller, runner, _manager, presenter, timer, clock = (
                _visible_controller(paths, labels, execute, delay=0.5)
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(
                _wait_until(lambda: controller.phase == "WAITING_ERROR")
            )
            self.assertFalse(timer.active)
            self.assertEqual(presenter.presented_image_ids, [])
            self.assertTrue(controller.retry_current())
            self.assertTrue(_wait_until(lambda: timer.active))
            self.assertEqual(calls, [calls[0], calls[0]])
            clock.advance(0.5)
            timer.fire()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

    def test_waiting_error_pause_resumes_same_error_without_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                return PredictionOutcome.failed(
                    request,
                    "model_prediction_failed",
                    "still failing",
                )

            controller, runner, _manager, _presenter, timer, _clock = (
                _visible_controller(paths, labels, execute, delay=0.5)
            )
            finished = []
            waiting = []
            controller.finished.connect(finished.append)
            controller.waiting_error.connect(waiting.append)
            controller.start()
            self.assertTrue(
                _wait_until(lambda: controller.phase == "WAITING_ERROR")
            )
            image_id = controller.waiting_image_id
            self.assertTrue(controller.request_pause())
            self.assertEqual(controller.phase, "PAUSED")
            self.assertFalse(timer.active)
            self.assertFalse(controller.resume("KEEP_PAUSED"))
            self.assertTrue(controller.resume("CLEAN"))
            self.assertEqual(controller.phase, "WAITING_ERROR")
            self.assertEqual(controller.waiting_image_id, image_id)
            self.assertEqual(calls, [image_id])
            self.assertEqual(waiting[-1]["image_id"], image_id)
            self.assertTrue(controller.skip_current())
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(finished[0]["explicit_error_skips"], 1)
            self.assertFalse(timer.active)

    def test_loading_stop_keeps_item_queued_and_never_submits(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            calls = []
            controller, runner, _manager, _presenter, timer, _clock = (
                _visible_controller(
                    paths,
                    labels,
                    lambda request, _lease: calls.append(request.image_id),
                    delay=0.5,
                )
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertEqual(controller.phase, "LOADING")
            self.assertTrue(controller.request_stop())
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(calls, [])
            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(item["execution_status"], "queued")
            self.assertEqual(
                item["prediction_attempts"][-1]["status"],
                "cancelled_before_submit",
            )
            self.assertEqual(finished[0]["processing_status"], "CANCELLED")
            self.assertFalse(timer.active)

    def test_input_failure_and_conflict_never_start_presentation_timer(self):
        for code, expected_status in (
            ("failed_input", "COMPLETED"),
            ("conflict_image_changed", "PARTIAL"),
        ):
            with (
                self.subTest(code=code),
                tempfile.TemporaryDirectory() as tmp,
            ):
                paths, labels = _images(Path(tmp), 1)
                controller, runner, _manager, presenter, timer, _clock = (
                    _visible_controller(
                        paths,
                        labels,
                        lambda request, _lease: PredictionOutcome.failed(
                            request,
                            code,
                            "not presentable",
                        ),
                        delay=0.5,
                    )
                )
                finished = []
                controller.finished.connect(finished.append)
                controller.start()
                self.assertTrue(_wait_until(lambda: bool(finished)))
                self.assertTrue(
                    _wait_until(lambda: runner.worker_thread is None)
                )
                self.assertFalse(timer.active)
                self.assertEqual(presenter.presented_image_ids, [])
                self.assertEqual(
                    finished[0]["processing_status"],
                    expected_status,
                )

    def test_presentation_failure_is_terminal_after_safe_commit(self):
        class FailingPresenter(_DiskPresenter):
            def present(self, *args):
                super().present(*args)
                raise FastControllerError(
                    "presentation_document_digest_mismatch"
                )

        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            calls = []
            presenter = FailingPresenter()
            controller, runner, _manager, _presenter, timer, _clock = (
                _visible_controller(
                    paths,
                    labels,
                    lambda request, _lease: (
                        calls.append(request.image_id) or _success(request)
                    ),
                    delay=0.5,
                    presenter=presenter,
                )
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(len(calls), 1)
            self.assertEqual(finished[0]["processing_status"], "FAILED")
            self.assertEqual(
                controller.hard_error["code"],
                "presentation_document_digest_mismatch",
            )
            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(item["staged_commit_status"], "committed")
            self.assertFalse(timer.active)

    def test_start_failure_deactivates_presenter(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            controller, runner, _manager, presenter, _timer, _clock = (
                _visible_controller(
                    paths,
                    labels,
                    lambda request, _lease: _success(request),
                    delay=0.5,
                )
            )
            runner.start = mock.Mock(return_value=False)
            with self.assertRaisesRegex(
                FastControllerError,
                "prediction_runner_start_failed",
            ):
                controller.start()
            self.assertEqual(presenter.deactivated, 1)


class PresentationAndDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_disk_presenter_rereads_without_dirty_save_or_item_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            label = root / "image.json"
            image.write_bytes(b"image")
            document = _valid_empty(image.name)
            atomic_write_label_document(
                label,
                document,
                pre_document_digest="MISSING",
            )
            resolved = resolve_existing_label(label)
            image_path = os.path.normcase(os.path.realpath(image))
            label_path = os.path.normcase(os.path.realpath(label))
            record = SimpleNamespace(
                image_id="image-a",
                canonical_session_image_path=image_path,
                canonical_session_label_path=label_path,
            )
            widget = SimpleNamespace(
                dirty=False,
                begin_sequence_presentation=mock.Mock(),
                load_sequence_image_for_presentation=mock.Mock(
                    return_value=True
                ),
                present_committed_sequence_document=mock.Mock(
                    return_value=True
                ),
                end_sequence_presentation=mock.Mock(return_value=True),
                set_dirty=mock.Mock(),
                save_file=mock.Mock(),
                save_labels=mock.Mock(),
            )
            presenter = LabelingWidgetSequencePresenter(widget)
            presenter.activate("token", {"image-a": record})
            presenter.load("token", 1, "image-a")
            item = {
                "image_id": "image-a",
                "item_revision": 9,
                "digests": {
                    "staged_document_digest": resolved.document_digest,
                    "staged_annotation_digest": resolved.semantic_digest,
                },
            }
            before_revision = item["item_revision"]
            shown = presenter.present("token", 1, "image-a", item)

            self.assertEqual(shown, document)
            self.assertEqual(item["item_revision"], before_revision)
            widget.set_dirty.assert_not_called()
            widget.save_file.assert_not_called()
            widget.save_labels.assert_not_called()
            self.assertFalse(widget.dirty)

    def test_presentation_tamper_and_missing_label_have_stable_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            label = root / "image.json"
            image.write_bytes(b"image")
            atomic_write_label_document(
                label,
                _valid_empty(image.name),
                pre_document_digest="MISSING",
            )
            resolved = resolve_existing_label(label)
            record = SimpleNamespace(
                image_id="image-a",
                canonical_session_image_path=os.path.normcase(
                    os.path.realpath(image)
                ),
                canonical_session_label_path=os.path.normcase(
                    os.path.realpath(label)
                ),
            )
            item = {
                "image_id": "image-a",
                "digests": {
                    "staged_document_digest": resolved.document_digest,
                    "staged_annotation_digest": resolved.semantic_digest,
                },
            }

            def tamper(*_args):
                changed = _valid_empty(image.name)
                changed["description"] = "tampered"
                atomic_write_label_document(
                    label,
                    changed,
                    pre_document_digest=resolve_existing_label(
                        label
                    ).document_digest,
                )
                return True

            widget = SimpleNamespace(
                dirty=False,
                begin_sequence_presentation=mock.Mock(),
                present_committed_sequence_document=mock.Mock(
                    side_effect=tamper
                ),
                end_sequence_presentation=mock.Mock(),
            )
            presenter = LabelingWidgetSequencePresenter(widget)
            presenter.activate("token", {"image-a": record})
            with self.assertRaises(FastControllerError) as raised:
                presenter.present("token", 1, "image-a", item)
            self.assertEqual(
                raised.exception.code,
                "presentation_document_digest_mismatch",
            )

            label.unlink()
            with self.assertRaises(FastControllerError) as raised:
                presenter.revalidate("token", "image-a", item)
            self.assertEqual(
                raised.exception.code, "presentation_label_missing"
            )

    def test_raw_widget_error_is_normalized_to_stable_presentation_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            label = root / "image.json"
            image.write_bytes(b"image")
            atomic_write_label_document(
                label,
                _valid_empty(image.name),
                pre_document_digest="MISSING",
            )
            resolved = resolve_existing_label(label)
            record = SimpleNamespace(
                image_id="image-a",
                canonical_session_image_path=os.path.normcase(
                    os.path.realpath(image)
                ),
                canonical_session_label_path=os.path.normcase(
                    os.path.realpath(label)
                ),
            )
            widget = SimpleNamespace(
                dirty=False,
                begin_sequence_presentation=mock.Mock(),
                present_committed_sequence_document=mock.Mock(
                    side_effect=ValueError("presentation_path_mismatch")
                ),
                end_sequence_presentation=mock.Mock(),
            )
            presenter = LabelingWidgetSequencePresenter(widget)
            presenter.activate("token", {"image-a": record})
            item = {
                "image_id": "image-a",
                "digests": {
                    "staged_document_digest": resolved.document_digest,
                    "staged_annotation_digest": resolved.semantic_digest,
                },
            }

            with self.assertRaises(FastControllerError) as raised:
                presenter.present("token", 1, "image-a", item)
            self.assertEqual(
                raised.exception.code,
                "presentation_path_mismatch",
            )

    def test_controlled_canvas_navigation_clears_zero_target_without_autosave(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            label = root / "image.json"
            image.write_bytes(b"image")
            atomic_write_label_document(
                label,
                _valid_empty(image.name),
                pre_document_digest="MISSING",
            )
            image_path = os.path.normcase(os.path.realpath(image))
            label_path = os.path.normcase(os.path.realpath(label))
            record = SimpleNamespace(
                image_id="image-a",
                canonical_session_image_path=image_path,
                canonical_session_label_path=label_path,
            )

            class CanvasWidget:
                _sequence_record_value = staticmethod(
                    LabelingWidget._sequence_record_value
                )
                _sequence_canonical_path = staticmethod(
                    LabelingWidget._sequence_canonical_path
                )
                begin_sequence_presentation = (
                    LabelingWidget.begin_sequence_presentation
                )
                _sequence_presentation_target = (
                    LabelingWidget._sequence_presentation_target
                )
                load_sequence_image_for_presentation = (
                    LabelingWidget.load_sequence_image_for_presentation
                )
                present_committed_sequence_document = (
                    LabelingWidget.present_committed_sequence_document
                )
                end_sequence_presentation = (
                    LabelingWidget.end_sequence_presentation
                )

                def __init__(self):
                    self.image_list = [image_path]
                    self.fn_to_index = {image_path: 0}
                    self.file_list_widget = QtWidgets.QListWidget()
                    self.file_list_widget.addItem(image_path)
                    self._config = {"keep_prev": True, "auto_save": True}
                    self.filename = None
                    self.canvas = SimpleNamespace(
                        shapes=[object()],
                        shapes_backups=[object()],
                    )
                    self.actions = SimpleNamespace(
                        undo=QtWidgets.QAction(None)
                    )
                    self.set_dirty = mock.Mock()
                    self.save_labels = mock.Mock()
                    self.clean_calls = 0

                def load_file(self, target):
                    self.filename = target
                    self.canvas.shapes = json.loads(
                        label.read_text(encoding="utf-8")
                    )["shapes"]
                    return True

                def get_label_file(self):
                    return label_path

                def set_clean(self):
                    self.clean_calls += 1

            widget = CanvasWidget()
            widget.begin_sequence_presentation("token", {"image-a": record})
            self.assertTrue(
                widget.present_committed_sequence_document(
                    "token",
                    1,
                    "image-a",
                    image_path,
                    label_path,
                )
            )
            self.assertEqual(widget.filename, image_path)
            self.assertEqual(widget.file_list_widget.currentRow(), 0)
            self.assertEqual(widget.canvas.shapes, [])
            self.assertEqual(widget.canvas.shapes_backups, [])
            self.assertFalse(widget.actions.undo.isEnabled())
            self.assertTrue(widget._config["keep_prev"])
            widget.set_dirty.assert_not_called()
            widget.save_labels.assert_not_called()

    def _run_mode(self, root, visible):
        paths, labels = _images(root, 1)
        initial = _valid_empty(Path(paths[0]).name)
        initial.update(
            flags={"reviewed": True},
            description="existing",
            custom={"kept": [1, 2]},
            shapes=[
                {
                    "label": "existing",
                    "points": [[0, 0], [1, 1]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
            ],
        )
        target = labels / (Path(paths[0]).stem + ".json")
        atomic_write_label_document(
            target,
            initial,
            pre_document_digest="MISSING",
        )
        manager = _Manager()
        runner = PredictionRunner(
            manager,
            request_executor=lambda request, _lease: _success(
                request, target_count=1
            ),
        )
        presenter = _DiskPresenter() if visible else None
        timer = _ManualPresentationTimer() if visible else None
        clock = _MonotonicClock() if visible else None
        options = (
            _sequence_options(2.0, write_policy="FORCE_MERGE")
            if visible
            else _options(write_policy="FORCE_MERGE")
        )
        controller_class = (
            ContinuousAutoLabelingController
            if visible
            else FastAutoLabelingController
        )
        controller = controller_class(
            runner,
            options,
            standalone_image_paths=paths,
            standalone_output_dir=str(labels),
            presentation_adapter=presenter,
            presentation_timer=timer,
            monotonic_ns=clock,
        )
        finished = []
        controller.finished.connect(finished.append)
        controller.start()
        if visible:
            self.assertTrue(_wait_until(lambda: timer.active))
            clock.advance(2.0)
            timer.fire()
        self.assertTrue(_wait_until(lambda: bool(finished)))
        self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
        item = controller.run_store.list_items(controller.run_id)[0]
        return resolve_existing_label(target), item, presenter

    def test_fast_visible_differential_has_identical_label_facts(self):
        with (
            tempfile.TemporaryDirectory() as fast_tmp,
            tempfile.TemporaryDirectory() as visible_tmp,
        ):
            fast, fast_item, _ = self._run_mode(Path(fast_tmp), False)
            visible, visible_item, presenter = self._run_mode(
                Path(visible_tmp), True
            )

            self.assertEqual(fast.document_digest, visible.document_digest)
            self.assertEqual(fast.semantic_digest, visible.semantic_digest)
            for field in (
                "shapes",
                "description",
                "flags",
                "custom",
            ):
                self.assertEqual(fast.document[field], visible.document[field])
            self.assertEqual(
                fast_item["digests"]["staged_document_digest"],
                visible_item["digests"]["staged_document_digest"],
            )
            self.assertEqual(
                fast_item["digests"]["staged_annotation_digest"],
                visible_item["digests"]["staged_annotation_digest"],
            )
            for field in (
                "target_count",
                "zero_target",
                "semantic_change",
            ):
                self.assertEqual(
                    fast_item["result_summary"][field],
                    visible_item["result_summary"][field],
                )
            self.assertEqual(len(presenter.presented_image_ids), 1)


class VisibleCanvasGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_paused_unlocks_editing_but_keeps_navigation_frozen(self):
        class Control:
            def __init__(self):
                self.enabled = True

            def isEnabled(self):
                return self.enabled

            def setEnabled(self, enabled):
                self.enabled = bool(enabled)

        class FileList(Control):
            def currentRow(self):
                return 0

            def setCurrentRow(self, _row):
                pass

        image = os.path.abspath("visible-current.png")
        widget = SimpleNamespace(
            filename=image,
            file_list_widget=FileList(),
            zoom_mode=0,
            zoom_widget=None,
            scroll_bars={},
            auto_labeling_host_context=None,
            canvas=SimpleNamespace(
                selected_shapes=[],
                set_sequence_edit_locked=mock.Mock(),
            ),
            auto_labeling_widget=Control(),
            label_list=Control(),
            unique_label_list=Control(),
            flag_widget=Control(),
            actions=SimpleNamespace(
                save=Control(),
                edit=Control(),
                zoom_in=Control(),
                zoom_out=Control(),
                open_next_image=Control(),
                open_prev_image=Control(),
                open_next_unchecked_image=Control(),
                open_prev_unchecked_image=Control(),
            ),
            load_file=mock.Mock(return_value=True),
        )
        guard = VisibleCanvasStateGuard(widget)
        guard.set_running(True)
        self.assertFalse(widget.actions.edit.enabled)
        self.assertFalse(widget.file_list_widget.enabled)
        self.assertTrue(widget.actions.zoom_in.enabled)
        self.assertTrue(widget.actions.zoom_out.enabled)
        guard.set_paused(True)
        self.assertTrue(widget.actions.edit.enabled)
        self.assertTrue(widget.actions.save.enabled)
        self.assertFalse(widget.file_list_widget.enabled)
        self.assertFalse(widget.actions.open_next_image.enabled)
        self.assertFalse(widget.actions.open_prev_image.enabled)
        guard.finish(True)
        widget.load_file.assert_not_called()

    def test_no_presented_item_restores_the_original_canvas(self):
        class Control:
            def __init__(self):
                self.enabled = True

            def isEnabled(self):
                return self.enabled

            def setEnabled(self, enabled):
                self.enabled = bool(enabled)

        class FileList(Control):
            def currentRow(self):
                return 2

            def setCurrentRow(self, _row):
                pass

        original = os.path.abspath("visible-original.png")
        widget = SimpleNamespace(
            filename=original,
            file_list_widget=FileList(),
            zoom_mode=0,
            zoom_widget=None,
            scroll_bars={},
            auto_labeling_host_context=None,
            canvas=SimpleNamespace(
                selected_shapes=[],
                set_sequence_edit_locked=mock.Mock(),
            ),
            auto_labeling_widget=Control(),
            label_list=Control(),
            unique_label_list=Control(),
            flag_widget=Control(),
            actions=SimpleNamespace(),
            load_file=mock.Mock(return_value=True),
        )
        guard = VisibleCanvasStateGuard(widget)
        widget.filename = os.path.abspath("visible-uncommitted.png")
        guard.finish(False)
        widget.load_file.assert_called_once_with(original)


if __name__ == "__main__":
    unittest.main()
