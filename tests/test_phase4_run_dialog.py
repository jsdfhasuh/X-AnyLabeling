import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt5 import QtCore, QtWidgets

from anylabeling.views.labeling.utils import batch
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_boolean_text_v1,
    auto_labeling_status_text_v1,
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    FastControllerError,
)
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)
from anylabeling.views.labeling.widgets.auto_labeling_run_dialog import (
    FastRunProgressDialog,
    FastRunSetupDialog,
    FastRunUiSession,
    continuous_start_failure_text_v1,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _ReviewWorkerThread(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, running):
        super().__init__()
        self.running = running

    def isRunning(self):
        return self.running


class FastRunDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_setup_exposes_only_zero_delay_fast_options(self):
        dialog = FastRunSetupDialog(
            {
                "workset_total": 151,
                "existing_annotation": 12,
                "host_prepare_failed": 3,
            },
            "YOLOv8 - conf=0.25 - IOU=0.45",
            current_anchor_available=True,
        )
        try:
            self.assertEqual(
                dialog.selected_values(),
                {
                    "range": "CURRENT_TO_END",
                    "filter": "ALL",
                    "write_policy": "SKIP_EXISTING",
                },
            )
            label_text = "\n".join(
                label.text() for label in dialog.findChildren(QtWidgets.QLabel)
            )
            self.assertEqual(dialog.delay_spin.value(), 0.0)
            self.assertFalse(dialog.delay_spin.isEnabled())
            self.assertIn("151", label_text)
            self.assertFalse(dialog.start_button.icon().isNull())
        finally:
            dialog.close()

    def test_active_process_error_does_not_advertise_persistent_resume(self):
        error = FastControllerError(
            "fast_controller_already_active",
            "current_process_sequence_is_active",
        )

        message = continuous_start_failure_text_v1(error)

        self.assertIn("fast_controller_already_active", message)
        self.assertNotIn("Process remaining auto-labeling", message)
        self.assertNotIn("Saved results", message)
        self.assertIn("current_process_sequence_is_active", message)

    def test_progress_controls_and_all_skipped_terminal_state(self):
        dialog = FastRunProgressDialog()
        pause = []
        resume = []
        dialog.pause_requested.connect(lambda: pause.append(True))
        dialog.resume_requested.connect(lambda: resume.append(True))
        try:
            self.assertEqual(
                dialog.pause_button.text(), auto_labeling_text_v1("pause")
            )
            dialog.pause_button.click()
            self.assertEqual(pause, [True])
            self.assertFalse(dialog.pause_button.isEnabled())
            self.assertEqual(
                dialog.state_label.text(),
                auto_labeling_text_v1("phase_pausing"),
            )
            dialog.set_phase("INFERENCING")
            self.assertFalse(dialog.pause_button.isEnabled())
            self.assertEqual(
                dialog.state_label.text(),
                auto_labeling_text_v1("phase_pausing"),
            )
            dialog.set_phase("PAUSED")
            self.assertTrue(dialog.pause_button.isEnabled())
            self.assertEqual(
                dialog.state_label.text(),
                auto_labeling_text_v1("phase_paused"),
            )
            dialog.pause_button.click()
            self.assertEqual(resume, [True])

            dialog.update_progress(
                {
                    "processed": 7,
                    "total": 10,
                    "current_filename": "image.jpg",
                    "succeeded": 2,
                    "zero_target": 1,
                    "remaining": 3,
                }
            )
            self.assertEqual(dialog.progress_bar.value(), 7)
            self.assertEqual(dialog.progress_bar.format(), "7 / 10")
            self.assertEqual(dialog.file_label.text(), "image.jpg")
            self.assertEqual(dialog.metric_labels["succeeded"].text(), "2")
            metric_names = {
                label.text() for label in dialog.findChildren(QtWidgets.QLabel)
            }
            self.assertIn(auto_labeling_text_v1("succeeded"), metric_names)

            dialog.show_waiting_error(
                {
                    "error_code": "model_prediction_failed",
                    "error_message": "oom",
                }
            )
            self.assertFalse(dialog.retry_button.isHidden())
            self.assertFalse(dialog.skip_button.isHidden())
            dialog.finish_run(
                {
                    "workset_total": 10,
                    "remaining": 0,
                    "eligible_for_inference": 0,
                    "completed_with_errors": False,
                    "processing_status": "COMPLETED",
                }
            )
            self.assertEqual(
                dialog.state_label.text(),
                auto_labeling_text_v1("nothing_to_process"),
            )
        finally:
            dialog.close()

    def test_progress_dialog_prefers_the_parent_window_edge(self):
        available = QtCore.QRect(0, 0, 1920, 1080)
        dialog_size = QtCore.QSize(520, 400)

        owner = QtCore.QRect(100, 100, 800, 700)
        outside = FastRunProgressDialog._edge_position(
            owner, dialog_size, available
        )
        self.assertGreater(outside.x(), owner.right())
        self.assertEqual(outside.y(), owner.top() + 16)

        maximized = QtCore.QRect(0, 0, 1920, 1080)
        inside = FastRunProgressDialog._edge_position(
            maximized, dialog_size, available
        )
        self.assertGreater(inside.x(), maximized.center().x())
        self.assertLessEqual(
            inside.x() + dialog_size.width() - 1,
            available.right() - 16,
        )

    def test_rejected_pause_request_restores_the_running_controls(self):
        dialog = FastRunProgressDialog()
        controller = SimpleNamespace(
            request_pause=mock.Mock(return_value=False)
        )
        session = SimpleNamespace(controller=controller, progress=dialog)
        dialog.pause_requested.connect(
            lambda: FastRunUiSession._pause(session)
        )
        try:
            dialog.set_phase("INFERENCING")
            dialog.pause_button.click()
            controller.request_pause.assert_called_once_with()
            self.assertTrue(dialog.pause_button.isEnabled())
            self.assertEqual(
                dialog.state_label.text(),
                auto_labeling_text_v1("phase_inferencing"),
            )
        finally:
            dialog.close()

    def test_progress_uses_absolute_pending_review_count(self):
        progress = SimpleNamespace(update_progress=mock.Mock())
        pending_setter = mock.Mock()
        session = SimpleNamespace(
            progress=progress,
            labeling_widget=SimpleNamespace(
                set_auto_labeling_pending_review_count=pending_setter,
            ),
        )
        summary = {"processed": 18, "pending_review": 18}

        FastRunUiSession._on_progress_changed(session, summary)
        FastRunUiSession._on_progress_changed(session, summary)

        self.assertEqual(progress.update_progress.call_count, 2)
        pending_setter.assert_has_calls(
            [
                mock.call(18, scope="session"),
                mock.call(18, scope="session"),
            ]
        )

        for invalid in (-1, True, "18", None):
            FastRunUiSession._on_progress_changed(
                session,
                {"pending_review": invalid},
            )
        self.assertEqual(pending_setter.call_count, 2)

    def test_verified_commit_checks_only_the_matching_file_list_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.normcase(
                os.path.realpath(os.path.join(tmp, "first.png"))
            )
            second = os.path.normcase(
                os.path.realpath(os.path.join(tmp, "second.png"))
            )
            for path in (first, second):
                Path(path).write_bytes(b"image")
            file_list = QtWidgets.QListWidget()
            for path in (first, second):
                item = QtWidgets.QListWidgetItem(path)
                item.setCheckState(QtCore.Qt.Unchecked)
                file_list.addItem(item)
            record = SimpleNamespace(
                image_id="image-a",
                canonical_session_image_path=first,
            )
            audit_state = {"review_status": "pending"}
            widget = SimpleNamespace(
                image_list=[first, second],
                fn_to_index={first: 0, second: 1},
                file_list_widget=file_list,
                audit_state=audit_state,
            )
            controller = SimpleNamespace(
                run_id="run-a", records_by_id={"image-a": record}
            )
            session = SimpleNamespace(
                controller=controller, labeling_widget=widget
            )
            event = {
                "run_id": "run-a",
                "image_id": "image-a",
                "attempt_id": "attempt-a",
                "canonical_image_path": first,
                "staged_document_digest": "aldoc1:" + "a" * 64,
            }

            FastRunUiSession._on_label_committed(session, event)
            self.assertEqual(file_list.item(0).checkState(), QtCore.Qt.Checked)
            self.assertEqual(
                file_list.item(1).checkState(), QtCore.Qt.Unchecked
            )
            self.assertEqual(audit_state, {"review_status": "pending"})

            ignored = [
                {**event, "run_id": "stale-run"},
                {**event, "image_id": "wrong-image"},
                {**event, "canonical_image_path": second},
                {**event, "staged_document_digest": "invalid"},
            ]
            for stale_event in ignored:
                file_list.item(0).setCheckState(QtCore.Qt.Unchecked)
                FastRunUiSession._on_label_committed(session, stale_event)
                self.assertEqual(
                    file_list.item(0).checkState(), QtCore.Qt.Unchecked
                )

            widget.fn_to_index[first] = 1
            FastRunUiSession._on_label_committed(session, event)
            self.assertEqual(
                file_list.item(0).checkState(), QtCore.Qt.Unchecked
            )
            self.assertEqual(audit_state, {"review_status": "pending"})


class FastRunRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    @staticmethod
    def _widget(model_type, opener):
        manager = SimpleNamespace(
            loaded_model_config={"type": model_type, "model": object()},
            new_model_status=SimpleNamespace(emit=mock.Mock()),
        )
        return SimpleNamespace(
            image_list=["image.jpg"],
            auto_labeling_widget=SimpleNamespace(
                model_manager=manager,
                open_continuous_auto_labeling=opener,
            ),
            tr=lambda value: value,
        )

    def test_supported_auto_run_and_button_share_fast_entry_without_fallback(
        self,
    ):
        opener = mock.Mock(return_value=False)
        widget = self._widget("yolov8", opener)
        with mock.patch.object(batch, "run_all_images_legacy") as legacy:
            self.assertFalse(batch.run_all_images(widget))
        opener.assert_called_once_with()
        legacy.assert_not_called()

        owner = SimpleNamespace(parent=widget)
        with mock.patch.object(
            batch, "run_all_images", return_value="same-entry"
        ) as routed:
            result = AutoLabelingWidget.run_continuous_auto_labeling(owner)
        self.assertEqual(result, "same-entry")
        routed.assert_called_once_with(widget)

    def test_supported_empty_visible_list_still_uses_fast_entry(self):
        opener = mock.Mock(return_value="fast")
        widget = self._widget("yolov8", opener)
        widget.image_list = []
        widget.auto_labeling_widget.auto_labeling_host_context = object()

        with mock.patch.object(batch, "run_all_images_legacy") as legacy:
            self.assertEqual(batch.run_all_images(widget), "fast")

        opener.assert_called_once_with()
        legacy.assert_not_called()

    def test_final_summary_exposes_every_required_gate_metric(self):
        dialog = FastRunProgressDialog()
        summary = {
            "workset_total": 17,
            "selected_by_range": 16,
            "eligible_for_inference": 12,
            "succeeded": 8,
            "zero_target": 3,
            "skipped_existing": 2,
            "skipped_outside_range": 1,
            "host_prepare_failed": 1,
            "failed_input": 1,
            "model_failed_unresolved": 1,
            "explicit_error_skips": 1,
            "conflicts": 1,
            "pending_review": 8,
            "remaining": 1,
            "processing_status": "PARTIAL",
            "completed_with_errors": True,
        }
        try:
            dialog.finish_run(summary)
            for field, value in summary.items():
                with self.subTest(field=field):
                    self.assertIn(field, dialog.metric_labels)
                    expected = value
                    if field == "processing_status":
                        expected = auto_labeling_status_text_v1(
                            value,
                            "processing",
                        )
                    elif field == "completed_with_errors":
                        expected = auto_labeling_boolean_text_v1(value)
                    self.assertEqual(
                        dialog.metric_labels[field].text(),
                        str(expected),
                    )
        finally:
            dialog.close()

    def test_legacy_only_model_stays_on_legacy_path(self):
        widget = self._widget("yolov8_det_track", mock.Mock())
        with mock.patch.object(
            batch, "run_all_images_legacy", return_value="legacy"
        ) as legacy:
            self.assertEqual(batch.run_all_images(widget), "legacy")
        legacy.assert_called_once_with(widget)
        widget.auto_labeling_widget.open_continuous_auto_labeling.assert_not_called()

    def test_dirty_save_discard_and_cancel_are_explicit(self):
        widget = SimpleNamespace(
            dirty=True,
            filename="image.jpg",
            save_file=mock.Mock(),
            set_clean=mock.Mock(),
            load_file=mock.Mock(return_value=True),
        )
        session = SimpleNamespace(
            labeling_widget=widget,
            auto_widget=SimpleNamespace(auto_labeling_host_context=None),
            tr=lambda value: value,
        )

        def save():
            widget.dirty = False

        widget.save_file.side_effect = save
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Save,
        ):
            self.assertEqual(
                FastRunUiSession._resolve_dirty(session, "start"), "SAVED"
            )

        widget.dirty = True
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Discard,
        ):
            self.assertEqual(
                FastRunUiSession._resolve_dirty(session, "resume"),
                "DISCARDED",
            )
        widget.set_clean.assert_called_once_with()
        widget.load_file.assert_called_once_with("image.jpg")

        widget.dirty = True
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.Cancel,
        ):
            self.assertIsNone(
                FastRunUiSession._resolve_dirty(session, "resume")
            )

        widget.dirty = True
        widget.set_dirty = mock.Mock()
        widget.load_file.return_value = False
        with (
            mock.patch.object(
                QtWidgets.QMessageBox,
                "question",
                return_value=QtWidgets.QMessageBox.Discard,
            ),
            self.assertRaisesRegex(
                FastControllerError,
                "manual_discard_reload_failed",
            ),
        ):
            FastRunUiSession._resolve_dirty(session, "resume")
        widget.set_dirty.assert_called_once_with()

    def test_paused_save_uses_unified_commit_bridge_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            label_path = root / "image.json"
            image_path.write_bytes(b"image")
            record = SimpleNamespace(
                image_id="image-a",
                canonical_session_image_path=os.path.normcase(
                    os.path.realpath(image_path)
                ),
                canonical_session_label_path=os.path.normcase(
                    os.path.realpath(label_path)
                ),
                source_image_digest="a" * 64,
            )
            sink = SimpleNamespace(publish=mock.Mock(return_value="APPLIED"))
            store = SimpleNamespace(
                read_item=mock.Mock(
                    return_value={
                        "item_revision": 7,
                        "latest_attempt_id": "attempt-a",
                    }
                )
            )
            context = SimpleNamespace(
                project_id="project-a",
                active_session_id="session-a",
                image_records_by_path={
                    record.canonical_session_image_path: record
                },
                annotation_item_store=store,
                annotation_commit_sink=sink,
            )
            widget = SimpleNamespace(
                dirty=True,
                filename=str(image_path),
                label_file=SimpleNamespace(filename=str(label_path)),
                set_dirty=mock.Mock(),
            )

            event = {
                "writer_kind": "MANUAL_SAVE",
                "mutation_mode": "APPLY_MANUAL_REVISION",
                "base_item_revision": 7,
                "image_id": "image-a",
            }

            def save():
                label_path.write_text(
                    json.dumps(
                        {
                            "version": "3.3.7",
                            "flags": {},
                            "shapes": [],
                            "imagePath": "image.png",
                            "imageData": None,
                            "imageHeight": 2,
                            "imageWidth": 2,
                            "description": "",
                        }
                    ),
                    encoding="utf-8",
                )
                sink.publish(
                    event,
                    label_path=record.canonical_session_label_path,
                )
                widget.dirty = False
                return True

            widget.save_file = mock.Mock(side_effect=save)
            session = SimpleNamespace(
                labeling_widget=widget,
                auto_widget=SimpleNamespace(
                    auto_labeling_host_context=context
                ),
                tr=lambda value: value,
            )
            with mock.patch.object(
                QtWidgets.QMessageBox,
                "question",
                return_value=QtWidgets.QMessageBox.Save,
            ):
                self.assertEqual(
                    FastRunUiSession._resolve_dirty(session, "resume"),
                    "SAVED",
                )
            sink.publish.assert_called_once()
            published = sink.publish.call_args.args[0]
            self.assertEqual(published["writer_kind"], "MANUAL_SAVE")
            self.assertEqual(
                published["mutation_mode"], "APPLY_MANUAL_REVISION"
            )
            self.assertEqual(published["base_item_revision"], 7)
            self.assertEqual(published["image_id"], "image-a")
            self.assertEqual(
                sink.publish.call_args.kwargs["label_path"],
                record.canonical_session_label_path,
            )

            widget.dirty = True

            def failed_save():
                widget.dirty = False
                widget.auto_labeling_commit_blocked = True
                return True

            widget.save_file.side_effect = failed_save
            with (
                mock.patch.object(
                    QtWidgets.QMessageBox,
                    "question",
                    return_value=QtWidgets.QMessageBox.Save,
                ),
                self.assertRaisesRegex(
                    FastControllerError,
                    "manual_save_failed",
                ),
            ):
                FastRunUiSession._resolve_dirty(session, "resume")
            self.assertFalse(widget.dirty)
            widget.set_dirty.assert_not_called()

    def test_paused_resume_replays_pending_commit_before_clean_shortcut(self):
        widget = SimpleNamespace(
            dirty=False,
            auto_labeling_commit_blocked=True,
        )

        def replay():
            widget.auto_labeling_commit_blocked = False
            return "IDEMPOTENT_NO_OP"

        widget.annotation_commit_bridge = SimpleNamespace(
            integrity_refresh=mock.Mock(side_effect=replay)
        )
        session = SimpleNamespace(
            labeling_widget=widget,
            tr=lambda value: value,
        )

        self.assertEqual(
            FastRunUiSession._resolve_dirty(session, "resume"),
            "CLEAN",
        )
        widget.annotation_commit_bridge.integrity_refresh.assert_called_once_with()

    def test_paused_resume_never_runs_while_pending_replay_is_blocked(self):
        widget = SimpleNamespace(
            dirty=False,
            auto_labeling_commit_blocked=True,
            annotation_commit_bridge=SimpleNamespace(
                integrity_refresh=mock.Mock(
                    side_effect=RuntimeError("checkpoint unavailable")
                )
            ),
        )
        controller = SimpleNamespace(resume=mock.Mock())
        progress = SimpleNamespace(clear_waiting_error=mock.Mock())
        session = SimpleNamespace(
            labeling_widget=widget,
            controller=controller,
            progress=progress,
            tr=lambda value: value,
            _show_control_failure=mock.Mock(),
        )

        FastRunUiSession._resume(session)

        session._show_control_failure.assert_called_once()
        controller.resume.assert_not_called()
        progress.clear_waiting_error.assert_not_called()

    def test_retry_and_skip_controls_remain_visible_until_runner_idle(self):
        progress = SimpleNamespace(clear_waiting_error=mock.Mock())
        controller = SimpleNamespace(
            retry_current=mock.Mock(return_value=False),
            skip_current=mock.Mock(return_value=False),
        )
        session = SimpleNamespace(progress=progress, controller=controller)

        FastRunUiSession._retry(session)
        FastRunUiSession._skip(session)
        progress.clear_waiting_error.assert_not_called()

        controller.retry_current.return_value = True
        controller.skip_current.return_value = True
        FastRunUiSession._retry(session)
        FastRunUiSession._skip(session)
        self.assertEqual(progress.clear_waiting_error.call_count, 2)

    def _review_session(self, *, worker_running):
        widget = SimpleNamespace(
            actions=SimpleNamespace(),
            start_auto_labeling_review=mock.Mock(return_value=True),
        )
        context = SimpleNamespace(activate_sequence_run=mock.Mock())
        controller = object()
        auto_widget = SimpleNamespace(
            prediction_runner=controller,
            _fast_run_session=None,
            auto_labeling_host_context=context,
            refresh_continuous_run_availability=mock.Mock(),
        )
        session = FastRunUiSession(widget, None)
        auto_widget._fast_run_session = session
        session.auto_widget = auto_widget
        session.controller = controller
        session.progress = SimpleNamespace(accept=mock.Mock())
        thread = _ReviewWorkerThread(worker_running)
        session.runner = SimpleNamespace(worker_thread=thread)
        session._audit_client = object()
        session.runner_class = mock.Mock()
        session.controller_class = mock.Mock()
        session._connect_thread_cleanup()
        return session, thread, widget, auto_widget, context

    def test_immediate_review_waits_for_worker_cleanup_and_starts_once(self):
        session, thread, widget, auto_widget, context = self._review_session(
            worker_running=True
        )

        self.assertTrue(session._review_now())
        self.assertFalse(session._review_now())
        widget.start_auto_labeling_review.assert_not_called()

        thread.running = False
        thread.finished.emit()
        thread.finished.emit()

        widget.start_auto_labeling_review.assert_called_once_with(
            session._audit_client
        )
        session.progress.accept.assert_called_once_with()
        self.assertIsNone(auto_widget.prediction_runner)
        self.assertIsNone(auto_widget._fast_run_session)
        session.runner_class.assert_not_called()
        session.controller_class.assert_not_called()
        context.activate_sequence_run.assert_not_called()

    def test_immediate_review_starts_at_once_when_worker_has_stopped(self):
        session, thread, widget, _auto_widget, context = self._review_session(
            worker_running=False
        )

        self.assertTrue(session._review_now())
        self.assertFalse(session._review_now())
        thread.finished.emit()

        widget.start_auto_labeling_review.assert_called_once_with(
            session._audit_client
        )
        session.progress.accept.assert_called_once_with()
        session.runner_class.assert_not_called()
        session.controller_class.assert_not_called()
        context.activate_sequence_run.assert_not_called()

    def test_review_later_closes_progress_without_starting_review(self):
        session, thread, widget, _auto_widget, context = self._review_session(
            worker_running=True
        )

        self.assertTrue(session._review_later())
        self.assertFalse(session._review_later())
        self.assertFalse(session._review_now())
        thread.running = False
        thread.finished.emit()

        session.progress.accept.assert_called_once_with()
        widget.start_auto_labeling_review.assert_not_called()
        session.runner_class.assert_not_called()
        session.controller_class.assert_not_called()
        context.activate_sequence_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
