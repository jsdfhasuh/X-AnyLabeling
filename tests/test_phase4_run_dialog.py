import unittest
from types import SimpleNamespace
from unittest import mock

from PyQt5 import QtWidgets

from anylabeling.views.labeling.utils import batch
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)
from anylabeling.views.labeling.widgets.auto_labeling_run_dialog import (
    FastRunProgressDialog,
    FastRunSetupDialog,
    FastRunUiSession,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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
                    "write_policy": "INHERIT_MODEL_POLICY",
                },
            )
            label_text = "\n".join(
                label.text() for label in dialog.findChildren(QtWidgets.QLabel)
            )
            self.assertIn("0.0", label_text)
            self.assertIn("151", label_text)
            self.assertFalse(dialog.start_button.icon().isNull())
        finally:
            dialog.close()

    def test_progress_controls_and_all_skipped_terminal_state(self):
        dialog = FastRunProgressDialog()
        pause = []
        resume = []
        dialog.pause_requested.connect(lambda: pause.append(True))
        dialog.resume_requested.connect(lambda: resume.append(True))
        try:
            dialog.pause_button.click()
            self.assertEqual(pause, [True])
            dialog.set_phase("PAUSED")
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
            self.assertEqual(dialog.state_label.text(), dialog.tr("无需处理"))
        finally:
            dialog.close()


class FastRunRoutingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
