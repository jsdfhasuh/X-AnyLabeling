import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from anylabeling.views.labeling.utils.empty_annotation import (
    AnnotationStatusDelegate,
    EmptyAnnotationController,
    STATUS_ROLE,
)


class ConfirmEmptyAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.widget = QtWidgets.QWidget()
        w = self.widget
        w.file_list_widget = QtWidgets.QListWidget(w)
        w.filename = str(self.root / "first.png")
        w.file_list_widget.addItems([w.filename, str(self.root / "last.png")])
        w.file_list_widget.setCurrentRow(0)
        w.fn_to_index = {
            w.file_list_widget.item(i).text(): i for i in range(2)
        }
        w.canvas = QtWidgets.QWidget(w)
        w.canvas.shapes = []
        w.canvas.current = None
        w.label_list = []
        w.image = QtGui.QImage(8, 8, QtGui.QImage.Format_RGB888)
        w.dirty = False
        w.auto_labeling_commit_blocked = False
        self.record = SimpleNamespace(
            image_id="first",
            canonical_session_label_path=str(self.root / "first.json"),
        )
        self.status = "unlabeled"
        self.context = SimpleNamespace(
            confirm_empty_annotations_enabled=True,
            session_purpose="labeling", images_ready=True,
            annotation_session_lease=SimpleNamespace(released=False),
            image_records_by_path={
                os.path.normcase(os.path.realpath(w.filename)): self.record
            },
            read_annotation_display_status=lambda _id: self.status,
        )
        w.auto_labeling_host_context = self.context
        w.status = mock.Mock()
        w.save_file = mock.Mock(side_effect=self.save)
        w.open_next_image = mock.Mock(side_effect=self.next)
        self.controller = EmptyAnnotationController(w, {})
        self.controller.start()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.temp.cleanup()

    def save(self):
        Path(self.record.canonical_session_label_path).write_text(
            json.dumps({
                "shapes": [], "imagePath": "first.png",
                "imageHeight": 8, "imageWidth": 8,
            }),
            encoding="utf-8",
        )
        self.status = "confirmed_empty"
        return True

    def next(self):
        self.widget.filename = self.widget.file_list_widget.item(1).text()

    def test_clean_blank_image_saves_then_advances_once(self):
        self.assertTrue(self.controller.action.isEnabled())
        self.assertEqual(self.controller.action.shortcut().toString(), "Alt+N")
        self.assertFalse(self.controller.action.autoRepeat())
        self.assertTrue(self.controller.confirm())
        self.widget.save_file.assert_called_once()
        self.widget.open_next_image.assert_called_once()
        self.assertEqual(
            self.widget.file_list_widget.item(0).data(STATUS_ROLE),
            "confirmed_empty",
        )
        self.assertFalse(self.controller.confirm())

    def test_save_failure_never_advances(self):
        self.widget.save_file.side_effect = None
        self.widget.save_file.return_value = False
        self.assertFalse(self.controller.confirm())
        self.widget.open_next_image.assert_not_called()
        self.assertEqual(self.widget.filename, str(self.root / "first.png"))

    def test_checkpoint_failure_is_not_file_success(self):
        def failed_checkpoint():
            self.save()
            self.widget.auto_labeling_commit_blocked = True
            return False
        self.widget.save_file.side_effect = failed_checkpoint
        self.assertFalse(self.controller.confirm())
        self.assertTrue(
            Path(self.record.canonical_session_label_path).exists()
        )
        self.widget.open_next_image.assert_not_called()

    def test_targets_and_unfinished_shapes_never_get_cleared(self):
        for attribute, value in (
            ("shapes", [object()]), ("current", object())
        ):
            with self.subTest(attribute=attribute):
                setattr(self.widget.canvas, attribute, value)
                self.assertFalse(self.controller.confirm())
                setattr(
                    self.widget.canvas, attribute,
                    [] if attribute == "shapes" else None,
                )
        self.widget.label_list = [object()]
        self.assertFalse(self.controller.confirm())
        self.widget.save_file.assert_not_called()

    def test_modes_and_lost_lease_fail_closed(self):
        inference_lease = SimpleNamespace(is_active=False)
        self.widget.auto_labeling_widget = SimpleNamespace(
            model_manager=SimpleNamespace(inference_lease=inference_lease)
        )
        for obj, attribute, value in (
            (self.context, "confirm_empty_annotations_enabled", False),
            (self.context, "session_purpose", "review"),
            (self.context, "images_ready", False),
            (self.context.annotation_session_lease, "released", True),
            (self.widget.canvas, "is_loading", True),
            (self.widget, "auto_labeling_commit_blocked", True),
            (inference_lease, "is_active", True),
        ):
            with self.subTest(attribute=attribute):
                old = getattr(obj, attribute, None)
                setattr(obj, attribute, value)
                self.assertFalse(self.controller.confirm())
                setattr(obj, attribute, old)
        self.widget.save_file.assert_not_called()

    def test_navigation_failure_preserves_successful_save_feedback(self):
        self.widget.open_next_image.side_effect = OSError("image unavailable")
        self.assertTrue(self.controller.confirm())
        self.assertEqual(self.status, "confirmed_empty")
        self.assertIn("下一张加载失败", self.widget.status.call_args.args[0])

    def test_qshortcut_conflict_preserves_binding(self):
        shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Alt+N"), self.widget
        )
        self.controller.configure_shortcut()
        self.assertTrue(self.controller.action.shortcut().isEmpty())
        self.assertEqual(shortcut.key().toString(), "Alt+N")

    def test_last_image_saves_without_reloading_or_closing(self):
        self.widget.file_list_widget.takeItem(1)
        self.assertTrue(self.controller.confirm())
        self.widget.open_next_image.assert_not_called()
        self.assertIn("最后一张", self.widget.status.call_args.args[0])

    def test_confirmed_empty_reuses_verified_file(self):
        self.save()
        self.assertTrue(self.controller.confirm())
        self.widget.save_file.assert_not_called()
        self.widget.open_next_image.assert_called_once()

    def test_pending_empty_still_requires_manual_save(self):
        self.save()
        self.status = "pending"
        self.assertTrue(self.controller.confirm())
        self.widget.save_file.assert_called_once()

    def test_shortcut_conflict_preserves_old_binding_and_button(self):
        existing = QtWidgets.QAction("existing", self.widget)
        existing.setShortcut("Alt+N")
        self.controller.configure_shortcut()
        self.assertTrue(self.controller.shortcut_conflict)
        self.assertEqual(existing.shortcut().toString(), "Alt+N")
        self.assertTrue(self.controller.action.shortcut().isEmpty())
        self.assertTrue(self.controller.action.isEnabled())
        self.assertIn("冲突", self.controller.action.toolTip())

    def test_text_input_blocks_shortcut_not_toolbar(self):
        edit = QtWidgets.QLineEdit(self.widget)
        with mock.patch.object(
            QtWidgets.QApplication, "focusWidget", return_value=edit
        ):
            event = QtGui.QShortcutEvent(QtGui.QKeySequence("Alt+N"), 1)
            self.assertTrue(
                self.controller.eventFilter(self.controller.action, event)
            )
        self.widget.save_file.assert_not_called()

    def test_modal_and_popup_block_shortcut_dispatch(self):
        for method in ("activeModalWidget", "activePopupWidget"):
            with self.subTest(method=method), mock.patch.object(
                QtWidgets.QApplication, "focusWidget", return_value=None
            ), mock.patch.object(
                QtWidgets.QApplication, method, return_value=self.widget
            ):
                event = QtGui.QShortcutEvent(
                    QtGui.QKeySequence("Alt+N"), 1
                )
                self.assertTrue(
                    self.controller.eventFilter(self.controller.action, event)
                )
        self.widget.save_file.assert_not_called()

    def test_other_window_shortcut_is_not_intercepted(self):
        other = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(other)
        edit = QtWidgets.QLineEdit(other)
        layout.addWidget(edit)
        action = QtWidgets.QAction("Existing Alt+N", other)
        action.setShortcut("Alt+N")
        other.addAction(action)
        activated = mock.Mock()
        action.triggered.connect(activated)
        try:
            for state in ("visible", "hidden", "unsupported"):
                with self.subTest(state=state):
                    if state == "visible":
                        self.widget.show()
                    else:
                        self.widget.hide()
                    if state == "unsupported":
                        self.widget.auto_labeling_host_context = None
                    self.controller.refresh_action()
                    other.show()
                    other.activateWindow()
                    edit.setFocus()
                    self.app.processEvents()
                    activated.reset_mock()
                    QtTest.QTest.keyClick(
                        edit, QtCore.Qt.Key_N, QtCore.Qt.AltModifier
                    )
                    self.app.processEvents()
                    activated.assert_called_once()
            self.widget.save_file.assert_not_called()
        finally:
            other.close()
            other.deleteLater()

    def test_override_only_applies_to_usable_local_action(self):
        edit = QtWidgets.QLineEdit(self.widget)
        self.widget.show()
        with mock.patch.object(
            QtWidgets.QApplication, "focusWidget", return_value=edit
        ):
            event = QtGui.QKeyEvent(
                QtCore.QEvent.ShortcutOverride,
                QtCore.Qt.Key_N, QtCore.Qt.AltModifier,
            )
            self.assertTrue(self.controller.eventFilter(edit, event))
            self.controller.action.setEnabled(False)
            self.assertFalse(self.controller.eventFilter(edit, event))
            self.controller.action.setEnabled(True)
            self.context.session_purpose = "review"
            self.assertFalse(self.controller.eventFilter(edit, event))

    def test_status_delegate_keeps_exact_navigation_path(self):
        item = self.widget.file_list_widget.item(0)
        original = item.text()
        item.setData(STATUS_ROLE, "confirmed_empty")
        option = QtWidgets.QStyleOptionViewItem()
        delegate = AnnotationStatusDelegate(self.widget)
        delegate.initStyleOption(
            option, self.widget.file_list_widget.model().index(0, 0)
        )
        self.assertIn("已确认无目标", option.text)
        self.assertEqual(item.text(), original)


if __name__ == "__main__":
    unittest.main()
