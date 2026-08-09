import copy
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_boolean_text_v1,
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.widgets.auto_labeling_audit_dialog import (
    AuditInteractionGuard,
    AutoLabelingAuditDialog,
    StagedAuditUiSession,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _item(image_id, sequence, status="pending"):
    return {
        "image_id": image_id,
        "sequence": sequence,
        "item_revision": sequence,
        "canonical_image_path": f"C:/session/{image_id}.png",
        "canonical_label_path": f"C:/session/{image_id}.json",
        "filename": f"{image_id}.png",
        "review_status": status,
        "staged_commit_status": "committed",
        "source_commit_status": "pending",
        "target_count": sequence,
        "zero_target": sequence == 0,
        "integrity_error": None,
    }


class _AuditClient:
    def __init__(self, items):
        self.items = {item["image_id"]: copy.deepcopy(item) for item in items}
        self.calls = []

    def list_review_items(self):
        reviewable = {"pending", "needs_fix", "stale"}
        return [
            copy.deepcopy(item)
            for item in self.items.values()
            if item["review_status"] in reviewable
        ]

    def read_review_item(self, image_id):
        return copy.deepcopy(self.items[image_id])

    def summary(self):
        counts = {
            "staged_approved": 0,
            "source_approved": 0,
            "needs_fix": 0,
            "pending": 0,
            "stale": 0,
        }
        for item in self.items.values():
            status = item["review_status"]
            if status == "staged_approved":
                counts["staged_approved"] += 1
            elif status == "approved":
                counts["source_approved"] += 1
            elif status in counts:
                counts[status] += 1
        return counts

    def integrity_refresh(self, image_id):
        self.calls.append(("refresh", image_id))
        return self.read_review_item(image_id)

    def approve(self, image_id, expected_revision):
        self.calls.append(("approve", image_id, expected_revision))
        item = self.items[image_id]
        if item["item_revision"] != expected_revision:
            raise RuntimeError("revision mismatch")
        item["item_revision"] += 1
        item["review_status"] = "staged_approved"
        return copy.deepcopy(item)

    def needs_fix(self, image_id, expected_revision):
        self.calls.append(("needs_fix", image_id, expected_revision))
        item = self.items[image_id]
        if item["item_revision"] != expected_revision:
            raise RuntimeError("revision mismatch")
        item["item_revision"] += 1
        item["review_status"] = "needs_fix"
        return copy.deepcopy(item)


def _widget(items):
    widget = QtWidgets.QWidget()
    action_names = AuditInteractionGuard._ACTION_NAMES
    actions = SimpleNamespace()
    for name in action_names:
        action = QtWidgets.QAction(widget)
        action.setEnabled(True)
        setattr(actions, name, action)
    widget.actions = actions
    widget.file_list_widget = QtWidgets.QListWidget(widget)
    widget.auto_labeling_widget = QtWidgets.QWidget(widget)
    widget.fn_to_index = {}
    for row, item in enumerate(
        sorted(items, key=lambda value: value["sequence"])
    ):
        widget.file_list_widget.addItem(item["canonical_image_path"])
        widget.fn_to_index[item["canonical_image_path"]] = row
    widget.dirty = False
    widget.auto_labeling_commit_blocked = False
    widget.auto_labeling_audit_active = False
    widget._auto_labeling_audit_session = None
    widget.loaded = []

    def load_file(path):
        widget.loaded.append(path)
        return True

    widget.load_file = load_file
    widget.save_file = mock.Mock(return_value=True)
    return widget


class AutoLabelingAuditDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def test_dialog_exposes_six_actions_and_summary_fields(self):
        dialog = AutoLabelingAuditDialog()
        callbacks = {
            "previous_requested": mock.Mock(),
            "next_requested": mock.Mock(),
            "needs_fix_requested": mock.Mock(),
            "save_approve_requested": mock.Mock(),
            "approve_requested": mock.Mock(),
            "finish_requested": mock.Mock(),
        }
        try:
            for signal_name, callback in callbacks.items():
                getattr(dialog, signal_name).connect(callback)
            for button in (
                dialog.previous_button,
                dialog.next_button,
                dialog.needs_fix_button,
                dialog.save_approve_button,
                dialog.approve_button,
                dialog.finish_button,
            ):
                button.click()
            for callback in callbacks.values():
                callback.assert_called_once_with()

            dialog.set_item(_item("image-a", 0), dirty=True)
            dialog.set_summary(
                {
                    "staged_approved": 2,
                    "source_approved": 3,
                    "needs_fix": 4,
                    "pending": 5,
                    "stale": 6,
                }
            )
            self.assertIn("target_count=0", dialog.detail_label.text())
            self.assertIn(
                f"zero_target={auto_labeling_boolean_text_v1(True)}",
                dialog.detail_label.text(),
            )
            self.assertIn(
                f"dirty={auto_labeling_boolean_text_v1(True)}",
                dialog.status_label.text(),
            )
            self.assertEqual(
                dialog.counts_label.text(),
                auto_labeling_text_v1(
                    "audit_summary",
                    staged_approved=2,
                    source_approved=3,
                    needs_fix=4,
                    pending=5,
                    stale=6,
                ),
            )
            for count in ("2", "3", "4", "5", "6"):
                self.assertIn(count, dialog.counts_label.text())
        finally:
            dialog.allow_close()
            dialog.close()

    def test_guard_disables_identity_actions_and_restores_original_state(self):
        items = [_item("image-a", 0)]
        widget = _widget(items)
        widget.actions.save_as.setEnabled(False)
        guard = AuditInteractionGuard(widget)
        try:
            guard.activate()
            self.assertTrue(widget.auto_labeling_audit_active)
            for name in AuditInteractionGuard._ACTION_NAMES:
                self.assertFalse(getattr(widget.actions, name).isEnabled())
            self.assertFalse(widget.file_list_widget.isEnabled())
            self.assertFalse(widget.auto_labeling_widget.isEnabled())

            guard.release()
            self.assertFalse(widget.auto_labeling_audit_active)
            self.assertFalse(widget.actions.save_as.isEnabled())
            self.assertTrue(widget.actions.open_next_image.isEnabled())
            self.assertTrue(widget.file_list_widget.isEnabled())
            self.assertTrue(widget.auto_labeling_widget.isEnabled())
        finally:
            widget.close()

    def test_sequence_navigation_uses_stable_image_ids(self):
        items = [
            _item("image-c", 30),
            _item("image-a", 10),
            _item("image-b", 20),
        ]
        widget = _widget(items)
        session = StagedAuditUiSession(widget, _AuditClient(items))
        try:
            self.assertTrue(session.begin())
            self.assertEqual(session.current_image_id, "image-a")
            session.next_pending()
            self.assertEqual(session.current_image_id, "image-b")
            session.previous()
            self.assertEqual(session.current_image_id, "image-a")
            self.assertEqual(
                widget.loaded,
                [
                    "C:/session/image-a.png",
                    "C:/session/image-b.png",
                    "C:/session/image-a.png",
                ],
            )
        finally:
            widget.dirty = False
            session.finish()
            widget.close()

    def test_dirty_approve_is_rejected_without_decision_or_navigation(self):
        items = [_item("image-a", 0), _item("image-b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        session._warn = mock.Mock()
        try:
            self.assertTrue(session.begin())
            widget.dirty = True
            session.approve_and_next()
            self.assertEqual(session.current_image_id, "image-a")
            self.assertNotIn("approve", [call[0] for call in client.calls])
            session._warn.assert_called_once()
            self.assertIn(
                "audit_dirty_requires_save",
                str(session._warn.call_args.args[0]),
            )
        finally:
            widget.dirty = False
            session.finish()
            widget.close()

    def test_save_approve_orders_save_refresh_decision_and_navigation(self):
        items = [_item("image-a", 0), _item("image-b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        actions = []

        def save():
            actions.append("save")
            widget.dirty = False
            return True

        widget.save_file.side_effect = save
        original_refresh = client.integrity_refresh
        original_approve = client.approve

        def refresh(image_id):
            actions.append("refresh")
            return original_refresh(image_id)

        def approve(image_id, revision):
            actions.append("approve")
            return original_approve(image_id, revision)

        client.integrity_refresh = refresh
        client.approve = approve
        session = StagedAuditUiSession(widget, client)
        try:
            self.assertTrue(session.begin())
            widget.dirty = True
            session.save_approve_and_next()
            self.assertEqual(actions, ["save", "refresh", "approve"])
            self.assertEqual(
                client.items["image-a"]["review_status"], "staged_approved"
            )
            self.assertEqual(session.current_image_id, "image-b")
        finally:
            widget.dirty = False
            session.finish()
            widget.close()

    def test_save_approve_failure_does_not_approve_or_navigate(self):
        items = [_item("image-a", 0), _item("image-b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        widget.save_file.return_value = False
        session = StagedAuditUiSession(widget, client)
        session._warn = mock.Mock()
        try:
            self.assertTrue(session.begin())
            widget.dirty = True
            session.save_approve_and_next()
            self.assertEqual(session.current_image_id, "image-a")
            self.assertEqual(
                client.items["image-a"]["review_status"], "pending"
            )
            self.assertNotIn("approve", [call[0] for call in client.calls])
            session._warn.assert_called_once()
        finally:
            widget.dirty = False
            session.finish()
            widget.close()

    def test_review_entry_only_constructs_audit_session(self):
        client = _AuditClient([_item("image-a", 0)])
        widget = SimpleNamespace(
            _auto_labeling_audit_session=None,
            auto_labeling_host_context=None,
            _standalone_auto_labeling_audit_client=None,
        )
        audit_session = SimpleNamespace(begin=mock.Mock(return_value=True))
        with mock.patch(
            "anylabeling.views.labeling.widgets.auto_labeling_audit_dialog."
            "StagedAuditUiSession",
            return_value=audit_session,
        ) as session_class:
            self.assertTrue(
                LabelingWidget.start_auto_labeling_review(widget, client)
            )
        session_class.assert_called_once_with(widget, client)
        audit_session.begin.assert_called_once_with()
        self.assertIs(widget._auto_labeling_audit_session, audit_session)


if __name__ == "__main__":
    unittest.main()
