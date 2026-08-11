import copy
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_boolean_text_v1,
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.widgets.auto_labeling_audit_dialog import (
    AuditInteractionGuard,
    AutoLabelingAuditDialog,
    AutoLabelingAuditPanel,
    StagedAuditUiSession,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_until(predicate, timeout_ms=3000):
    app = _app()
    deadline = time.monotonic() + (timeout_ms / 1000)
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    return predicate()


def _finish_session(session):
    session.widget.dirty = False
    session.finish()
    return _wait_until(lambda: session.worker_thread is None)


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
        self.list_calls = 0
        self.read_calls = 0
        self.summary_calls = 0
        self.call_thread_ids = []

    def _record_thread(self):
        self.call_thread_ids.append(int(QtCore.QThread.currentThreadId()))

    def list_review_items(self):
        self._record_thread()
        self.list_calls += 1
        reviewable = {"pending", "needs_fix", "stale"}
        return [
            copy.deepcopy(item)
            for item in self.items.values()
            if item["review_status"] in reviewable
        ]

    def read_review_item(self, image_id):
        self._record_thread()
        self.read_calls += 1
        return copy.deepcopy(self.items[image_id])

    def summary(self):
        self._record_thread()
        self.summary_calls += 1
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
        self._record_thread()
        self.calls.append(("refresh", image_id))
        return self.read_review_item(image_id)

    def approve(self, image_id, expected_revision):
        self._record_thread()
        self.calls.append(("approve", image_id, expected_revision))
        item = self.items[image_id]
        if item["item_revision"] != expected_revision:
            raise RuntimeError("revision mismatch")
        item["item_revision"] += 1
        item["review_status"] = "staged_approved"
        return copy.deepcopy(item)

    def needs_fix(self, image_id, expected_revision):
        self._record_thread()
        self.calls.append(("needs_fix", image_id, expected_revision))
        item = self.items[image_id]
        if item["item_revision"] != expected_revision:
            raise RuntimeError("revision mismatch")
        item["item_revision"] += 1
        item["review_status"] = "needs_fix"
        return copy.deepcopy(item)


def _widget(items):
    widget = QtWidgets.QWidget()
    action_names = (
        AuditInteractionGuard._ACTION_NAMES
        + AuditInteractionGuard._TRANSACTION_ACTION_NAMES
    )
    actions = SimpleNamespace()
    for name in dict.fromkeys(action_names):
        action = QtWidgets.QAction(widget)
        action.setEnabled(True)
        setattr(actions, name, action)
    widget.actions = actions
    widget.file_list_widget = QtWidgets.QListWidget(widget)
    widget.auto_labeling_widget = QtWidgets.QWidget(widget)
    widget.canvas = QtWidgets.QWidget(widget)
    widget.label_list = QtWidgets.QWidget(widget)
    widget.unique_label_list = QtWidgets.QWidget(widget)
    widget.label_filter_combobox = QtWidgets.QWidget(widget)
    widget.shape_dock = QtWidgets.QWidget(widget)
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
            dialog.set_busy(True)
            self.assertFalse(dialog.activity_bar.isHidden())
            self.assertFalse(dialog.approve_button.isEnabled())
            self.assertTrue(dialog.finish_button.isEnabled())
            dialog.set_busy(False)
            self.assertFalse(dialog.activity_bar.isVisible())
            self.assertTrue(dialog.approve_button.isEnabled())
        finally:
            dialog.allow_close()
            dialog.close()

    def test_embedded_panel_exposes_the_same_actions_without_modal_ui(self):
        parent = QtWidgets.QWidget()
        panel = AutoLabelingAuditPanel(parent)
        callbacks = {
            "previous_requested": mock.Mock(),
            "next_requested": mock.Mock(),
            "needs_fix_requested": mock.Mock(),
            "save_approve_requested": mock.Mock(),
            "approve_requested": mock.Mock(),
            "finish_requested": mock.Mock(),
        }
        try:
            self.assertIsInstance(panel, QtWidgets.QWidget)
            self.assertNotIsInstance(panel, QtWidgets.QDialog)
            for signal_name, callback in callbacks.items():
                getattr(panel, signal_name).connect(callback)
            for button in (
                panel.previous_button,
                panel.next_button,
                panel.needs_fix_button,
                panel.save_approve_button,
                panel.approve_button,
                panel.finish_button,
            ):
                button.click()
            for callback in callbacks.values():
                callback.assert_called_once_with()
            panel.set_item(_item("image-a", 0), dirty=False)
            panel.set_summary(
                {
                    "staged_approved": 0,
                    "source_approved": 0,
                    "needs_fix": 0,
                    "pending": 1,
                    "stale": 0,
                }
            )
            self.assertIn("image-a", panel.identity_label.text())
            self.assertIn("1", panel.counts_label.text())
        finally:
            panel.allow_close()
            panel.close()
            parent.close()

    def test_sidebar_panel_lifecycle_and_pending_counts_share_one_entry(self):
        host = QtWidgets.QWidget()
        host.right_sidebar_tabs = QtWidgets.QTabWidget(host)
        host.labels_page = QtWidgets.QWidget(host.right_sidebar_tabs)
        host.audit_sidebar_page = QtWidgets.QWidget(host.right_sidebar_tabs)
        host.right_sidebar_tabs.addTab(host.labels_page, "Labels")
        host.right_sidebar_tabs.addTab(host.audit_sidebar_page, "Audit")
        host.audit_sidebar_layout = QtWidgets.QVBoxLayout(
            host.audit_sidebar_page
        )
        host.audit_empty_label = QtWidgets.QLabel(host.audit_sidebar_page)
        host.audit_open_button = QtWidgets.QPushButton(host.audit_sidebar_page)
        host.audit_sidebar_layout.addWidget(host.audit_empty_label)
        host.audit_sidebar_layout.addWidget(host.audit_open_button)
        host.auto_labeling_widget = SimpleNamespace(
            set_pending_review_count=mock.Mock()
        )
        host._session_pending_review_count = 0
        host._historical_pending_review_count = 0
        host._refresh_pending_review_controls = lambda: (
            LabelingWidget._refresh_pending_review_controls(host)
        )
        panel = AutoLabelingAuditPanel(host.audit_sidebar_page)
        try:
            LabelingWidget.show_auto_labeling_audit_panel(host, panel)
            self.assertIs(
                host.right_sidebar_tabs.currentWidget(),
                host.audit_sidebar_page,
            )
            self.assertTrue(host.audit_empty_label.isHidden())
            self.assertTrue(host.audit_open_button.isHidden())

            LabelingWidget.set_auto_labeling_pending_review_count(
                host, 2, scope="session"
            )
            LabelingWidget.set_auto_labeling_pending_review_count(
                host, 3, scope="historical"
            )
            self.assertIn("5", host.audit_open_button.text())
            host.auto_labeling_widget.set_pending_review_count.assert_called_with(
                5
            )

            LabelingWidget.remove_auto_labeling_audit_panel(host, panel)
            self.assertFalse(host.audit_empty_label.isHidden())
            self.assertFalse(host.audit_open_button.isHidden())
        finally:
            panel.allow_close()
            panel.close()
            host.close()

    def test_review_entry_prefers_current_session_then_historical_callback(
        self,
    ):
        current_client = object()
        callback = mock.Mock(return_value=True)
        host = SimpleNamespace(
            _auto_labeling_audit_session=None,
            right_sidebar_tabs=mock.Mock(),
            audit_sidebar_page=object(),
            auto_labeling_host_context=SimpleNamespace(
                audit_client=current_client,
                request_historical_review=callback,
            ),
            _session_pending_review_count=1,
            _historical_pending_review_count=4,
            start_auto_labeling_review=mock.Mock(return_value=True),
        )
        self.assertTrue(LabelingWidget.open_auto_labeling_review(host))
        host.start_auto_labeling_review.assert_called_once_with(current_client)
        callback.assert_not_called()

        host._session_pending_review_count = 0
        host.start_auto_labeling_review.reset_mock()
        self.assertTrue(LabelingWidget.open_auto_labeling_review(host))
        callback.assert_called_once_with()
        host.start_auto_labeling_review.assert_not_called()

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

            guard.set_transaction_busy(True)
            self.assertFalse(widget.canvas.isEnabled())
            self.assertFalse(widget.actions.save.isEnabled())
            guard.set_transaction_busy(False)
            self.assertTrue(widget.canvas.isEnabled())
            self.assertTrue(widget.actions.save.isEnabled())

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
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            self.assertTrue(session.begin())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            self.assertEqual(session.current_image_id, "image-a")
            self.assertEqual(client.list_calls, 1)
            self.assertTrue(session.next_pending())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-b")
            )
            self.assertEqual(session.current_image_id, "image-b")
            self.assertEqual(client.list_calls, 1)
            self.assertTrue(session.previous())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
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
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_idle_audit_session_does_not_poll_persistent_client(self):
        items = [_item("image-a", 0), _item("image-b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            self.assertTrue(session.begin())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            baseline = (
                client.list_calls,
                client.read_calls,
                client.summary_calls,
            )

            QtTest.QTest.qWait(350)

            self.assertEqual(
                (
                    client.list_calls,
                    client.read_calls,
                    client.summary_calls,
                ),
                baseline,
            )
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_dirty_approve_is_rejected_without_decision_or_navigation(self):
        items = [_item("image-a", 0), _item("image-b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        session._warn = mock.Mock()
        try:
            self.assertTrue(session.begin())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
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
            self.assertTrue(_finish_session(session))
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
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            widget.dirty = True
            self.assertTrue(session.save_approve_and_next())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-b")
            )
            self.assertEqual(actions, ["save", "refresh", "approve"])
            self.assertEqual(
                client.items["image-a"]["review_status"], "staged_approved"
            )
            self.assertEqual(session.current_image_id, "image-b")
        finally:
            self.assertTrue(_finish_session(session))
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
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            widget.dirty = True
            session.save_approve_and_next()
            self.assertEqual(session.current_image_id, "image-a")
            self.assertEqual(
                client.items["image-a"]["review_status"], "pending"
            )
            self.assertNotIn("approve", [call[0] for call in client.calls])
            session._warn.assert_called_once()
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_persistent_client_calls_run_only_on_worker_thread(self):
        items = [_item("image-a", 0), _item("image-b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        gui_thread_id = int(QtCore.QThread.currentThreadId())
        try:
            self.assertTrue(session.begin())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            self.assertTrue(session.approve_and_next())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-b")
            )
            self.assertTrue(client.call_thread_ids)
            self.assertNotIn(gui_thread_id, client.call_thread_ids)
            self.assertEqual(len(set(client.call_thread_ids)), 1)
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_blocked_write_keeps_gui_responsive_and_safe_close_waits(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingClient(_AuditClient):
            def approve(self, image_id, expected_revision):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test write timeout")
                return super().approve(image_id, expected_revision)

        items = [_item("image-a", 0), _item("image-b", 1)]
        client = BlockingClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        heartbeats = []
        safe_generations = []
        timer = QtCore.QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: heartbeats.append(time.monotonic()))
        session.safe_to_close.connect(safe_generations.append)
        try:
            self.assertTrue(session.begin())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            timer.start()
            self.assertTrue(session.approve_and_next())
            self.assertTrue(_wait_until(entered.is_set))
            self.assertFalse(session.is_idle())
            self.assertFalse(session.dialog.approve_button.isEnabled())
            self.assertFalse(widget.canvas.isEnabled())
            self.assertFalse(session.approve_and_next())

            QtTest.QTest.qWait(120)

            self.assertGreaterEqual(len(heartbeats), 3)
            generation = session.request_safe_shutdown("window_close")
            self.assertEqual(safe_generations, [])
            self.assertIsNotNone(session.worker_thread)

            release.set()
            self.assertTrue(
                _wait_until(lambda: safe_generations == [generation])
            )
            self.assertIsNone(session.worker_thread)
            self.assertEqual(
                [call[0] for call in client.calls].count("approve"),
                1,
            )
        finally:
            timer.stop()
            release.set()
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_worker_failure_restores_controls_without_navigation(self):
        class FailingClient(_AuditClient):
            def approve(self, image_id, expected_revision):
                self._record_thread()
                raise RuntimeError("disk unavailable")

        items = [_item("image-a", 0), _item("image-b", 1)]
        client = FailingClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        session._warn = mock.Mock()
        try:
            self.assertTrue(session.begin())
            self.assertTrue(
                _wait_until(lambda: session.current_image_id == "image-a")
            )
            self.assertTrue(session.approve_and_next())
            self.assertTrue(_wait_until(lambda: session._warn.called))
            self.assertTrue(session.is_idle())
            self.assertEqual(session.current_image_id, "image-a")
            self.assertTrue(session.dialog.approve_button.isEnabled())
            self.assertTrue(widget.canvas.isEnabled())
        finally:
            self.assertTrue(_finish_session(session))
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
