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
from anylabeling.views.labeling.widgets.auto_labeling_audit_dialog import (
    AuditInteractionGuard,
    AutoLabelingAuditDialog,
    AutoLabelingAuditPanel,
    StagedAuditUiSession,
    _StagedAuditOperationWorker,
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


class _AuditWidget(QtWidgets.QWidget):
    annotation_changed = QtCore.pyqtSignal()


def _widget(items):
    widget = _AuditWidget()
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

    def test_unreadable_first_item_keeps_queue_and_can_be_skipped(self):
        for failure in ("integrity", "load", "read"):
            with self.subTest(failure=failure):
                items = [_item("a", 0), _item("b", 1)]
                client = _AuditClient(items)
                widget = _widget(items)
                if failure == "integrity":
                    client.items["a"]["integrity_error"] = "audit_image_not_regular"
                elif failure == "load":
                    original = widget.load_file
                    widget.load_file = lambda path: (
                        False if path == items[0]["canonical_image_path"]
                        else original(path)
                    )
                else:
                    original = client.read_review_item
                    def read(image_id):
                        if image_id == "a":
                            raise OSError("unreadable item")
                        return original(image_id)
                    client.read_review_item = read
                session = StagedAuditUiSession(widget, client)
                try:
                    session.begin()
                    self.assertTrue(_wait_until(session.is_idle))
                    self.assertEqual(len(session.items), 2)
                    self.assertEqual(session.current_image_id, "a")
                    self.assertFalse(session.dialog.approve_button.isEnabled())
                    self.assertFalse(session.dialog.needs_fix_button.isEnabled())
                    self.assertFalse(session.approve_and_next())
                    self.assertFalse(session.save_approve_and_next())
                    self.assertTrue(session.dialog.defer_button.isEnabled())
                    session.dialog.defer_button.click()
                    self.assertTrue(_wait_until(session.is_idle))
                    self.assertEqual(session.current_image_id, "b")
                    self.assertTrue(session.dialog.approve_button.isEnabled())
                    self.assertEqual(client.items["a"]["review_status"], "pending")
                    self.assertEqual(session._summary["pending_review"], 2)
                finally:
                    self.assertTrue(_finish_session(session))
                    widget.close()

    def test_dialog_and_panel_offer_one_adaptive_confirmation(self):
        for cls in (AutoLabelingAuditDialog, AutoLabelingAuditPanel):
            view = cls()
            try:
                view.set_item(_item("image-a", 0), dirty=False)
                self.assertEqual(view.approve_button.text(), "Confirm and next")
                self.assertIn("No objects", view.detail_label.text())
                view.set_item(_item("image-a", 0), dirty=True)
                self.assertEqual(
                    view.approve_button.text(), "Save, confirm and next"
                )
                self.assertIn("Unsaved changes", view.detail_label.text())
                self.assertNotIn("dirty=", view.status_label.text())
                self.assertTrue(view.technical_details.isHidden())
                for name in ("previous", "next", "needs_fix", "approve",
                             "defer", "finish"):
                    callback = mock.Mock()
                    getattr(view, name + "_requested").connect(callback)
                    getattr(view, name + "_button").click()
                    callback.assert_called_once_with()
                view.set_summary({
                    "staged_approved": 2, "source_approved": 3,
                    "needs_fix": 4, "pending": 5, "stale": 6,
                })
                self.assertEqual(
                    view.counts_label.text(),
                    "15 awaiting confirmation  |  5 confirmed",
                )
                view.set_busy(True, stage="begin")
                self.assertFalse(view.activity_bar.isHidden())
                self.assertIn("Preparing", view.activity_label.text())
                self.assertFalse(view.approve_button.isEnabled())
                self.assertTrue(view.finish_button.isEnabled())
                view.set_busy(False)
                self.assertTrue(view.approve_button.isEnabled())
            finally:
                view.allow_close()
                view.close()

    def test_unknown_count_is_not_presented_as_empty(self):
        panel = AutoLabelingAuditPanel()
        try:
            item = _item("image-a", 0)
            item["target_count"] = None
            panel.set_item(item, False)
            self.assertIn("unavailable", panel.detail_label.text())
            self.assertNotIn("No objects", panel.detail_label.text())
            self.assertIsInstance(panel, QtWidgets.QWidget)
            self.assertNotIsInstance(panel, QtWidgets.QDialog)
        finally:
            panel.allow_close()
            panel.close()

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
        host.audit_sidebar_layout.addStretch(1)
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
            self.assertEqual(host.audit_sidebar_layout.stretch(0), 1)
            self.assertEqual(host.audit_sidebar_layout.stretch(3), 0)

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
            self.assertEqual(host.audit_sidebar_layout.stretch(2), 1)
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
            self.assertEqual(client.summary_calls, 1)
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
            self.assertTrue(session.confirm_and_next())
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

    def test_snapshot_is_used_once_and_navigation_updates_only_two_items(self):
        client = _AuditClient([_item("a", 0), _item("b", 1)])
        client.read_review_snapshot = mock.Mock(return_value={
            "items": list(client.items.values()), "summary": client.summary(),
        })
        client.summary_calls = 0
        worker = _StagedAuditOperationWorker(client)
        worker._execute("begin", {})
        client.items["a"]["review_status"] = "staged_approved"
        before_reads = client.read_calls
        result = worker._execute("load", {
            "image_id": "b", "previous_image_id": "a",
        })
        self.assertEqual(client.read_calls - before_reads, 2)
        self.assertEqual(result["summary"]["pending_review"], 1)
        self.assertEqual(result["summary"]["staged_approved"], 1)
        self.assertEqual(client.summary_calls, 0)
        self.assertEqual(client.list_calls, 0)
        client.read_review_snapshot.assert_called_once_with()

    def test_defer_preserves_pending_and_finishes_with_remaining_count(self):
        items = [_item("a", 0), _item("b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            session.defer_current()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "b"))
            session.defer_current()
            self.assertTrue(_wait_until(lambda: session._round_complete))
            self.assertEqual(client.calls, [])
            self.assertTrue(all(
                item["review_status"] == "pending"
                for item in client.items.values()
            ))
            self.assertIn("2 images", session.dialog.detail_label.text())
            self.assertFalse(session.dialog.approve_button.isEnabled())
            self.assertTrue(session.dialog.finish_button.isEnabled())
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_integrity_failure_is_inline_and_reload_never_approves(self):
        items = [_item("a", 0), _item("b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            client.integrity_refresh = mock.Mock(return_value=dict(
                client.items["a"], integrity_error="audit_integrity_digest_mismatch"
            ))
            session.confirm_and_next()
            self.assertTrue(_wait_until(session.is_idle))
            self.assertFalse(session.dialog.error_label.isHidden())
            self.assertEqual(session.current_image_id, "a")
            self.assertEqual(client.calls, [])
            session.dialog.reload_button.click()
            self.assertTrue(_wait_until(session.is_idle))
            self.assertEqual(client.calls, [])
            self.assertEqual(session.current_image_id, "a")
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_last_approval_keeps_finished_summary_visible(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            session.confirm_and_next()
            self.assertTrue(_wait_until(lambda: session._round_complete))
            self.assertFalse(session._closed)
            self.assertEqual(client.summary_calls, 1)
            self.assertIn("No labels", session.dialog.detail_label.text())
            self.assertFalse(session.confirm_and_next())
            self.assertEqual(
                sum(call[0] == "approve" for call in client.calls), 1
            )
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_begin_failure_can_be_reloaded_without_closing(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        original = client.list_review_items
        client.list_review_items = mock.Mock(side_effect=OSError("unavailable"))
        try:
            session.begin()
            self.assertTrue(_wait_until(session.is_idle))
            self.assertFalse(session._closed)
            self.assertFalse(session.dialog.error_label.isHidden())
            self.assertFalse(session.dialog.approve_button.isEnabled())
            client.list_review_items = original
            session.reload_current()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_busy_save_confirm_does_not_save_again(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertFalse(session.save_approve_and_next())
            widget.save_file.assert_not_called()
            self.assertTrue(_wait_until(session.is_idle))
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_conflicting_item_remains_reachable(self):
        items = [_item("a", 0), _item("b", 1, "conflict")]
        client = _AuditClient(items)
        client.read_review_snapshot = lambda: {
            "items": items,
            "summary": {"pending": 1, "stale": 1, "needs_fix": 0,
                        "staged_approved": 0, "source_approved": 0},
        }
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            session.next_pending()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "b"))
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_cancel_dirty_navigation_does_not_defer_or_save(self):
        items = [_item("a", 0), _item("b", 1)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            widget.dirty = True
            session._ask_unsaved_changes = mock.Mock(
                return_value=QtWidgets.QMessageBox.Cancel
            )
            self.assertFalse(session.defer_current())
            self.assertFalse(session.next_pending())
            self.assertFalse(session.finish())
            self.assertEqual(session._deferred_ids, set())
            self.assertEqual(session.current_image_id, "a")
            widget.save_file.assert_not_called()
            self.assertEqual(client.calls, [])
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_manual_save_before_last_defer_refreshes_finished_counts(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)

        def save():
            widget.dirty = False
            client.items["a"]["review_status"] = "staged_approved"
            return True

        widget.save_file.side_effect = save
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            widget.dirty = True
            session._ask_unsaved_changes = mock.Mock(
                return_value=QtWidgets.QMessageBox.Save
            )
            session.defer_current()
            self.assertTrue(_wait_until(lambda: session._round_complete))
            self.assertEqual(session._summary["pending_review"], 0)
            self.assertEqual(session._summary["staged_approved"], 1)
            self.assertEqual(client.calls, [])
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_dirty_action_updates_primary_text_without_storage_reads(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            reads = client.read_calls
            widget.dirty = True
            widget.actions.save.setEnabled(False)
            widget.actions.save.setEnabled(True)
            self.assertEqual(
                session.dialog.approve_button.text(), "Save, confirm and next"
            )
            self.assertEqual(client.read_calls, reads)
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()


    def test_missing_previous_image_does_not_block_next_or_defer(self):
        from tests.test_auto_labeling_audit import StandaloneStagedAuditTests
        from anylabeling.views.labeling.utils.auto_labeling_audit import (
            InMemoryStagedAuditClientV1,
        )
        from anylabeling.views.labeling.utils.continuous_auto_labeling import (
            InMemoryFastRunStoreV1,
        )

        for action in ("next_pending", "defer_current"):
            with self.subTest(action=action):
                first = StandaloneStagedAuditTests()
                second = StandaloneStagedAuditTests()
                first.setUp()
                second.setUp()
                try:
                    run_id = first.run_id
                    item_a = first.store.read_item(run_id, "image-a")
                    item_b = second.store.read_item(run_id, "image-a")
                    item_b.update(image_id="image-b", sequence=1)
                    record_b = dict(
                        second.client.records_by_id["image-a"],
                        image_id="image-b", manifest_sequence=1,
                    )
                    store = InMemoryFastRunStoreV1(
                        first.store.read_config(run_id),
                        {"run_id": run_id, "entries": [
                            {"sequence": 0, "image_id": "image-a"},
                            {"sequence": 1, "image_id": "image-b"},
                        ]},
                        first.store.read_state(run_id), [item_a, item_b],
                    )
                    client = InMemoryStagedAuditClientV1(
                        run_store=store, run_id=run_id,
                        records_by_id={
                            "image-a": first.client.records_by_id["image-a"],
                            "image-b": record_b,
                        },
                    )
                    widget = _widget(client.list_review_items())
                    session = StagedAuditUiSession(widget, client)
                    try:
                        session.begin()
                        self.assertTrue(_wait_until(
                            lambda: session.current_image_id == "image-a"
                        ))
                        first.image_path.unlink()
                        session.confirm_and_next()
                        self.assertTrue(_wait_until(session.is_idle))
                        self.assertEqual(session.current_image_id, "image-a")
                        self.assertFalse(session.dialog.error_label.isHidden())
                        self.assertTrue(getattr(session, action)())
                        self.assertTrue(_wait_until(
                            lambda: session.current_image_id == "image-b"
                        ))
                        self.assertEqual(
                            session._cached_item("image-a")["integrity_error"],
                            "audit_image_not_regular",
                        )
                        self.assertEqual(session._summary["pending_review"], 2)
                        self.assertEqual(
                            store.read_item(run_id, "image-a")["audit_decisions"],
                            [],
                        )
                        self.assertTrue(session.previous())
                        self.assertTrue(_wait_until(session.is_idle))
                        self.assertEqual(session.current_image_id, "image-a")
                        self.assertFalse(session.dialog.approve_button.isEnabled())
                        self.assertTrue(session.dialog.next_button.isEnabled())
                        self.assertFalse(session.dialog.error_label.isHidden())
                    finally:
                        self.assertTrue(_finish_session(session))
                        widget.close()
                finally:
                    first.tearDown()
                    second.tearDown()

    def test_thread_cleanup_accepts_an_already_deleted_worker(self):
        from PyQt5 import sip

        items = [_item("a", 0)]
        widget = _widget(items)
        session = StagedAuditUiSession(widget, _AuditClient(items))
        worker = _StagedAuditOperationWorker(session.client)
        session._worker = worker
        session._execute_requested.connect(worker.execute)
        try:
            sip.delete(worker)
            self.assertEqual(session.receivers(session._execute_requested), 0)
            session._on_thread_finished()
            self.assertTrue(session._closed)
            self.assertIsNone(session._worker)
            self.assertEqual(widget.receivers(widget.annotation_changed), 0)
        finally:
            session.dialog.allow_close()
            session.dialog.close()
            widget.close()

    def test_completed_round_reload_resolves_dirty_edits_and_reopens(self):
        for answer, save_ok in (
            (QtWidgets.QMessageBox.Cancel, True),
            (QtWidgets.QMessageBox.Save, False),
            (QtWidgets.QMessageBox.Save, True),
            (QtWidgets.QMessageBox.Discard, True),
        ):
            with self.subTest(answer=answer, save_ok=save_ok):
                items = [_item("a", 0)]
                client = _AuditClient(items)
                widget = _widget(items)
                session = StagedAuditUiSession(widget, client)
                try:
                    session.begin()
                    self.assertTrue(_wait_until(
                        lambda: session.current_image_id == "a"
                    ))
                    session.defer_current()
                    self.assertTrue(_wait_until(lambda: session._round_complete))
                    widget.filename = items[0]["canonical_image_path"]
                    widget.dirty = True
                    widget.save_file.return_value = False
                    session._ask_unsaved_changes = mock.Mock(
                        return_value=QtWidgets.QMessageBox.Save
                    )
                    self.assertFalse(session.finish())
                    self.assertFalse(session.dialog.reload_button.isHidden())
                    self.assertTrue(session.dialog.reload_button.isEnabled())

                    def save():
                        if save_ok:
                            widget.dirty = False
                        return save_ok

                    widget.save_file.side_effect = save
                    widget.set_clean = lambda: setattr(widget, "dirty", False)
                    session._ask_unsaved_changes.return_value = answer
                    loads = len(widget.loaded)
                    session.dialog.reload_button.click()
                    self.assertTrue(_wait_until(session.is_idle))
                    allowed = answer == QtWidgets.QMessageBox.Discard or (
                        answer == QtWidgets.QMessageBox.Save and save_ok
                    )
                    self.assertEqual(session._round_complete, not allowed)
                    self.assertEqual(widget.dirty, not allowed)
                    self.assertEqual(len(widget.loaded) > loads, allowed)
                    self.assertEqual(session.current_image_id, "a")
                    self.assertEqual(client.calls, [])
                    if allowed:
                        self.assertTrue(session.dialog.approve_button.isEnabled())
                        self.assertTrue(session.dialog.error_label.isHidden())
                finally:
                    self.assertTrue(_finish_session(session))
                    widget.close()

    def test_repeated_real_dirty_updates_counts_without_clearing_errors(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        widget._config = {"auto_save": False}
        widget.filename = items[0]["canonical_image_path"]
        widget.canvas.is_shape_restorable = False
        widget.canvas.shapes = []
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            widget.actions.save.setEnabled(False)
            reads = client.read_calls
            for count in (1, 2, 1, 0):
                widget.canvas.shapes = [object() for _ in range(count)]
                LabelingWidget.set_dirty(widget)
                expected = f"{count} objects" if count else "No objects"
                self.assertIn(expected, session.dialog.detail_label.text())
                self.assertIn("Unsaved", session.dialog.detail_label.text())
            self.assertEqual(client.read_calls, reads)
            widget.auto_labeling_commit_blocked = True
            session._warn(RuntimeError("save still blocked"))
            details = session.dialog.technical_details.toPlainText()
            LabelingWidget.set_dirty(widget)
            self.assertFalse(session.dialog.error_label.isHidden())
            self.assertFalse(session.dialog.recover_button.isHidden())
            self.assertEqual(session.dialog.technical_details.toPlainText(), details)
            widget.auto_labeling_commit_blocked = False
            widget.dirty = False
            widget.actions.save.setEnabled(False)
            self.assertTrue(session.dialog.error_label.isHidden())
            self.assertTrue(session.dialog.recover_button.isHidden())
        finally:
            widget.auto_labeling_commit_blocked = False
            self.assertTrue(_finish_session(session))
            self.assertEqual(widget.receivers(widget.annotation_changed), 0)
            widget.close()

    def test_editing_finished_round_reopens_current_item(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        widget._config = {"auto_save": False}
        widget.filename = items[0]["canonical_image_path"]
        widget.canvas.is_shape_restorable = False
        widget.canvas.shapes = []
        session = StagedAuditUiSession(widget, client)
        try:
            session.begin()
            self.assertTrue(_wait_until(lambda: session.current_image_id == "a"))
            session.confirm_and_next()
            self.assertTrue(_wait_until(lambda: session._round_complete))
            reads = client.read_calls
            widget.canvas.shapes.append(object())
            LabelingWidget.set_dirty(widget)
            self.assertFalse(session._round_complete)
            self.assertTrue(session.dialog.approve_button.isEnabled())
            self.assertEqual(session.dialog.approve_button.text(), "Save, confirm and next")
            self.assertIn("1 objects", session.dialog.detail_label.text())
            self.assertEqual(client.read_calls, reads)
            self.assertEqual(session.current_image_id, "a")
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_begin_retry_resolves_unsaved_edits_before_loading(self):
        for answer, save_ok in (
            (QtWidgets.QMessageBox.Cancel, True),
            (QtWidgets.QMessageBox.Save, False),
            (QtWidgets.QMessageBox.Save, True),
            (QtWidgets.QMessageBox.Discard, True),
        ):
            with self.subTest(answer=answer, save_ok=save_ok):
                items = [_item("a", 0)]
                client = _AuditClient(items)
                widget = _widget(items)
                session = StagedAuditUiSession(widget, client)
                original = client.list_review_items
                client.list_review_items = mock.Mock(
                    side_effect=OSError("temporarily unavailable")
                )
                try:
                    session.begin()
                    self.assertTrue(_wait_until(session.is_idle))
                    widget.dirty = True
                    widget.filename = "C:/session/original.png"
                    widget.set_clean = lambda: setattr(widget, "dirty", False)
                    observed_loads = []

                    def load(path):
                        observed_loads.append((path, widget.dirty))
                        return True

                    def save():
                        if save_ok:
                            widget.set_clean()
                        return save_ok

                    widget.load_file = load
                    widget.save_file.side_effect = save
                    session._ask_unsaved_changes = mock.Mock(
                        return_value=answer
                    )
                    client.list_review_items = original
                    permitted = (
                        answer == QtWidgets.QMessageBox.Discard
                        or answer == QtWidgets.QMessageBox.Save and save_ok
                    )
                    self.assertEqual(session.reload_current(), permitted)
                    self.assertTrue(_wait_until(session.is_idle))
                    session._ask_unsaved_changes.assert_called_once_with()
                    if permitted:
                        self.assertEqual(session.current_image_id, "a")
                        self.assertTrue(observed_loads)
                        self.assertTrue(all(
                            not dirty for _, dirty in observed_loads
                        ))
                    else:
                        self.assertIsNone(session.current_image_id)
                        self.assertEqual(observed_loads, [])
                        self.assertTrue(widget.dirty)
                    self.assertEqual(client.calls, [])
                finally:
                    self.assertTrue(_finish_session(session))
                    widget.close()

    def test_retry_during_busy_or_shutdown_does_not_prompt_or_save(self):
        client = _AuditClient([_item("a", 0)])
        widget = _widget(list(client.items.values()))
        session = StagedAuditUiSession(widget, client)
        session._ask_unsaved_changes = mock.Mock()
        try:
            session.begin()
            widget.dirty = True
            self.assertFalse(session.reload_current())
            self.assertFalse(session.recover_save())
            session._ask_unsaved_changes.assert_not_called()
            widget.save_file.assert_not_called()
            self.assertTrue(_wait_until(session.is_idle))
            self.assertTrue(_finish_session(session))
            self.assertFalse(session.reload_current())
            self.assertFalse(session.recover_save())
        finally:
            self.assertTrue(_finish_session(session))
            widget.close()

    def test_blocked_save_has_explicit_retry_without_approval_or_navigation(self):
        for outcome in ("success", "failure", "still_blocked", "still_dirty"):
            with self.subTest(outcome=outcome):
                items = [_item("a", 0), _item("b", 1)]
                client = _AuditClient(items)
                widget = _widget(items)
                session = StagedAuditUiSession(widget, client)
                try:
                    session.begin()
                    self.assertTrue(_wait_until(
                        lambda: session.current_image_id == "a"
                    ))
                    widget.dirty = True
                    widget.auto_labeling_commit_blocked = True
                    self.assertFalse(session.reload_current())
                    self.assertTrue(session.dialog.reload_button.isHidden())
                    self.assertFalse(
                        session.dialog.recover_button.isHidden()
                    )

                    def save():
                        if outcome != "failure":
                            widget.auto_labeling_commit_blocked = (
                                outcome == "still_blocked"
                            )
                            widget.dirty = outcome == "still_dirty"
                        return outcome != "failure"

                    widget.save_file.side_effect = save
                    loaded = list(widget.loaded)
                    reads = client.read_calls
                    session.dialog.recover_button.click()
                    self.assertTrue(_wait_until(session.is_idle))
                    widget.save_file.assert_called_once_with()
                    self.assertEqual(client.calls, [])
                    self.assertEqual(widget.loaded, loaded)
                    self.assertEqual(session.current_image_id, "a")
                    self.assertEqual(
                        client.read_calls - reads,
                        1 if outcome == "success" else 0,
                    )
                    self.assertEqual(
                        session.dialog.error_label.isHidden(),
                        outcome == "success",
                    )
                finally:
                    widget.auto_labeling_commit_blocked = False
                    self.assertTrue(_finish_session(session))
                    widget.close()

    def test_initial_retry_keeps_commit_blocker_until_explicit_save(self):
        items = [_item("a", 0)]
        client = _AuditClient(items)
        widget = _widget(items)
        session = StagedAuditUiSession(widget, client)
        original = client.list_review_items
        client.list_review_items = mock.Mock(side_effect=OSError("busy"))
        try:
            session.begin()
            self.assertTrue(_wait_until(session.is_idle))
            widget.auto_labeling_commit_blocked = True
            widget.dirty = True
            client.list_review_items = original
            self.assertFalse(session.reload_current())
            self.assertEqual(widget.loaded, [])
            widget.save_file.assert_not_called()

            def save():
                widget.auto_labeling_commit_blocked = False
                widget.dirty = False
                return True

            widget.save_file.side_effect = save
            self.assertTrue(session.recover_save())
            self.assertIsNone(session.current_image_id)
            self.assertEqual(widget.loaded, [])
            self.assertFalse(session.dialog.reload_button.isHidden())
            self.assertTrue(session.reload_current())
            self.assertTrue(_wait_until(
                lambda: session.current_image_id == "a"
            ))
            self.assertEqual(client.calls, [])
        finally:
            widget.auto_labeling_commit_blocked = False
            self.assertTrue(_finish_session(session))
            widget.close()


    def test_retry_button_uses_real_save_recovery_and_rejects_disk_changes(self):
        from tests.test_auto_labeling_commit_bridge import (
            AnnotationCommitBridgeTests, _document, _write,
        )
        from anylabeling.views.labeling.utils.auto_labeling_commit_bridge import (
            AnnotationCommitBridgeV1,
        )

        for external_change in (False, True):
            with self.subTest(external_change=external_change):
                fixture = AnnotationCommitBridgeTests()
                fixture.setUp()
                item = _item("image-a", 0)
                item["canonical_image_path"] = str(fixture.image_path)
                widget = _widget([item])
                client = _AuditClient([item])
                controls = {
                    name: getattr(widget, name) for name in (
                        "actions", "canvas", "file_list_widget",
                    )
                }
                for name, value in vars(fixture.widget).items():
                    setattr(widget, name, value)
                fixture.widget = widget
                fixture.bridge = AnnotationCommitBridgeV1(widget)
                fixture.bridge.record_loaded_label(fixture.label_path)
                fixture._configure_save_surface()
                for name, value in controls.items():
                    setattr(widget, name, value)
                widget.save_file = lambda: LabelingWidget.save_file(widget)
                fixture.sink.publish.side_effect = [
                    RuntimeError("response lost"), "IDEMPOTENT_NO_OP",
                    "APPLIED",
                ]
                session = StagedAuditUiSession(widget, client)
                try:
                    session.begin()
                    self.assertTrue(_wait_until(
                        lambda: session.current_image_id == "image-a"
                    ))
                    widget.dirty = True
                    self.assertFalse(session.confirm_and_next())
                    self.assertTrue(widget.auto_labeling_commit_blocked)
                    pending = fixture.bridge.pending_commit
                    if external_change:
                        _write(fixture.label_path, _document("external"))
                    before_bytes = fixture.label_path.read_bytes()
                    loaded = list(widget.loaded)
                    session.dialog.recover_button.click()
                    self.assertTrue(_wait_until(session.is_idle))
                    self.assertEqual(client.calls, [])
                    self.assertEqual(widget.loaded, loaded)
                    self.assertEqual(
                        fixture.label_path.read_bytes(), before_bytes,
                    )
                    if external_change:
                        self.assertTrue(widget.auto_labeling_commit_blocked)
                        self.assertTrue(widget.dirty)
                        self.assertEqual(
                            fixture.bridge.pending_commit, pending,
                        )
                        self.assertEqual(fixture.sink.publish.call_count, 1)
                        self.assertFalse(
                            session.dialog.recover_button.isHidden()
                        )
                    else:
                        self.assertFalse(widget.auto_labeling_commit_blocked)
                        self.assertFalse(widget.dirty)
                        self.assertIsNone(fixture.bridge.pending_commit)
                        calls = fixture.sink.publish.call_args_list
                        self.assertEqual(len(calls), 3)
                        self.assertEqual(calls[0].args[0], calls[1].args[0])
                finally:
                    widget.auto_labeling_commit_blocked = False
                    self.assertTrue(_finish_session(session))
                    widget.close()
                    fixture.tearDown()


class ReviewDisplayedContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = _app()

    def setUp(self):
        from tests.test_auto_labeling_audit import StandaloneStagedAuditTests
        from anylabeling.views.labeling.label_file import LabelFile

        self.fixture = StandaloneStagedAuditTests()
        self.fixture.setUp()
        self.widget = _widget(self.fixture.client.list_review_items())

        def load(path):
            self.widget.loaded.append(path)
            label = LabelFile(str(self.fixture.label_path), capture_audit=True)
            self.widget.label_file = label
            self.widget.image_data = label.image_data
            return True

        self.widget.load_file = load
        self.session = StagedAuditUiSession(self.widget, self.fixture.client)
        self.session.begin()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertTrue(self.session._item_presented)

    def tearDown(self):
        self.assertTrue(_finish_session(self.session))
        self.widget.close()
        self.fixture.tearDown()

    def _decisions(self):
        return self.fixture.store.read_item(
            self.fixture.run_id, "image-a"
        )["audit_decisions"]

    def test_changed_labels_are_not_approved_until_loaded(self):
        from tests.test_auto_labeling_audit import _document, _write

        _write(self.fixture.label_path, _document(description="external edit"))
        for _ in range(2):
            self.session.dialog.approve_button.click()
            self.assertTrue(_wait_until(self.session.is_idle))
            self.assertEqual(self._decisions(), [])
        self.assertEqual(len(self.widget.loaded), 1)
        self.assertFalse(self.session.dialog.error_label.isHidden())
        self.session.reload_current()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(len(self._decisions()), 1)

    def test_needs_fix_is_also_bound_to_displayed_content(self):
        from tests.test_auto_labeling_audit import _document, _write

        _write(self.fixture.label_path, _document(description="external edit"))
        self.session.dialog.needs_fix_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(self._decisions(), [])

    def test_changed_image_is_not_approved_until_loaded(self):
        from PIL import Image

        Image.new("RGB", (8, 6), (200, 100, 50)).save(self.fixture.image_path)
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(self._decisions(), [])
        self.session.reload_current()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(len(self._decisions()), 1)

    def test_formatting_changes_do_not_block_confirmation(self):
        import json

        self.fixture.label_path.write_text(
            json.dumps(self.fixture.document, indent=4, sort_keys=True),
            encoding="utf-8",
        )
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(len(self._decisions()), 1)
        self.assertEqual(len(self.widget.loaded), 1)

    def _save_edits(self):
        from anylabeling.views.labeling.label_file import LabelFile

        label = LabelFile()
        label.save(
            str(self.fixture.label_path), shapes=[], image_path="image.png",
            image_height=6, image_width=8,
            other_data={"description": "my edit"},
        )
        self.widget.label_file = label
        self.widget.dirty = False
        return True

    def test_save_and_confirm_uses_saved_content_without_extra_click(self):
        self.widget.dirty = True
        self.widget.save_file = self._save_edits
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(len(self._decisions()), 1)
        self.assertEqual(len(self.widget.loaded), 1)

    def test_regular_save_then_confirm_uses_saved_content(self):
        self._save_edits()
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(len(self._decisions()), 1)

    def test_missing_initial_image_digest_does_not_force_reload(self):
        item = self.fixture.store.read_item(self.fixture.run_id, "image-a")
        digests = dict(item["digests"], source_image_digest=None)
        self.fixture.store.update_item(
            self.fixture.run_id, "image-a", item["item_revision"],
            {"digests": digests},
        )
        self.session.dialog.approve_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(len(self._decisions()), 1)


if __name__ == "__main__":
    unittest.main()
