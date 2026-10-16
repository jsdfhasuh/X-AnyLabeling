import copy
import threading
import unittest
from unittest import mock

from PyQt5 import QtCore, QtWidgets

from tests import test_auto_labeling_audit as staged_fixtures
from tests.test_auto_labeling_audit_dialog import (
    _app, _AuditClient, _finish_session, _item, _wait_until, _widget,
)
from anylabeling.views.labeling.widgets.auto_labeling_audit_dialog import (
    BatchConfirmationDialog, StagedAuditUiSession,
    _StagedAuditOperationWorker,
)


def _pending(image_id, sequence, status="pending"):
    return dict(
        _item(image_id, sequence, status), audit_scope="STAGED",
        review_semantic_digest="semantic-" + image_id,
        review_image_digest="image-" + image_id,
    )


class BatchWorkerTests(unittest.TestCase):
    def setUp(self):
        self.items = [_pending("a", 0), _pending("b", 1), _pending("c", 2)]
        self.client = _AuditClient(self.items)
        self.worker = _StagedAuditOperationWorker(self.client)
        self.snapshot = self.worker._execute("batch_prepare", {})
        self.cancel = threading.Event()

    def run_batch(self, items=None):
        return self.worker._execute("batch_approve", {
            "items": self.snapshot["items"] if items is None else items,
            "cancel": self.cancel,
        })

    def test_all_confirmed_with_one_snapshot_and_no_duplicate_retry(self):
        result = self.run_batch()
        self.assertEqual(result["approved"], 3)
        self.assertEqual(result["summary"]["pending_review"], 0)
        self.assertEqual(self.client.list_calls, 1)
        self.assertEqual(self.client.summary_calls, 1)
        self.assertEqual(self.run_batch()["approved"], 0)
        self.assertEqual(len([c for c in self.client.calls if c[0] == "approve"]), 3)

    def test_changed_revision_status_scope_or_fingerprint_skips_only_bad_item(self):
        for field, value in (
            ("item_revision", 99), ("review_status", "needs_fix"),
            ("audit_scope", "SOURCE"), ("review_semantic_digest", "changed"),
            ("review_image_digest", "changed"), ("integrity_error", "missing"),
        ):
            with self.subTest(field=field):
                self.setUp()
                self.client.items["b"][field] = value
                result = self.run_batch()
                self.assertEqual(result["approved"], 2)
                self.assertEqual(result["skipped"][0]["image_id"], "b")
                self.assertFalse(any(c[:2] == ("approve", "b") for c in self.client.calls))

    def test_scope_changed_during_refresh_is_not_approved(self):
        original = self.client.integrity_refresh

        def refresh(image_id):
            if image_id == "b":
                self.client.items[image_id]["audit_scope"] = "SOURCE"
            return original(image_id)

        self.client.integrity_refresh = refresh
        self.assertEqual(self.run_batch()["approved"], 2)

    def test_displayed_current_digest_overrides_snapshot(self):
        self.snapshot["items"][0]["observed"] = {
            "review_semantic_digest": "previously-displayed",
        }
        result = self.run_batch()
        self.assertEqual(result["approved"], 2)
        self.assertEqual(result["skipped"][0]["image_id"], "a")

    def test_missing_fingerprint_does_not_silently_confirm(self):
        del self.snapshot["items"][0]["review_image_digest"]
        self.assertEqual(self.run_batch()["approved"], 2)

    def test_unknown_write_outcome_is_not_retried_or_counted_as_success(self):
        original = self.client.approve

        def approve(image_id, revision):
            result = original(image_id, revision)
            if image_id == "b":
                raise OSError("reply lost after write")
            return result

        self.client.approve = approve
        result = self.run_batch()
        self.assertEqual(result["approved"], 2)
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(result["summary"]["pending_review"], 0)
        self.assertEqual(len([c for c in self.client.calls if c[:2] == ("approve", "b")]), 1)

    def test_cancel_retains_completed_and_leaves_remaining_pending(self):
        original = self.client.approve

        def approve(image_id, revision):
            result = original(image_id, revision)
            self.cancel.set()
            return result

        self.client.approve = approve
        result = self.run_batch()
        self.assertEqual(result["approved"], 1)
        self.assertEqual(result["remaining"], 2)
        self.assertEqual(result["summary"]["pending_review"], 2)

    def test_standalone_refresh_cannot_adopt_external_change_into_batch(self):
        fixture = staged_fixtures.StandaloneStagedAuditTests()
        fixture.setUp()
        try:
            worker = _StagedAuditOperationWorker(fixture.client)
            snapshot = worker._execute("batch_prepare", {})
            changed = copy.deepcopy(fixture.document)
            changed["description"] = "external change after consent"
            staged_fixtures._write(fixture.label_path, changed)
            result = worker._execute("batch_approve", {
                "items": snapshot["items"], "cancel": threading.Event(),
            })
            self.assertEqual(result["approved"], 0)
            item = fixture.store.read_item(fixture.run_id, "image-a")
            self.assertEqual(item["audit_decisions"], [])
            fresh = worker._execute("batch_prepare", {})
            result = worker._execute("batch_approve", {
                "items": fresh["items"], "cancel": threading.Event(),
            })
            self.assertEqual(result["approved"], 1)
        finally:
            fixture.tearDown()


class BatchUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def setUp(self):
        self.items = [_pending("a", 0), _pending("b", 1), _pending("c", 2)]
        self.client = _AuditClient(self.items)
        self.widget = _widget(self.items)
        self.session = StagedAuditUiSession(self.widget, self.client)
        self.session.begin()
        self.assertTrue(_wait_until(self.session.is_idle))

    def tearDown(self):
        self.assertTrue(_finish_session(self.session))
        self.widget.close()

    def test_dialog_ranges_counts_and_empty_confirmation(self):
        dialog = BatchConfirmationDialog(self.items, 1, config={"language": "zh_CN"})
        try:
            self.assertEqual(dialog.selected_items(), self.items)
            self.assertIn("3", dialog.confirm_button.text())
            self.assertEqual(dialog.range_combo.itemText(0), "\u5168\u90e8\u5f85\u786e\u8ba4")
            dialog.range_combo.setCurrentIndex(1)
            self.assertEqual(dialog.selected_items(), self.items[1:])
            self.assertIn("2", dialog.confirm_button.text())
            dialog._current_sequence = 99
            dialog._update_count()
            self.assertFalse(dialog.confirm_button.isEnabled())
        finally:
            dialog.close()

    def test_batch_button_confirms_without_loading_every_image(self):
        self.session._choose_batch = lambda candidates: candidates
        self.session.dialog.batch_button.click()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(self.session._summary["pending_review"], 0)
        self.assertTrue(self.session._round_complete)
        self.assertEqual(len(self.widget.loaded), 1)
        self.assertIn("3 confirmed", self.session.dialog.batch_result_label.text())
        gui_thread = int(QtCore.QThread.currentThreadId())
        self.assertNotIn(gui_thread, self.client.call_thread_ids)

    def test_eligibility_excludes_deferred_needs_fix_errors_and_unpresented(self):
        self.session._deferred_ids.add("a")
        self.client.items["b"]["review_status"] = "needs_fix"
        self.client.items["c"]["integrity_error"] = "missing"
        choose = mock.Mock(return_value=[])
        self.session._choose_batch = choose
        self.session.confirm_batch()
        self.assertTrue(_wait_until(self.session.is_idle))
        choose.assert_called_once_with([])
        self.session._deferred_ids.clear()
        self.session._item_presented = False
        choose.reset_mock()
        self.session.confirm_batch()
        self.assertTrue(_wait_until(self.session.is_idle))
        choose.assert_called_once_with([])
        self.assertFalse(any(c[0] == "approve" for c in self.client.calls))

    def test_cancel_range_selection_writes_nothing(self):
        self.session._choose_batch = lambda candidates: []
        self.session.confirm_batch()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(self.client.calls, [])
        self.assertTrue(self.widget.canvas.isEnabled())

    def test_partial_batch_advances_to_remaining_without_reconfirming_success(self):
        self.client.items["b"]["review_status"] = "needs_fix"
        self.session._choose_batch = lambda candidates: candidates
        self.session.confirm_batch()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(self.session.current_image_id, "b")
        self.assertEqual(self.session._summary["pending_review"], 1)
        self.assertIn("2 confirmed", self.session.dialog.batch_result_label.text())

    def test_saving_current_changes_before_batch_uses_saved_fingerprint(self):
        self.widget.dirty = True
        self.session._ask_unsaved_changes = lambda: QtWidgets.QMessageBox.Save

        def save():
            self.widget.dirty = False
            self.client.items["a"]["review_semantic_digest"] = "saved-changes"
            self.widget.label_file = mock.Mock(semantic_digest="saved-changes")
            return True

        self.widget.save_file = save
        self.session._choose_batch = lambda candidates: candidates
        self.session.confirm_batch()
        self.assertTrue(_wait_until(self.session.is_idle))
        self.assertEqual(self.session._summary["pending_review"], 0)

    def test_dirty_cancel_and_save_failure_do_not_start_batch(self):
        self.widget.dirty = True
        for answer in (QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Save):
            self.session._ask_unsaved_changes = lambda: answer
            self.assertFalse(self.session.confirm_batch())
        self.assertEqual(self.client.list_calls, 1)
        self.assertEqual(self.client.calls, [])

    def test_stop_and_close_wait_for_current_write_without_blocking_gui(self):
        started, release = threading.Event(), threading.Event()
        original = self.client.approve

        def approve(image_id, revision):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test release not signaled")
            return original(image_id, revision)

        self.client.approve = approve
        self.session._choose_batch = lambda candidates: candidates
        self.session.confirm_batch()
        try:
            self.assertTrue(_wait_until(started.is_set))
            self.assertFalse(self.widget.canvas.isEnabled())
            self.assertFalse(self.session.dialog.batch_button.isEnabled())
            self.assertFalse(self.session.dialog.cancel_batch_button.isHidden())
            heartbeat = []
            QtCore.QTimer.singleShot(0, lambda: heartbeat.append(True))
            self.assertTrue(_wait_until(lambda: bool(heartbeat)))
            self.session.dialog.cancel_batch_button.click()
            self.assertFalse(self.session.finish())
            self.assertIsNotNone(self.session.worker_thread)
        finally:
            release.set()
        self.assertTrue(_wait_until(lambda: self.session.worker_thread is None))
        self.assertEqual(self.client.items["a"]["review_status"], "staged_approved")
        self.assertEqual(self.client.items["b"]["review_status"], "pending")
        self.assertEqual(self.client.items["c"]["review_status"], "pending")
