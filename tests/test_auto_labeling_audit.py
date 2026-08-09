import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from anylabeling.views.labeling.utils.auto_labeling_audit import (
    InMemoryStagedAuditClientV1,
    build_staged_audit_decision_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_contracts import (
    ContractValidationError,
    canonical_document_digest_v1,
    semantic_annotation_digest_v1,
    validate_audit_decision_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_run_store import (
    InMemoryAnnotationCommitSinkV1,
    InMemoryCommitStoreV1,
    StoreConflictError,
    build_annotation_commit_event_v1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    InMemoryFastRunStoreV1,
)


def _document(description="", marker=None):
    data = {
        "version": "3.3.7",
        "flags": {},
        "shapes": [],
        "imagePath": "image.png",
        "imageData": None,
        "imageHeight": 6,
        "imageWidth": 8,
        "description": description,
    }
    if marker is not None:
        data["writerMarker"] = marker
    return data


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


class AuditDecisionContractTests(unittest.TestCase):
    def test_staged_decision_is_strict_and_source_authority_is_forbidden(self):
        decision = build_staged_audit_decision_v1(
            project_id="project-a",
            run_id="run-a",
            image_id="image-a",
            session_id="session-a",
            reviewed_annotation_digest="alsem1:" + ("1" * 64),
            reviewed_image_digest="2" * 64,
            reviewer_action="APPROVE",
            base_item_revision=4,
            decision_id="12345678-1234-4234-8234-123456789abc",
            created_at="2026-08-09T00:00:00Z",
        )
        self.assertIs(validate_audit_decision_v1(decision), decision)

        invalid = copy.deepcopy(decision)
        invalid["authority_source_commit_sequence"] = 7
        with self.assertRaisesRegex(
            ContractValidationError,
            "staged_decision_source_authority_present",
        ):
            validate_audit_decision_v1(invalid)

        missing = copy.deepcopy(decision)
        del missing["base_item_revision"]
        with self.assertRaisesRegex(ContractValidationError, "missing_fields"):
            validate_audit_decision_v1(missing)


class StandaloneStagedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "image.png"
        self.label_path = self.root / "image.json"
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(self.image_path)
        self.document = _document()
        _write(self.label_path, self.document)
        self.image_digest = hashlib.sha256(
            self.image_path.read_bytes()
        ).hexdigest()
        self.run_id = "run-a"
        item = {
            "run_id": self.run_id,
            "image_id": "image-a",
            "sequence": 0,
            "item_revision": 0,
            "execution_status": "succeeded",
            "failure_resolution": "not_applicable",
            "staged_commit_status": "committed",
            "source_commit_status": "pending",
            "review_status": "pending",
            "latest_attempt_id": "attempt-a",
            "commit_intent": None,
            "prediction_attempts": [],
            "digests": {
                "staged_document_digest": canonical_document_digest_v1(
                    self.document
                ),
                "staged_annotation_digest": semantic_annotation_digest_v1(
                    self.document
                ),
                "source_image_digest": self.image_digest,
                "reviewed_annotation_digest": None,
                "reviewed_image_digest": None,
            },
            "result_summary": {
                "target_count": 0,
                "zero_target": True,
                "semantic_change": True,
                "error_code": None,
                "error_message": None,
                "skip_reason": None,
                "conflict_code": None,
            },
            "error": None,
            "conflict": None,
            "last_commit_event_id": None,
            "recent_commit_event_ids": [],
            "last_audit_decision_id": None,
            "audit_decisions": [],
            "created_at": "2026-08-09T00:00:00Z",
            "updated_at": "2026-08-09T00:00:00Z",
        }
        config = {"run_id": self.run_id}
        queue = {
            "run_id": self.run_id,
            "entries": [{"sequence": 0, "image_id": "image-a"}],
        }
        state = {
            "run_id": self.run_id,
            "revision": 0,
            "review_progress": "NOT_STARTED",
            "updated_at": "2026-08-09T00:00:00Z",
        }
        self.store = InMemoryFastRunStoreV1(config, queue, state, [item])
        record = {
            "image_id": "image-a",
            "canonical_session_image_path": os.path.normcase(
                os.path.realpath(self.image_path)
            ),
            "canonical_session_label_path": os.path.normcase(
                os.path.realpath(self.label_path)
            ),
            "manifest_sequence": 0,
        }
        self.client = InMemoryStagedAuditClientV1(
            run_store=self.store,
            run_id=self.run_id,
            records_by_id={"image-a": record},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zero_target_approval_appends_full_decision_and_is_cas_safe(self):
        before = self.client.integrity_refresh("image-a")
        approved = self.client.approve(
            "image-a",
            before["item_revision"],
        )
        self.assertEqual(approved["review_status"], "staged_approved")
        item = self.store.read_item(self.run_id, "image-a")
        self.assertEqual(len(item["audit_decisions"]), 1)
        decision = item["audit_decisions"][0]
        self.assertEqual(decision["reviewer_action"], "APPROVE")
        self.assertEqual(
            decision["reviewed_annotation_digest"],
            semantic_annotation_digest_v1(self.document),
        )
        self.assertTrue(item["result_summary"]["zero_target"])
        with self.assertRaisesRegex(StoreConflictError, "revision mismatch"):
            self.client.needs_fix("image-a", before["item_revision"])

    def test_summary_keeps_staged_and_source_approval_distinct(self):
        approved = self.client.approve("image-a", 0)
        summary = self.client.summary()

        self.assertEqual(summary["staged_approved"], 1)
        self.assertEqual(summary["source_approved"], 0)
        self.assertNotIn("approved", summary)

        self.store.update_item(
            self.run_id,
            "image-a",
            approved["item_revision"],
            {"review_status": "approved"},
        )
        summary = self.client.summary()
        self.assertEqual(summary["staged_approved"], 0)
        self.assertEqual(summary["source_approved"], 1)
        self.assertNotIn("approved", summary)

    def test_integrity_refresh_distinguishes_document_and_semantic_change(
        self,
    ):
        approved = self.client.approve("image-a", 0)
        approved_revision = approved["item_revision"]

        document_only = _document()
        document_only["imageWidth"] = 9
        _write(self.label_path, document_only)
        refreshed = self.client.integrity_refresh("image-a")
        self.assertGreater(refreshed["item_revision"], approved_revision)
        self.assertEqual(refreshed["review_status"], "staged_approved")

        _write(self.label_path, _document(description="semantic change"))
        refreshed = self.client.integrity_refresh("image-a")
        self.assertEqual(refreshed["review_status"], "stale")

    def test_needs_fix_semantic_edit_returns_pending(self):
        needs_fix = self.client.needs_fix("image-a", 0)
        self.assertEqual(needs_fix["review_status"], "needs_fix")
        _write(self.label_path, _document(description="fixed"))
        refreshed = self.client.integrity_refresh("image-a")
        self.assertEqual(refreshed["review_status"], "pending")

    def test_missing_and_corrupt_labels_fail_closed(self):
        self.label_path.unlink()
        missing = self.client.integrity_refresh("image-a")
        self.assertEqual(missing["integrity_error"], "audit_label_missing")
        with self.assertRaisesRegex(Exception, "audit_label_missing"):
            self.client.approve("image-a", missing["item_revision"])

        self.label_path.write_text("{broken", encoding="utf-8")
        corrupt = self.client.integrity_refresh("image-a")
        self.assertEqual(corrupt["integrity_error"], "audit_label_invalid")


class ManualEventInvalidationTests(unittest.TestCase):
    def _event(self, kind, document, revision=0):
        return build_annotation_commit_event_v1(
            project_id="project-a",
            session_id="session-a",
            run_id="run-a",
            image_id="image-a",
            attempt_id="attempt-a",
            writer_kind=kind,
            commit_scope="STAGED",
            mutation_mode="APPLY_MANUAL_REVISION",
            document_digest=canonical_document_digest_v1(document),
            semantic_digest=semantic_annotation_digest_v1(document),
            source_image_digest="a" * 64,
            base_item_revision=revision,
            created_at="2026-08-09T00:00:00Z",
        )

    def test_only_semantic_or_image_change_invalidates_approval(self):
        store = InMemoryCommitStoreV1()
        store.create_item("image-a", "attempt-a")
        base = _document()
        first = self._event("MANUAL_SAVE", base)
        store.apply_manual_event(first)
        store._items["image-a"]["review_status"] = "staged_approved"

        document_only = _document()
        document_only["imageWidth"] = 9
        store.apply_manual_event(self._event("MANUAL_SAVE", document_only))
        self.assertEqual(
            store.read_item("image-a")["review_status"],
            "staged_approved",
        )

        semantic = _document(description="changed")
        store.apply_manual_event(self._event("MANUAL_SAVE", semantic))
        self.assertEqual(store.read_item("image-a")["review_status"], "stale")

    def test_sink_reread_still_enforces_disk_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "image.json"
            document = _document()
            _write(label_path, document)
            store = InMemoryCommitStoreV1()
            store.create_item("image-a", "attempt-a")
            sink = InMemoryAnnotationCommitSinkV1(store)
            self.assertEqual(
                sink.publish(
                    self._event("MANUAL_SAVE", document),
                    label_path=label_path,
                ),
                "APPLIED_MANUAL_REVISION",
            )


if __name__ == "__main__":
    unittest.main()
