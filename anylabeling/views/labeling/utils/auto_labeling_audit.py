"""Staged review protocols and the standalone Phase 6 reference service."""

import copy
import hashlib
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .auto_labeling_commit import resolve_existing_label
from .auto_labeling_contracts import validate_audit_decision_v1
from .auto_labeling_run_store import (
    InMemoryAnnotationCommitSinkV1,
    StoreConflictError,
)


_REVIEWABLE = {"pending", "needs_fix", "stale"}
_APPROVED = {"staged_approved", "approved"}


class StagedAuditError(RuntimeError):
    """Fail-closed staged review error with a stable code."""

    def __init__(self, code, detail=""):
        self.code = str(code)
        self.detail = str(detail or "")
        message = self.code
        if self.detail:
            message += f": {self.detail}"
        super().__init__(message)


@runtime_checkable
class StagedAuditClientProtocolV1(Protocol):
    def list_review_items(self):
        pass

    def read_review_item(self, image_id):
        pass

    def summary(self):
        pass

    def integrity_refresh(self, image_id):
        pass

    def approve(self, image_id, expected_revision):
        pass

    def needs_fix(self, image_id, expected_revision):
        pass


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_staged_audit_decision_v1(
    *,
    project_id,
    run_id,
    image_id,
    session_id,
    reviewed_annotation_digest,
    reviewed_image_digest,
    reviewer_action,
    base_item_revision,
    decision_id=None,
    created_at=None,
):
    decision = {
        "audit_decision_schema_version": 1,
        "decision_id": decision_id or str(uuid.uuid4()),
        "project_id": project_id,
        "run_id": run_id,
        "image_id": image_id,
        "session_id": session_id,
        "scope": "STAGED",
        "reviewed_annotation_digest": reviewed_annotation_digest,
        "reviewed_image_digest": reviewed_image_digest,
        "reviewer_action": reviewer_action,
        "authority_source_commit_sequence": None,
        "base_item_revision": base_item_revision,
        "base_overlay_revision": None,
        "created_at": created_at or _utc_now(),
    }
    return validate_audit_decision_v1(decision)


class StandaloneAnnotationCommitSinkV1:
    """Mirror the Phase 1 commit store into the standalone review store."""

    def __init__(self, commit_store, run_store, run_id):
        self._sink = InMemoryAnnotationCommitSinkV1(commit_store)
        self.commit_store = commit_store
        self.run_store = run_store
        self.run_id = run_id
        self._lock = threading.RLock()

    def publish(self, event, *, label_path=None):
        with self._lock:
            result = self._sink.publish(event, label_path=label_path)
            source = self.commit_store.read_item(event["image_id"])
            target = self.run_store.read_item(
                self.run_id,
                event["image_id"],
            )
            if (
                target.get("last_commit_event_id") == event["event_id"]
                and target["digests"].get("staged_document_digest")
                == source["digests"].get("staged_document_digest")
                and target["digests"].get("staged_annotation_digest")
                == source["digests"].get("staged_annotation_digest")
            ):
                return result
            digests = copy.deepcopy(target["digests"])
            digests.update(copy.deepcopy(source["digests"]))
            changes = {
                "staged_commit_status": source["staged_commit_status"],
                "source_commit_status": source["source_commit_status"],
                "review_status": source["review_status"],
                "digests": digests,
                "last_commit_event_id": source["last_commit_event_id"],
                "recent_commit_event_ids": copy.deepcopy(
                    source["recent_commit_event_ids"]
                ),
            }
            self.run_store.update_item(
                self.run_id,
                event["image_id"],
                target["item_revision"],
                changes,
            )
            return result


class InMemoryStagedAuditClientV1:
    """Session-local staged audit authority for standalone operation."""

    def __init__(
        self,
        *,
        run_store,
        run_id,
        records_by_id,
        project_id=None,
        session_id=None,
        commit_store=None,
    ):
        self.run_store = run_store
        self.run_id = run_id
        self.project_id = project_id or f"standalone:{run_id}"
        self.session_id = session_id or f"standalone-session:{run_id}"
        self.records_by_id = dict(records_by_id)
        self.records_by_path = {}
        self._lock = threading.RLock()
        for image_id, record in self.records_by_id.items():
            if _record_value(record, "image_id") != image_id:
                raise StagedAuditError("audit_record_image_identity_mismatch")
            image_path = _canonical(
                _record_value(record, "canonical_session_image_path")
            )
            if image_path in self.records_by_path:
                raise StagedAuditError("audit_duplicate_image_path")
            self.records_by_path[image_path] = record
        self.annotation_commit_sink = (
            StandaloneAnnotationCommitSinkV1(
                commit_store,
                run_store,
                run_id,
            )
            if commit_store is not None
            else None
        )

    def commit_binding(self, image_path):
        record = self.records_by_path.get(_canonical(image_path))
        if record is None:
            return None
        image_id = _record_value(record, "image_id")
        item = self.run_store.read_item(self.run_id, image_id)
        return {
            "project_id": self.project_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "image_id": image_id,
            "attempt_id": item.get("latest_attempt_id"),
            "item_revision": item["item_revision"],
            "label_path": _canonical(
                _record_value(record, "canonical_session_label_path")
            ),
            "source_image_digest": item["digests"].get("source_image_digest"),
            "sink": self.annotation_commit_sink,
        }

    def _ordered_items(self):
        queue = self.run_store.read_queue(self.run_id)
        entries = sorted(queue["entries"], key=lambda value: value["sequence"])
        return [
            self.run_store.read_item(self.run_id, entry["image_id"])
            for entry in entries
        ]

    def _view(self, item, integrity_error=None):
        record = self.records_by_id.get(item["image_id"])
        if record is None:
            raise StagedAuditError("audit_record_missing", item["image_id"])
        image_path = _canonical(
            _record_value(record, "canonical_session_image_path")
        )
        label_path = _canonical(
            _record_value(record, "canonical_session_label_path")
        )
        if not os.path.isfile(image_path) or os.path.islink(image_path):
            integrity_error = "audit_image_not_regular"
        return {
            "image_id": item["image_id"],
            "sequence": item["sequence"],
            "item_revision": item["item_revision"],
            "canonical_image_path": image_path,
            "canonical_label_path": label_path,
            "filename": os.path.basename(image_path),
            "review_status": item["review_status"],
            "staged_commit_status": item["staged_commit_status"],
            "source_commit_status": item["source_commit_status"],
            "target_count": item["result_summary"].get("target_count"),
            "zero_target": item["result_summary"].get("zero_target"),
            "integrity_error": integrity_error,
        }

    def list_review_items(self):
        with self._lock:
            return [
                self._view(item)
                for item in self._ordered_items()
                if item["review_status"] in _REVIEWABLE
            ]

    def read_review_item(self, image_id):
        with self._lock:
            item = self.run_store.read_item(self.run_id, image_id)
            return self._view(item)

    def summary(self):
        with self._lock:
            counts = {
                "staged_approved": 0,
                "source_approved": 0,
                "needs_fix": 0,
                "pending": 0,
                "stale": 0,
            }
            for item in self._ordered_items():
                status = item["review_status"]
                if status == "approved":
                    counts["source_approved"] += 1
                elif status in counts:
                    counts[status] += 1
            counts["pending_review"] = sum(
                counts[field] for field in ("pending", "needs_fix", "stale")
            )
            return counts

    def _disk_facts(self, item):
        record = self.records_by_id.get(item["image_id"])
        if record is None:
            raise StagedAuditError("audit_record_missing", item["image_id"])
        image_path = _canonical(
            _record_value(record, "canonical_session_image_path")
        )
        label_path = _canonical(
            _record_value(record, "canonical_session_label_path")
        )
        if not os.path.isfile(image_path) or os.path.islink(image_path):
            raise StagedAuditError("audit_image_not_regular")
        current = resolve_existing_label(label_path)
        if current.presence == "MISSING":
            raise StagedAuditError("audit_label_missing")
        if current.presence == "INVALID":
            raise StagedAuditError("audit_label_invalid")
        return current, _file_sha256(image_path)

    def integrity_refresh(self, image_id):
        with self._lock:
            item = self.run_store.read_item(self.run_id, image_id)
            try:
                current, image_digest = self._disk_facts(item)
            except StagedAuditError as exc:
                if item["review_status"] in _APPROVED:
                    updated = self.run_store.update_item(
                        self.run_id,
                        image_id,
                        item["item_revision"],
                        {"review_status": "stale"},
                    )
                    return self._view(updated, exc.code)
                return self._view(item, exc.code)
            digests = copy.deepcopy(item["digests"])
            old_semantic = digests.get("staged_annotation_digest")
            old_image = digests.get("source_image_digest")
            semantic_changed = old_semantic != current.semantic_digest
            image_changed = old_image not in {None, image_digest}
            document_changed = (
                digests.get("staged_document_digest")
                != current.document_digest
            )
            if not (document_changed or semantic_changed or image_changed):
                return self._view(item)
            digests["staged_document_digest"] = current.document_digest
            digests["staged_annotation_digest"] = current.semantic_digest
            digests["source_image_digest"] = image_digest
            review_status = item["review_status"]
            if review_status in _APPROVED and (
                semantic_changed or image_changed
            ):
                review_status = "stale"
            elif review_status == "needs_fix" and semantic_changed:
                review_status = "pending"
            updated = self.run_store.update_item(
                self.run_id,
                image_id,
                item["item_revision"],
                {
                    "staged_commit_status": "committed",
                    "source_commit_status": "pending",
                    "review_status": review_status,
                    "digests": digests,
                },
            )
            self._refresh_progress()
            return self._view(updated)

    def _verified_item(self, image_id, expected_revision):
        item = self.run_store.read_item(self.run_id, image_id)
        if item["item_revision"] != expected_revision:
            raise StoreConflictError("item revision mismatch")
        if item["staged_commit_status"] != "committed":
            raise StagedAuditError("audit_staged_checkpoint_missing")
        current, image_digest = self._disk_facts(item)
        digests = item["digests"]
        if current.document_digest != digests.get("staged_document_digest"):
            raise StagedAuditError("audit_document_digest_mismatch")
        if current.semantic_digest != digests.get("staged_annotation_digest"):
            raise StagedAuditError("audit_semantic_digest_mismatch")
        expected_image = digests.get("source_image_digest")
        if expected_image is not None and image_digest != expected_image:
            raise StagedAuditError("audit_image_digest_mismatch")
        return item, current, image_digest

    def _decide(self, image_id, expected_revision, action):
        with self._lock:
            item, current, image_digest = self._verified_item(
                image_id,
                expected_revision,
            )
            decision = build_staged_audit_decision_v1(
                project_id=self.project_id,
                run_id=self.run_id,
                image_id=image_id,
                session_id=self.session_id,
                reviewed_annotation_digest=current.semantic_digest,
                reviewed_image_digest=image_digest,
                reviewer_action=action,
                base_item_revision=item["item_revision"],
            )
            decisions = copy.deepcopy(item.get("audit_decisions", []))
            if any(
                value.get("decision_id") == decision["decision_id"]
                for value in decisions
            ):
                return self._view(item)
            decisions.append(decision)
            digests = copy.deepcopy(item["digests"])
            digests["reviewed_annotation_digest"] = current.semantic_digest
            digests["reviewed_image_digest"] = image_digest
            updated = self.run_store.update_item(
                self.run_id,
                image_id,
                item["item_revision"],
                {
                    "review_status": (
                        "staged_approved"
                        if action == "APPROVE"
                        else "needs_fix"
                    ),
                    "last_audit_decision_id": decision["decision_id"],
                    "audit_decisions": decisions,
                    "digests": digests,
                },
            )
            self._refresh_progress()
            view = self._view(updated)
            view["decision"] = copy.deepcopy(decision)
            return view

    def approve(self, image_id, expected_revision):
        return self._decide(image_id, expected_revision, "APPROVE")

    def needs_fix(self, image_id, expected_revision):
        return self._decide(image_id, expected_revision, "NEEDS_FIX")

    def _refresh_progress(self):
        state = self.run_store.read_state(self.run_id)
        pending = self.summary()["pending_review"]
        progress = "COMPLETED" if pending == 0 else "IN_PROGRESS"
        if state.get("review_progress") == progress:
            return
        self.run_store.update_state(
            self.run_id,
            state["revision"],
            {"review_progress": progress},
        )
