"""Phase 1 commit protocols and in-memory test/reference adapters."""

import copy
import threading
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .auto_labeling_contracts import (
    annotation_commit_event_id_v1,
    validate_annotation_commit_event_v1,
)


class CommitStoreError(RuntimeError):
    """Base error for the Phase 1 structural store protocol."""


class StoreConflictError(CommitStoreError):
    """Raised when an item revision or attempt does not match."""


class InjectedCommitCrash(BaseException):
    """Simulate process death without normal exception recovery."""

    def __init__(self, point):
        self.point = point
        super().__init__(point)


@runtime_checkable
class CommitStoreProtocolV1(Protocol):
    """Strict structural protocol needed by the Phase 1 coordinator."""

    def read_item(self, image_id):
        pass

    def write_commit_intent(
        self,
        image_id,
        attempt_id,
        intent,
        expected_item_revision,
        result_summary=None,
    ):
        pass

    def checkpoint_staged_commit(
        self,
        image_id,
        attempt_id,
        *,
        expected_item_revision,
        staged_document_digest,
        staged_annotation_digest,
        target_count,
        zero_target,
        semantic_change,
    ):
        pass

    def rollback_prepared_commit(
        self, image_id, attempt_id, expected_item_revision
    ):
        pass

    def reset_interrupted_item(self, image_id, expected_item_revision):
        pass

    def mark_existing_annotation_skipped(
        self, image_id, attempt_id, expected_item_revision
    ):
        pass

    def mark_commit_conflict(
        self, image_id, conflict_code, expected_item_revision
    ):
        pass

    def rebuild_summary(self):
        pass


@runtime_checkable
class AnnotationCommitSinkProtocolV1(Protocol):
    """Production-facing event interface without any audit UI dependency."""

    def publish(self, event, *, label_path=None):
        pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_annotation_commit_event_v1(
    *,
    project_id,
    session_id,
    run_id,
    image_id,
    attempt_id,
    writer_kind,
    commit_scope,
    mutation_mode,
    document_digest,
    semantic_digest,
    source_image_digest=None,
    base_item_revision=None,
    created_at=None,
):
    """Build and validate the frozen deterministic commit event."""

    event = {
        "event_schema_version": 1,
        "event_id": "0" * 64,
        "project_id": project_id,
        "session_id": session_id,
        "run_id": run_id,
        "image_id": image_id,
        "attempt_id": attempt_id,
        "writer_kind": writer_kind,
        "commit_scope": commit_scope,
        "mutation_mode": mutation_mode,
        "document_digest": document_digest,
        "semantic_digest": semantic_digest,
        "source_image_digest": source_image_digest,
        "base_item_revision": base_item_revision,
        "created_at": created_at or _utc_now(),
    }
    event["event_id"] = annotation_commit_event_id_v1(event)
    validate_annotation_commit_event_v1(event)
    return event


class InMemoryCommitStoreV1:
    """Thread-safe reference adapter; it deliberately has no file backend."""

    def __init__(self):
        self._lock = threading.RLock()
        self._items = {}
        self._continuous_events = {}
        self._manual_events = {}
        self.summary = {"revision": 0, "counts": {}}

    def create_item(self, image_id, attempt_id, execution_status="running"):
        with self._lock:
            if image_id in self._items:
                raise StoreConflictError("duplicate image_id")
            item = {
                "image_id": image_id,
                "item_revision": 0,
                "execution_status": execution_status,
                "failure_resolution": "not_applicable",
                "staged_commit_status": "none",
                "source_commit_status": "not_applicable",
                "review_status": "not_applicable",
                "latest_attempt_id": attempt_id,
                "commit_intent": None,
                "digests": {
                    "pre_file_sha256": None,
                    "pre_document_digest": None,
                    "pre_semantic_digest": None,
                    "intended_document_digest": None,
                    "intended_semantic_digest": None,
                    "staged_document_digest": None,
                    "staged_annotation_digest": None,
                },
                "result_summary": {
                    "target_count": None,
                    "zero_target": None,
                    "semantic_change": None,
                    "skip_reason": None,
                },
                "last_commit_event_id": None,
                "recent_commit_event_ids": [],
                "conflict": None,
            }
            self._items[image_id] = item
            return copy.deepcopy(item)

    def begin_attempt(self, image_id, attempt_id):
        """Create or reset an uncommitted standalone item for a new attempt."""

        with self._lock:
            if image_id not in self._items:
                return self.create_item(image_id, attempt_id)
            item = self._items[image_id]
            if item["staged_commit_status"] == "prepared" or (
                item["staged_commit_status"] == "committed"
                and item["execution_status"] == "succeeded"
            ):
                raise StoreConflictError("committed item cannot begin attempt")
            item["latest_attempt_id"] = attempt_id
            item["execution_status"] = "running"
            item["failure_resolution"] = "not_applicable"
            item["conflict"] = None
            item["result_summary"]["skip_reason"] = None
            return self._advance(item)

    def _item_for_update(self, image_id, expected_item_revision):
        try:
            item = self._items[image_id]
        except KeyError as exc:
            raise StoreConflictError("unknown image_id") from exc
        if item["item_revision"] != expected_item_revision:
            raise StoreConflictError("item revision mismatch")
        return item

    @staticmethod
    def _check_attempt(item, attempt_id):
        if item["latest_attempt_id"] != attempt_id:
            raise StoreConflictError("attempt mismatch")

    @staticmethod
    def _advance(item):
        item["item_revision"] += 1
        return copy.deepcopy(item)

    def read_item(self, image_id):
        with self._lock:
            try:
                return copy.deepcopy(self._items[image_id])
            except KeyError as exc:
                raise StoreConflictError("unknown image_id") from exc

    def write_commit_intent(
        self,
        image_id,
        attempt_id,
        intent,
        expected_item_revision,
        result_summary=None,
    ):
        with self._lock:
            item = self._item_for_update(image_id, expected_item_revision)
            self._check_attempt(item, attempt_id)
            if item["commit_intent"] is not None:
                raise StoreConflictError("commit intent already exists")
            if intent.get("attempt_id") != attempt_id:
                raise StoreConflictError("intent attempt mismatch")
            if intent.get("phase") != "prepared":
                raise StoreConflictError("intent phase mismatch")
            item["commit_intent"] = copy.deepcopy(intent)
            item["staged_commit_status"] = "prepared"
            item["digests"].update(
                {
                    "pre_file_sha256": intent["pre_file_sha256"],
                    "pre_document_digest": intent["pre_document_digest"],
                    "pre_semantic_digest": intent["pre_semantic_digest"],
                    "intended_document_digest": intent[
                        "intended_document_digest"
                    ],
                    "intended_semantic_digest": intent[
                        "intended_semantic_digest"
                    ],
                }
            )
            if result_summary is not None:
                expected_fields = {
                    "target_count",
                    "zero_target",
                    "semantic_change",
                }
                if set(result_summary) != expected_fields:
                    raise StoreConflictError("invalid prepared result summary")
                item["result_summary"].update(copy.deepcopy(result_summary))
            return self._advance(item)

    def checkpoint_staged_commit(
        self,
        image_id,
        attempt_id,
        *,
        expected_item_revision,
        staged_document_digest,
        staged_annotation_digest,
        target_count,
        zero_target,
        semantic_change,
    ):
        with self._lock:
            item = self._item_for_update(image_id, expected_item_revision)
            self._check_attempt(item, attempt_id)
            intent = item["commit_intent"]
            if item["staged_commit_status"] != "prepared" or intent is None:
                raise StoreConflictError("item is not prepared")
            if intent["intended_document_digest"] != staged_document_digest:
                raise StoreConflictError("intended document digest mismatch")
            if intent["intended_semantic_digest"] != staged_annotation_digest:
                raise StoreConflictError("intended semantic digest mismatch")
            item["execution_status"] = "succeeded"
            item["failure_resolution"] = "not_applicable"
            item["staged_commit_status"] = "committed"
            item["source_commit_status"] = "pending"
            item["commit_intent"] = None
            item["digests"].update(
                {
                    "staged_document_digest": staged_document_digest,
                    "staged_annotation_digest": staged_annotation_digest,
                }
            )
            item["result_summary"].update(
                {
                    "target_count": target_count,
                    "zero_target": zero_target,
                    "semantic_change": semantic_change,
                }
            )
            item["review_status"] = "pending"
            return self._advance(item)

    def rollback_prepared_commit(
        self, image_id, attempt_id, expected_item_revision
    ):
        with self._lock:
            item = self._item_for_update(image_id, expected_item_revision)
            self._check_attempt(item, attempt_id)
            if item["staged_commit_status"] != "prepared":
                raise StoreConflictError("item is not prepared")
            item["commit_intent"] = None
            item["staged_commit_status"] = "none"
            item["execution_status"] = "queued"
            item["digests"]["intended_document_digest"] = None
            item["digests"]["intended_semantic_digest"] = None
            return self._advance(item)

    def reset_interrupted_item(self, image_id, expected_item_revision):
        with self._lock:
            item = self._item_for_update(image_id, expected_item_revision)
            if item["commit_intent"] is not None:
                raise StoreConflictError("prepared item requires recovery")
            if item["execution_status"] == "running":
                item["execution_status"] = "queued"
                return self._advance(item)
            return copy.deepcopy(item)

    def mark_existing_annotation_skipped(
        self, image_id, attempt_id, expected_item_revision
    ):
        with self._lock:
            item = self._item_for_update(image_id, expected_item_revision)
            self._check_attempt(item, attempt_id)
            if item["commit_intent"] is not None:
                raise StoreConflictError("cannot skip a prepared item")
            item["execution_status"] = "skipped"
            item["failure_resolution"] = "not_applicable"
            item["staged_commit_status"] = "none"
            item["source_commit_status"] = "not_applicable"
            item["review_status"] = "not_applicable"
            item["result_summary"]["skip_reason"] = "existing_annotation"
            return self._advance(item)

    def mark_commit_conflict(
        self, image_id, conflict_code, expected_item_revision
    ):
        with self._lock:
            item = self._item_for_update(image_id, expected_item_revision)
            item["execution_status"] = "conflict"
            item["failure_resolution"] = "unresolved"
            item["conflict"] = {"code": conflict_code}
            return self._advance(item)

    def rebuild_summary(self):
        with self._lock:
            counts = {}
            for item in self._items.values():
                status = item["execution_status"]
                counts[status] = counts.get(status, 0) + 1
            self.summary = {
                "revision": self.summary["revision"] + 1,
                "counts": counts,
            }
            return copy.deepcopy(self.summary)

    @staticmethod
    def _append_event_id(item, event_id):
        item["last_commit_event_id"] = event_id
        recent = item["recent_commit_event_ids"]
        if event_id not in recent:
            recent.append(event_id)
            del recent[:-32]

    def record_continuous_event(self, event):
        validate_annotation_commit_event_v1(event)
        event_id = event["event_id"]
        with self._lock:
            previous = self._continuous_events.get(event_id)
            if previous is not None:
                if (
                    previous["document_digest"] == event["document_digest"]
                    and previous["semantic_digest"] == event["semantic_digest"]
                ):
                    return "IDEMPOTENT_NO_OP"
                raise StoreConflictError("commit event id collision")
            item = self._items.get(event["image_id"])
            if item is None:
                raise StoreConflictError("unknown image_id")
            self._check_attempt(item, event["attempt_id"])
            if item["staged_commit_status"] != "committed":
                raise StoreConflictError("continuous event before checkpoint")
            if (
                item["digests"]["staged_document_digest"]
                != event["document_digest"]
                or item["digests"]["staged_annotation_digest"]
                != event["semantic_digest"]
            ):
                raise StoreConflictError("continuous event digest mismatch")
            self._continuous_events[event_id] = copy.deepcopy(event)
            self._append_event_id(item, event_id)
            self._advance(item)
            return "NOTIFIED"

    def apply_manual_event(self, event):
        validate_annotation_commit_event_v1(event)
        event_id = event["event_id"]
        with self._lock:
            previous = self._manual_events.get(event_id)
            if previous is not None:
                if (
                    previous["document_digest"] == event["document_digest"]
                    and previous["semantic_digest"] == event["semantic_digest"]
                ):
                    return "IDEMPOTENT_NO_OP"
                raise StoreConflictError("commit event id collision")
            item = self._items.get(event["image_id"])
            if item is None:
                raise StoreConflictError("unknown image_id")
            if (
                item["digests"]["staged_document_digest"]
                == event["document_digest"]
                and item["digests"]["staged_annotation_digest"]
                == event["semantic_digest"]
            ):
                self._manual_events[event_id] = copy.deepcopy(event)
                self._append_event_id(item, event_id)
                self._advance(item)
                return "IDEMPOTENT_NO_OP"
            item["staged_commit_status"] = "committed"
            item["source_commit_status"] = "pending"
            item["digests"]["staged_document_digest"] = event[
                "document_digest"
            ]
            item["digests"]["staged_annotation_digest"] = event[
                "semantic_digest"
            ]
            item["review_status"] = (
                "stale"
                if item["review_status"] in {"staged_approved", "approved"}
                else "pending"
            )
            self._manual_events[event_id] = copy.deepcopy(event)
            self._append_event_id(item, event_id)
            self._advance(item)
            return "APPLIED_MANUAL_REVISION"


class FaultInjectionCommitStoreV1(InMemoryCommitStoreV1):
    """Reference store with deterministic one-shot crash injection."""

    def __init__(self, crash_point=None):
        super().__init__()
        self.crash_point = crash_point
        self.crashed = False

    def fault(self, point):
        if self.crash_point == point and not self.crashed:
            self.crashed = True
            raise InjectedCommitCrash(point)


class InMemoryAnnotationCommitSinkV1:
    """Reference sink enforcing continuous/manual ownership and idempotency."""

    def __init__(self, store):
        if not isinstance(store, CommitStoreProtocolV1):
            raise TypeError("store does not implement CommitStoreProtocolV1")
        self.store = store
        self._source_events = {}

    def publish(self, event, *, label_path=None):
        validate_annotation_commit_event_v1(event)
        writer_kind = event["writer_kind"]
        if writer_kind == "CONTINUOUS":
            return self.store.record_continuous_event(event)
        if writer_kind in {"MANUAL_SAVE", "AUTO_SAVE", "DELETE_LABEL"}:
            if label_path is None:
                raise StoreConflictError("manual event requires label path")
            from .auto_labeling_commit import resolve_existing_label

            current = resolve_existing_label(label_path)
            if writer_kind == "DELETE_LABEL":
                if (
                    current.document_digest != "MISSING"
                    or current.semantic_digest != "MISSING"
                ):
                    raise StoreConflictError("stale_commit_event")
            elif (
                current.document_digest != event["document_digest"]
                or current.semantic_digest != event["semantic_digest"]
            ):
                raise StoreConflictError("stale_commit_event")
            return self.store.apply_manual_event(event)
        event_id = event["event_id"]
        previous = self._source_events.get(event_id)
        if previous is not None:
            if (
                previous["document_digest"] == event["document_digest"]
                and previous["semantic_digest"] == event["semantic_digest"]
            ):
                return "IDEMPOTENT_NO_OP"
            raise StoreConflictError("commit event id collision")
        self._source_events[event_id] = copy.deepcopy(event)
        return "NOTIFIED_SOURCE"
