"""Unified manual LabelingWidget commit bridge for staged annotations."""

import copy
import os

from .auto_labeling_commit import (
    atomic_delete_label_document,
    resolve_existing_label,
)
from .auto_labeling_run_store import build_annotation_commit_event_v1


class AnnotationCommitBridgeError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = str(code)
        self.detail = str(detail or "")
        message = self.code
        if self.detail:
            message += f": {self.detail}"
        super().__init__(message)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _record_value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


class AnnotationCommitBridgeV1:
    """Publish exactly one event after each successful authoritative mutation."""

    def __init__(self, labeling_widget):
        self.widget = labeling_widget
        self._pending_commit = None

    @property
    def pending_commit(self):
        return copy.deepcopy(self._pending_commit)

    def record_loaded_label(self, label_path):
        path = _canonical(label_path)
        current = resolve_existing_label(path)
        self.widget._loaded_label_path = path
        self.widget._loaded_label_document_digest = current.document_digest
        return current

    def pre_document_digest(self, label_path):
        if self._pending_commit is not None:
            raise AnnotationCommitBridgeError(
                "manual_commit_recovery_required"
            )
        path = _canonical(label_path)
        if getattr(self.widget, "_loaded_label_path", None) == path:
            digest = getattr(
                self.widget,
                "_loaded_label_document_digest",
                None,
            )
            if digest is not None:
                return digest
        return resolve_existing_label(path).document_digest

    def _standalone_client(self):
        return getattr(
            self.widget,
            "_standalone_auto_labeling_audit_client",
            None,
        )

    def _target(self, image_path):
        context = getattr(self.widget, "auto_labeling_host_context", None)
        if context is not None and getattr(context, "active_run_id", None):
            canonical_image = _canonical(image_path)
            record = context.image_records_by_path.get(canonical_image)
            if record is None:
                raise AnnotationCommitBridgeError(
                    "manual_save_image_identity_missing"
                )
            image_id = _record_value(record, "image_id")
            item = context.run_store.read_item(
                context.active_run_id,
                image_id,
            )
            return {
                "project_id": context.project_id,
                "session_id": context.active_session_id,
                "run_id": context.active_run_id,
                "image_id": image_id,
                "attempt_id": item.get("latest_attempt_id"),
                "item_revision": item["item_revision"],
                "label_path": _canonical(
                    _record_value(record, "canonical_session_label_path")
                ),
                "source_image_digest": _record_value(
                    record,
                    "source_image_digest",
                ),
                "sink": context.annotation_commit_sink,
            }
        client = self._standalone_client()
        binding = getattr(client, "commit_binding", None)
        if callable(binding):
            return binding(image_path)
        return None

    def authoritative_label_path(self, image_path):
        target = self._target(image_path)
        return None if target is None else target["label_path"]

    def _mark_failure(self, code, detail, image_path, label_path):
        self.widget.auto_labeling_commit_blocked = True
        self.widget.dirty = True
        save_action = getattr(
            getattr(self.widget, "actions", None),
            "save",
            None,
        )
        if callable(getattr(save_action, "setEnabled", None)):
            save_action.setEnabled(True)
        pending = self._pending_commit or {}
        event = pending.get("event", {})
        self.widget.auto_labeling_commit_error = {
            "code": code,
            "detail": str(detail),
            "image_path": _canonical(image_path),
            "label_path": _canonical(label_path),
            "event_id": event.get("event_id"),
            "document_digest": pending.get("document_digest"),
            "semantic_digest": pending.get("semantic_digest"),
            "raw_file_sha256": pending.get("raw_file_sha256"),
        }

    def _clear_failure(self):
        self.widget.auto_labeling_commit_blocked = False
        self.widget.auto_labeling_commit_error = None
        self._pending_commit = None

    def _ensure_failure_marked(
        self,
        error,
        image_path,
        label_path,
    ):
        if not getattr(self.widget, "auto_labeling_commit_blocked", False):
            self._mark_failure(
                error.code,
                error,
                image_path,
                label_path,
            )

    @staticmethod
    def _pending_record(event, image_path, label_path, current):
        return {
            "event": copy.deepcopy(event),
            "image_path": _canonical(image_path),
            "label_path": _canonical(label_path),
            "raw_file_sha256": current.raw_file_sha256,
            "document_digest": current.document_digest,
            "semantic_digest": current.semantic_digest,
        }

    @staticmethod
    def _target_matches_event(target, event):
        return all(
            target[field] == event[field]
            for field in (
                "project_id",
                "session_id",
                "run_id",
                "image_id",
                "attempt_id",
            )
        )

    def _publish(self, writer_kind, image_path, label_path, current):
        if self._pending_commit is not None:
            raise AnnotationCommitBridgeError(
                "manual_commit_recovery_required"
            )
        if getattr(self.widget, "_sequence_presentation_session", None):
            return "PRESENTATION_NO_EVENT"
        target = self._target(image_path)
        if target is None:
            self._clear_failure()
            return "UNBOUND_NO_EVENT"
        canonical_label = _canonical(label_path)
        if canonical_label != target["label_path"]:
            raise AnnotationCommitBridgeError(
                "manual_save_non_authoritative_path"
            )
        sink = target["sink"]
        event = build_annotation_commit_event_v1(
            project_id=target["project_id"],
            session_id=target["session_id"],
            run_id=target["run_id"],
            image_id=target["image_id"],
            attempt_id=target["attempt_id"],
            writer_kind=writer_kind,
            commit_scope="STAGED",
            mutation_mode="APPLY_MANUAL_REVISION",
            document_digest=current.document_digest,
            semantic_digest=current.semantic_digest,
            source_image_digest=target["source_image_digest"],
            base_item_revision=target["item_revision"],
        )
        self._pending_commit = self._pending_record(
            event,
            image_path,
            canonical_label,
            current,
        )
        try:
            if not callable(getattr(sink, "publish", None)):
                raise AnnotationCommitBridgeError(
                    "manual_commit_sink_unavailable"
                )
            result = sink.publish(
                copy.deepcopy(event),
                label_path=canonical_label,
            )
        except Exception as exc:
            code = getattr(exc, "code", "manual_commit_sink_failed")
            self._mark_failure(
                code,
                exc,
                image_path,
                canonical_label,
            )
            raise AnnotationCommitBridgeError(code, exc) from exc
        self._clear_failure()
        return result

    def publish_saved_label(self, writer_kind, image_path, label_path):
        if writer_kind not in {"MANUAL_SAVE", "AUTO_SAVE"}:
            raise AnnotationCommitBridgeError("invalid_manual_writer_kind")
        current = resolve_existing_label(label_path)
        if current.presence not in {"VALID_EMPTY", "VALID_NONEMPTY"}:
            error = AnnotationCommitBridgeError(
                "manual_save_document_unavailable"
            )
            self._ensure_failure_marked(error, image_path, label_path)
            raise error
        try:
            result = self._publish(
                writer_kind,
                image_path,
                label_path,
                current,
            )
        except AnnotationCommitBridgeError as error:
            self._ensure_failure_marked(error, image_path, label_path)
            raise
        self.widget._loaded_label_path = _canonical(label_path)
        self.widget._loaded_label_document_digest = current.document_digest
        return result

    def delete_label(self, image_path, label_path):
        if self._pending_commit is not None:
            raise AnnotationCommitBridgeError(
                "manual_commit_recovery_required"
            )
        canonical_label = _canonical(label_path)
        target = self._target(image_path)
        if target is not None and canonical_label != target["label_path"]:
            error = AnnotationCommitBridgeError(
                "manual_save_non_authoritative_path"
            )
            self._ensure_failure_marked(error, image_path, canonical_label)
            raise error
        pre_digest = self.pre_document_digest(canonical_label)
        try:
            missing = atomic_delete_label_document(
                canonical_label,
                pre_document_digest=pre_digest,
                allowed_root=os.path.dirname(canonical_label),
            )
        except Exception as exc:
            code = getattr(exc, "code", "label_delete_failed")
            error = AnnotationCommitBridgeError(code, exc)
            self._ensure_failure_marked(
                error,
                image_path,
                canonical_label,
            )
            raise error from exc
        self.widget._loaded_label_path = canonical_label
        self.widget._loaded_label_document_digest = "MISSING"
        try:
            return self._publish(
                "DELETE_LABEL",
                image_path,
                canonical_label,
                missing,
            )
        except AnnotationCommitBridgeError as error:
            self._ensure_failure_marked(
                error,
                image_path,
                canonical_label,
            )
            raise

    def integrity_refresh(self):
        if not getattr(self.widget, "auto_labeling_commit_blocked", False):
            return None
        pending = self._pending_commit
        if pending is None:
            raise AnnotationCommitBridgeError(
                "manual_commit_pending_record_missing"
            )
        image_path = pending["image_path"]
        current_image_path = getattr(self.widget, "filename", None)
        if (
            not current_image_path
            or _canonical(current_image_path) != image_path
        ):
            raise AnnotationCommitBridgeError(
                "manual_commit_pending_image_mismatch"
            )
        target = self._target(image_path)
        if target is None:
            raise AnnotationCommitBridgeError(
                "manual_commit_pending_target_missing"
            )
        event = pending["event"]
        if _canonical(target["label_path"]) != pending[
            "label_path"
        ] or not self._target_matches_event(target, event):
            raise AnnotationCommitBridgeError(
                "manual_commit_pending_identity_mismatch"
            )
        current = resolve_existing_label(pending["label_path"])
        observed = (
            current.raw_file_sha256,
            current.document_digest,
            current.semantic_digest,
        )
        intended = (
            pending["raw_file_sha256"],
            pending["document_digest"],
            pending["semantic_digest"],
        )
        if observed != intended:
            code = "manual_commit_pending_disk_mismatch"
            self._mark_failure(
                code,
                code,
                image_path,
                pending["label_path"],
            )
            raise AnnotationCommitBridgeError(code)
        sink = target["sink"]
        if not callable(getattr(sink, "publish", None)):
            code = "manual_commit_sink_unavailable"
            self._mark_failure(
                code,
                code,
                image_path,
                pending["label_path"],
            )
            raise AnnotationCommitBridgeError(code)
        try:
            result = sink.publish(
                copy.deepcopy(event),
                label_path=pending["label_path"],
            )
        except Exception as exc:
            code = getattr(exc, "code", "manual_commit_sink_failed")
            self._mark_failure(
                code,
                exc,
                image_path,
                pending["label_path"],
            )
            raise AnnotationCommitBridgeError(code, exc) from exc
        self.widget._loaded_label_path = pending["label_path"]
        self.widget._loaded_label_document_digest = current.document_digest
        self._clear_failure()
        return result
