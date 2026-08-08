"""Event-driven Phase 4 zero-delay continuous auto-labeling controller."""

import copy
import hashlib
import os
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from PyQt5 import QtCore

from anylabeling.app_info import __version__
from anylabeling.services.auto_labeling.prediction_job import (
    PredictionOutcome,
    PredictionRequest,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    LabelConflictError,
    commit_label_for_image_v1,
    resolve_existing_label,
    validate_output_paths_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_run_store import (
    InMemoryCommitStoreV1,
)

from .auto_labeling_sequence import FastRunOptionsV1, thaw_json


_INTENT_PRIORITY = {"NONE": 0, "PAUSE": 1, "STOP": 2, "CLOSE": 3}
_INPUT_FAILURE_CODES = {
    "failed_input",
    "failed_input_not_regular_file",
    "exif_not_normalized",
    "session_image_path_escape",
    "session_image_symlink",
}


class FastControllerError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        self.detail = str(detail or "")
        super().__init__(code if not self.detail else f"{code}: {self.detail}")


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def standalone_image_id_v1(path):
    value = unicodedata.normalize("NFC", _canonical(path)).encode("utf-8")
    return (
        "standalone:" + hashlib.sha256(b"ALSTANDALONE1\0" + value).hexdigest()
    )


def _empty_result_summary():
    return {
        "target_count": None,
        "zero_target": None,
        "error_code": None,
        "error_message": None,
        "skip_reason": None,
        "conflict_code": None,
        "semantic_change": None,
    }


def _new_fast_item(run_id, image_id, sequence, now):
    return {
        "run_id": run_id,
        "image_id": image_id,
        "sequence": sequence,
        "item_revision": 0,
        "execution_status": "queued",
        "failure_resolution": "not_applicable",
        "staged_commit_status": "none",
        "source_commit_status": "not_applicable",
        "review_status": "not_applicable",
        "latest_attempt_id": None,
        "commit_intent": None,
        "prediction_attempts": [],
        "digests": {
            "pre_file_sha256": None,
            "pre_document_digest": None,
            "pre_semantic_digest": None,
            "intended_document_digest": None,
            "intended_semantic_digest": None,
            "staged_document_digest": None,
            "staged_annotation_digest": None,
        },
        "result_summary": _empty_result_summary(),
        "error": None,
        "conflict": None,
        "created_at": now,
        "updated_at": now,
    }


def _derive_basic_counts(items):
    statuses = {
        "queued": 0,
        "running": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
        "conflict": 0,
    }
    pending_review = 0
    unresolved = 0
    for item in items:
        statuses[item["execution_status"]] += 1
        if item["review_status"] in {"pending", "needs_fix", "stale"}:
            pending_review += 1
        if (
            item["execution_status"] in {"queued", "running", "conflict"}
            or item["failure_resolution"] == "unresolved"
        ):
            unresolved += 1
    return {
        "total": len(items),
        "execution_status": statuses,
        "pending_review": pending_review,
        "unresolved_processing": unresolved,
    }


class InMemoryFastRunStoreV1:
    """Session-local run facts for standalone Fast mode."""

    def __init__(self, config, queue, state, items):
        self._lock = threading.RLock()
        self.config = copy.deepcopy(config)
        self.queue = copy.deepcopy(queue)
        self.state = copy.deepcopy(state)
        self.items = {item["image_id"]: copy.deepcopy(item) for item in items}
        self.state["counts"] = _derive_basic_counts(self.items.values())

    def read_config(self, run_id):
        self._check_run(run_id)
        return copy.deepcopy(self.config)

    def read_queue(self, run_id):
        self._check_run(run_id)
        return copy.deepcopy(self.queue)

    def read_state(self, run_id):
        self._check_run(run_id)
        return copy.deepcopy(self.state)

    def read_item(self, run_id, image_id):
        self._check_run(run_id)
        return copy.deepcopy(self.items[image_id])

    def list_items(self, run_id):
        self._check_run(run_id)
        return [
            copy.deepcopy(item)
            for item in sorted(
                self.items.values(), key=lambda value: value["sequence"]
            )
        ]

    def update_state(self, run_id, expected_state_revision, changes):
        with self._lock:
            self._check_run(run_id)
            if self.state["revision"] != expected_state_revision:
                raise FastControllerError("state_revision_mismatch")
            self.state.update(copy.deepcopy(changes))
            self.state["revision"] += 1
            self.state["updated_at"] = _utc_now()
            return copy.deepcopy(self.state)

    def update_item(self, run_id, image_id, expected_item_revision, changes):
        with self._lock:
            self._check_run(run_id)
            item = self.items[image_id]
            if item["item_revision"] != expected_item_revision:
                raise FastControllerError("item_revision_mismatch")
            item.update(copy.deepcopy(changes))
            item["item_revision"] += 1
            item["updated_at"] = _utc_now()
            self.state["counts"] = _derive_basic_counts(self.items.values())
            self.state["revision"] += 1
            self.state["updated_at"] = _utc_now()
            return copy.deepcopy(item)

    def _check_run(self, run_id):
        if run_id != self.config["run_id"]:
            raise FastControllerError("unknown_run_id")


@dataclass(frozen=True)
class _StandaloneRecord:
    image_id: str
    canonical_session_image_path: str
    canonical_session_label_path: str
    source_image_digest: str | None
    manifest_sequence: int


@dataclass(frozen=True)
class _StandaloneActivation:
    run_id: str
    run_store: InMemoryFastRunStoreV1
    commit_store: InMemoryCommitStoreV1
    records_by_id: dict


def _mark_standalone_skip(item, reason):
    item["execution_status"] = "skipped"
    item["result_summary"]["skip_reason"] = reason


def _mark_standalone_conflict(item, code):
    item["execution_status"] = "conflict"
    item["failure_resolution"] = "unresolved"
    item["conflict"] = {"code": code}
    item["result_summary"]["conflict_code"] = code


def _same_label_content(first, second):
    if first.raw_file_sha256 == second.raw_file_sha256:
        return True
    return (
        first.presence in {"VALID_EMPTY", "VALID_NONEMPTY"}
        and second.presence in {"VALID_EMPTY", "VALID_NONEMPTY"}
        and first.document_digest == second.document_digest
    )


def _reject_standalone_shadow_labels(path_entries, output_dir):
    if output_dir is None:
        return
    for entry in path_entries:
        image_path = entry["image_path"]
        label_path = entry["label_path"]
        sibling_path = _canonical(os.path.splitext(image_path)[0] + ".json")
        if sibling_path == label_path or not os.path.lexists(sibling_path):
            continue
        authoritative = resolve_existing_label(label_path)
        shadow = resolve_existing_label(sibling_path)
        if not _same_label_content(authoritative, shadow):
            raise LabelConflictError("conflict_shadow_label", sibling_path)


def create_standalone_fast_activation_v1(
    image_paths,
    options,
    *,
    output_dir=None,
):
    if not isinstance(options, FastRunOptionsV1):
        raise TypeError("options must be FastRunOptionsV1")
    if options.workset_source != "CURRENT_FILE_LIST_SNAPSHOT":
        raise FastControllerError("standalone_workset_source_invalid")
    paths = [_canonical(path) for path in image_paths]
    if not paths:
        raise FastControllerError("standalone_workset_empty")
    if len(set(paths)) != len(paths):
        raise FastControllerError("standalone_workset_duplicate_path")
    run_id = str(uuid.uuid4())
    now = _utc_now()
    records = []
    path_entries = []
    for sequence, image_path in enumerate(paths):
        if not os.path.isfile(image_path) or os.path.islink(image_path):
            raise FastControllerError(
                "failed_input_not_regular_file", image_path
            )
        image_id = standalone_image_id_v1(image_path)
        label_name = (
            os.path.splitext(os.path.basename(image_path))[0] + ".json"
        )
        label_path = (
            os.path.join(_canonical(output_dir), label_name)
            if output_dir
            else os.path.splitext(image_path)[0] + ".json"
        )
        label_path = _canonical(label_path)
        records.append(
            _StandaloneRecord(
                image_id=image_id,
                canonical_session_image_path=image_path,
                canonical_session_label_path=label_path,
                source_image_digest=None,
                manifest_sequence=sequence,
            )
        )
        path_entries.append(
            {"image_path": image_path, "label_path": label_path}
        )
    allowed_root = _canonical(output_dir) if output_dir else None
    validate_output_paths_v1(
        path_entries,
        allowed_root=allowed_root,
        require_unique_basename=bool(output_dir),
    )
    _reject_standalone_shadow_labels(path_entries, output_dir)
    by_id = {record.image_id: record for record in records}
    if len(by_id) != len(records):
        raise FastControllerError("standalone_workset_duplicate_image_id")
    anchor = 0
    if options.range == "CURRENT_TO_END":
        matches = [
            record.manifest_sequence
            for record in records
            if record.image_id == options.current_anchor_image_id
        ]
        if len(matches) != 1:
            raise FastControllerError("current_anchor_image_id_not_found")
        anchor = matches[0]
    items = []
    for record in records:
        item = _new_fast_item(
            run_id, record.image_id, record.manifest_sequence, now
        )
        if record.manifest_sequence < anchor:
            _mark_standalone_skip(item, "outside_selected_range")
        else:
            existing = resolve_existing_label(
                record.canonical_session_label_path
            )
            if existing.presence == "INVALID":
                _mark_standalone_conflict(item, "invalid_existing_label")
            elif existing.presence in {"VALID_EMPTY", "VALID_NONEMPTY"} and (
                options.filter == "ONLY_WITHOUT_VALID_ANNOTATION"
                or options.write_policy == "SKIP_EXISTING"
            ):
                _mark_standalone_skip(item, "existing_annotation")
        items.append(item)
    queue = {
        "run_id": run_id,
        "entries": [
            {
                "sequence": record.manifest_sequence,
                "image_id": record.image_id,
            }
            for record in records
        ],
    }
    state = {
        "run_id": run_id,
        "revision": 0,
        "processing_status": "PREPARED",
        "review_progress": "NOT_STARTED",
        "phase": "IDLE",
        "control_intent": "NONE",
        "cursor_sequence": 0,
        "counts": {},
        "last_error": None,
        "created_at": now,
        "updated_at": now,
    }
    config = {
        "run_id": run_id,
        "project_id": None,
        "workset_source": "CURRENT_FILE_LIST_SNAPSHOT",
        "delay_seconds": 0.0,
        "range": options.range,
        "filter": options.filter,
        "write_policy": options.write_policy,
        "model_fingerprint": thaw_json(options.model_fingerprint),
        "parameter_snapshot": thaw_json(options.parameter_snapshot),
    }
    return _StandaloneActivation(
        run_id=run_id,
        run_store=InMemoryFastRunStoreV1(config, queue, state, items),
        commit_store=InMemoryCommitStoreV1(),
        records_by_id=by_id,
    )


def build_fast_run_summary_v1(items, state=None):
    items = list(items)
    summary = {
        "workset_total": len(items),
        "selected_by_range": 0,
        "eligible_for_inference": 0,
        "succeeded": 0,
        "zero_target": 0,
        "skipped_existing": 0,
        "skipped_outside_range": 0,
        "host_prepare_failed": 0,
        "failed_input": 0,
        "model_failed_unresolved": 0,
        "explicit_error_skips": 0,
        "conflicts": 0,
        "pending_review": 0,
        "remaining": 0,
        "processing_status": None,
        "review_progress": None,
        "completed_with_errors": False,
    }
    for item in items:
        status = item["execution_status"]
        resolution = item["failure_resolution"]
        result = item["result_summary"]
        skip_reason = result.get("skip_reason")
        error_code = result.get("error_code")
        if skip_reason == "outside_selected_range":
            summary["skipped_outside_range"] += 1
            continue
        summary["selected_by_range"] += 1
        if error_code == "host_prepare_failed":
            summary["host_prepare_failed"] += 1
            continue
        if skip_reason == "existing_annotation":
            summary["skipped_existing"] += 1
            continue
        if status == "conflict":
            summary["conflicts"] += 1
            if result.get("conflict_code") != "invalid_existing_label":
                summary["eligible_for_inference"] += 1
            continue
        summary["eligible_for_inference"] += 1
        if status == "succeeded":
            summary["succeeded"] += 1
            if result.get("zero_target") is True:
                summary["zero_target"] += 1
        elif status == "failed":
            if resolution == "auto_skip":
                summary["failed_input"] += 1
            elif resolution == "unresolved":
                summary["model_failed_unresolved"] += 1
        elif status == "skipped" and resolution == "explicit_skip":
            summary["explicit_error_skips"] += 1
        elif status in {"queued", "running"}:
            summary["remaining"] += 1
        if item["review_status"] in {"pending", "needs_fix", "stale"}:
            summary["pending_review"] += 1
    summary["completed_with_errors"] = any(
        summary[field]
        for field in (
            "host_prepare_failed",
            "failed_input",
            "model_failed_unresolved",
            "explicit_error_skips",
            "conflicts",
        )
    )
    if state is not None:
        if type(state) is not dict:
            raise FastControllerError("fast_run_summary_state_invalid")
        summary["processing_status"] = state.get("processing_status")
        summary["review_progress"] = state.get("review_progress")
    return summary


def read_fast_run_summary_v1(run_store, run_id):
    """Rebuild one consistent summary from persistent state and item facts."""

    for _attempt in range(3):
        before = run_store.read_state(run_id)
        items = run_store.list_items(run_id)
        after = run_store.read_state(run_id)
        if before.get("revision") == after.get("revision"):
            return build_fast_run_summary_v1(items, after)
    raise FastControllerError("fast_run_summary_snapshot_unstable")


class FastAutoLabelingController(QtCore.QObject):
    state_changed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(object)
    waiting_error = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)
    safe_to_close = QtCore.pyqtSignal(int)

    _registry_lock = threading.Lock()
    _active_manager_ids = set()

    def __init__(
        self,
        runner,
        options,
        *,
        host_context=None,
        standalone_image_paths=(),
        standalone_output_dir=None,
        context_replacer=None,
        pose_config=None,
        created_by_app_version=__version__,
    ):
        super().__init__()
        if not isinstance(options, FastRunOptionsV1):
            raise TypeError("options must be FastRunOptionsV1")
        self.runner = runner
        self.options = options
        self.host_context = host_context
        self.standalone_image_paths = tuple(standalone_image_paths)
        self.standalone_output_dir = standalone_output_dir
        self.context_replacer = context_replacer
        self.pose_config = copy.deepcopy(pose_config)
        self.created_by_app_version = created_by_app_version

        self.run_id = None
        self.run_store = None
        self.commit_store = None
        self.event_sink = None
        self.records_by_id = {}
        self.queue_entries = []
        self.queue_position = 0
        self.control_intent = "NONE"
        self.phase = "IDLE"
        self.active_request = None
        self.waiting_image_id = None
        self.waiting_error_details = None
        self.hard_error = None
        self.modified_image_ids = set()
        self._handled_identities = set()
        self._started = False
        self._finished = False
        self._runner_started_here = False
        self._manager_key = id(self.runner.model_manager)
        self._safe_generation = None
        self._live_summary = None
        self._item_contributions = {}

        self.runner.outcome_ready.connect(self._on_outcome_ready)
        self.runner.runner_idle.connect(self._on_runner_idle)
        self.runner.runner_error.connect(self._on_runner_error)
        self.runner.safe_to_close.connect(self._on_safe_to_close)

    def start(self):
        if self._started or self._finished:
            return False
        lease = getattr(self.runner.model_manager, "inference_lease", None)
        if bool(getattr(lease, "is_active", False)):
            raise FastControllerError("inference_lease_busy")
        with self._registry_lock:
            if self._manager_key in self._active_manager_ids:
                raise FastControllerError("fast_controller_already_active")
            self._active_manager_ids.add(self._manager_key)
        try:
            self._set_phase("LOADING")
            self._activate_run()
            if not self.runner.start():
                raise FastControllerError("prediction_runner_start_failed")
            self._runner_started_here = True
            self._set_run_state(
                processing_status="RUNNING",
                review_progress="NOT_STARTED",
                phase="INFERENCING",
                control_intent="NONE",
                last_error=None,
            )
            self._started = True
            self._set_phase("INFERENCING")
            self._dispatch_next()
            return True
        except Exception as exc:
            if self.run_store is not None and self.run_id is not None:
                try:
                    self._set_run_state(
                        processing_status="FAILED",
                        phase="FINISHED",
                        last_error={
                            "code": getattr(
                                exc, "code", "fast_run_start_failed"
                            ),
                            "message": str(exc),
                        },
                    )
                except Exception:
                    pass
            with self._registry_lock:
                self._active_manager_ids.discard(self._manager_key)
            if self._runner_started_here:
                self.runner.shutdown_when_idle()
            raise

    def _activate_run(self):
        if self.host_context is None:
            activation = create_standalone_fast_activation_v1(
                self.standalone_image_paths,
                self.options,
                output_dir=self.standalone_output_dir,
            )
            self.run_id = activation.run_id
            self.run_store = activation.run_store
            self.commit_store = activation.commit_store
            self.records_by_id = activation.records_by_id
            return self._load_queue()

        if not bool(getattr(self.host_context, "images_ready", False)):
            raise FastControllerError("images_not_ready")
        activate = getattr(self.host_context, "activate_fast_run", None)
        if not callable(activate):
            raise FastControllerError("host_activation_service_unavailable")
        request = self.options.activation_request(
            run_id=str(uuid.uuid4()),
            session_attempt_id=str(uuid.uuid4()),
            created_by_app_version=self.created_by_app_version,
        )
        activation = activate(request)
        bound_context = activation.context
        self.host_context = bound_context
        self.run_id = activation.run_id
        self.run_store = bound_context.run_store
        self.commit_store = activation.commit_store
        self.event_sink = activation.annotation_commit_sink
        self.records_by_id = {
            _record_value(record, "image_id"): record
            for record in bound_context.image_records_by_path.values()
        }
        if callable(self.context_replacer):
            self.context_replacer(bound_context)
        self._load_queue()

    def _load_queue(self):
        queue = self.run_store.read_queue(self.run_id)
        self.queue_entries = list(queue["entries"])
        self.queue_position = 0
        items = self.run_store.list_items(self.run_id)
        self._live_summary = build_fast_run_summary_v1(
            items, self.run_store.read_state(self.run_id)
        )
        self._item_contributions = {
            item["image_id"]: build_fast_run_summary_v1([item])
            for item in items
        }

    def _track_item(self, item):
        if self._live_summary is None:
            return item
        image_id = item["image_id"]
        previous = self._item_contributions.get(image_id)
        current = build_fast_run_summary_v1([item])
        if previous is not None:
            for field, value in current.items():
                if field in {
                    "processing_status",
                    "review_progress",
                    "completed_with_errors",
                }:
                    continue
                if type(value) is int:
                    self._live_summary[field] += value - previous[field]
        self._live_summary["completed_with_errors"] = any(
            self._live_summary[field]
            for field in (
                "host_prepare_failed",
                "failed_input",
                "model_failed_unresolved",
                "explicit_error_skips",
                "conflicts",
            )
        )
        self._item_contributions[image_id] = current
        return item

    def _set_phase(self, phase):
        self.phase = phase
        self.state_changed.emit(phase)

    def _set_run_state(self, **changes):
        state = self.run_store.read_state(self.run_id)
        return self.run_store.update_state(
            self.run_id, state["revision"], changes
        )

    def _update_item(self, image_id, changes):
        item = self.run_store.read_item(self.run_id, image_id)
        return self._track_item(
            self.run_store.update_item(
                self.run_id, image_id, item["item_revision"], changes
            )
        )

    def _dispatch_next(self):
        if self._finished or self.active_request is not None:
            return
        if not self.runner.is_idle():
            self._hard_fail("dispatch_before_runner_idle")
            return
        while self.queue_position < len(self.queue_entries):
            entry = self.queue_entries[self.queue_position]
            self.queue_position += 1
            item = self.run_store.read_item(self.run_id, entry["image_id"])
            if item["execution_status"] == "queued":
                self._dispatch_item(item)
                return
        self._finalize("complete")

    def _dispatch_item(self, item):
        image_id = item["image_id"]
        if not self.runner.is_idle():
            self._hard_fail("dispatch_before_runner_idle", image_id)
            return
        if item["execution_status"] not in {"queued", "failed"}:
            self._hard_fail("item_not_dispatchable", image_id)
            return
        if item["execution_status"] == "failed" and (
            item["failure_resolution"] != "unresolved"
        ):
            self._hard_fail("item_not_retryable", image_id)
            return
        record = self.records_by_id.get(image_id)
        if record is None:
            self._hard_fail("host_record_missing", image_id)
            return
        attempt_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = _utc_now()
        attempt = {
            "attempt_id": attempt_id,
            "job_id": job_id,
            "generation": self.runner.generation,
            "started_at": now,
            "ended_at": None,
            "status": "running",
            "error_code": None,
            "error_message": None,
        }
        attempts = copy.deepcopy(item["prediction_attempts"])
        attempts.append(attempt)
        result_summary = copy.deepcopy(item["result_summary"])
        result_summary.update(
            error_code=None,
            error_message=None,
            skip_reason=None,
            conflict_code=None,
        )
        updated = self._update_item(
            image_id,
            {
                "execution_status": "running",
                "failure_resolution": "not_applicable",
                "latest_attempt_id": attempt_id,
                "prediction_attempts": attempts,
                "result_summary": result_summary,
                "error": None,
                "conflict": None,
            },
        )
        if isinstance(self.commit_store, InMemoryCommitStoreV1):
            self.commit_store.begin_attempt(image_id, attempt_id)
        self._set_run_state(
            cursor_sequence=updated["sequence"],
            phase="INFERENCING",
        )
        parameters = thaw_json(self.options.parameter_snapshot)
        image_path = _record_value(record, "canonical_session_image_path")
        expected = _record_value(record, "source_image_digest") or None
        parameters.update(
            image_input_source=(
                "SESSION_MANIFEST"
                if self.host_context is not None
                else "STANDALONE_FILE_LIST"
            ),
            expected_image_sha256=expected,
        )
        if self.host_context is not None:
            parameters["session_images_root"] = getattr(
                self.host_context,
                "_session_root",
                os.path.dirname(image_path),
            )
        request = PredictionRequest(
            run_id=self.run_id,
            job_id=job_id,
            attempt_id=attempt_id,
            image_id=image_id,
            canonical_image_path=image_path,
            generation=self.runner.generation,
            parameter_snapshot=parameters,
            existing_shapes_input=[],
            delivery_mode="RETURN_ONLY",
        )
        self.active_request = request
        if not self.runner.submit(request):
            self.active_request = None
            self._hard_fail("prediction_submit_rejected", image_id)
            return
        # Publish the active filename and total before a long inference ends.
        # Completion and the next admission still remain gated by runner_idle.
        self._emit_progress(image_id)

    @QtCore.pyqtSlot(object)
    def _on_outcome_ready(self, outcome):
        if not isinstance(outcome, PredictionOutcome) or self._finished:
            return
        request = self.active_request
        if request is None or outcome.context.identity != request.identity:
            return
        if request.identity in self._handled_identities:
            return
        self._handled_identities.add(request.identity)
        try:
            if outcome.status == "succeeded":
                self._handle_success(outcome)
            else:
                self._handle_failure(outcome)
        except Exception as exc:  # noqa: B902
            self._hard_fail(
                getattr(exc, "code", "controller_outcome_failed"), str(exc)
            )

    def _handle_success(self, outcome):
        request = outcome.context
        payload = outcome.result
        if (
            payload.input_width is None
            or payload.input_height is None
            or payload.source_image_digest is None
        ):
            self._hard_fail(
                "prediction_input_metadata_missing", request.image_id
            )
            return
        record = self.records_by_id[request.image_id]
        expected_digest = _record_value(record, "source_image_digest") or None
        if (
            expected_digest is not None
            and payload.source_image_digest != expected_digest
        ):
            self._hard_fail(
                "prediction_source_digest_mismatch", request.image_id
            )
            return
        label_path = _record_value(record, "canonical_session_label_path")
        image_path = _record_value(record, "canonical_session_image_path")
        if self.host_context is not None:
            allowed_root = getattr(self.host_context, "_session_root", None)
        elif self.standalone_output_dir is not None:
            allowed_root = _canonical(self.standalone_output_dir)
        else:
            allowed_root = os.path.dirname(label_path)
        prediction = {
            "status": "succeeded",
            "shapes": payload.shapes,
            "replace": payload.replace,
            "description": payload.description,
            "zero_target": outcome.zero_target,
        }
        try:
            self._set_run_state(phase="COMMITTING")
            self._set_phase("COMMITTING")
            result = commit_label_for_image_v1(
                store=self.commit_store,
                image_id=request.image_id,
                attempt_id=request.attempt_id,
                label_path=label_path,
                image_path=image_path,
                image_height=payload.input_height,
                image_width=payload.input_width,
                prediction_outcome=prediction,
                write_policy=self.options.write_policy,
                pose_config=self.pose_config,
                allowed_root=allowed_root,
                event_sink=self.event_sink,
                project_id=(
                    self.host_context.project_id
                    if self.host_context is not None
                    else None
                ),
                session_id=(
                    self.host_context.active_session_id
                    if self.host_context is not None
                    else None
                ),
                run_id=self.run_id if self.host_context is not None else None,
                source_image_digest=payload.source_image_digest,
            )
            if isinstance(self.commit_store, InMemoryCommitStoreV1):
                self._sync_standalone_commit(request.image_id)
            self._finish_attempt(request, "succeeded", None, None)
            if result.composition.action != "skipped":
                self.modified_image_ids.add(request.image_id)
            self._emit_progress(request.image_id)
        except LabelConflictError as exc:
            if isinstance(self.commit_store, InMemoryCommitStoreV1):
                self._sync_standalone_commit(request.image_id)
            self._finish_attempt(request, "conflict", exc.code, str(exc))
            self._emit_progress(request.image_id)
        except Exception as exc:  # noqa: B902
            self._hard_fail(
                getattr(exc, "code", "commit_handler_failed"), str(exc)
            )

    def _handle_failure(self, outcome):
        code = outcome.error_code
        request = outcome.context
        if code == "conflict_image_changed":
            self._finish_attempt(
                request, "conflict", code, outcome.error_message
            )
            self._emit_progress(request.image_id)
            return
        if code in _INPUT_FAILURE_CODES:
            self._finish_attempt(
                request, "failed_input", code, outcome.error_message
            )
            self._emit_progress(request.image_id)
            return
        self._finish_attempt(
            request, "model_failed", code, outcome.error_message
        )
        error = {
            "image_id": request.image_id,
            "error_code": code,
            "error_message": outcome.error_message,
        }
        self._emit_progress(request.image_id)
        if self.control_intent in {"CLOSE", "STOP"}:
            return
        self.waiting_image_id = request.image_id
        self.waiting_error_details = error
        if self.control_intent == "PAUSE":
            return
        self._publish_waiting_error()

    def _publish_waiting_error(self):
        if self.waiting_image_id is None or self.waiting_error_details is None:
            return
        self._set_run_state(phase="WAITING_ERROR")
        self._set_phase("WAITING_ERROR")
        self.waiting_error.emit(copy.deepcopy(self.waiting_error_details))

    def _finish_attempt(self, request, result, error_code, error_message):
        item = self.run_store.read_item(self.run_id, request.image_id)
        if item["latest_attempt_id"] != request.attempt_id:
            return item
        attempts = copy.deepcopy(item["prediction_attempts"])
        for attempt in reversed(attempts):
            if attempt.get("attempt_id") == request.attempt_id:
                attempt.update(
                    ended_at=_utc_now(),
                    status=result,
                    error_code=error_code,
                    error_message=error_message,
                )
                break
        changes = {"prediction_attempts": attempts}
        summary = copy.deepcopy(item["result_summary"])
        if result == "failed_input":
            summary.update(error_code=error_code, error_message=error_message)
            changes.update(
                execution_status="failed",
                failure_resolution="auto_skip",
                result_summary=summary,
                error={"code": error_code, "message": error_message},
            )
        elif result == "model_failed":
            summary.update(error_code=error_code, error_message=error_message)
            changes.update(
                execution_status="failed",
                failure_resolution="unresolved",
                result_summary=summary,
                error={"code": error_code, "message": error_message},
            )
        elif result == "conflict":
            summary["conflict_code"] = error_code
            changes.update(
                execution_status="conflict",
                failure_resolution="unresolved",
                result_summary=summary,
                conflict={"code": error_code, "message": error_message},
            )
        return self._track_item(
            self.run_store.update_item(
                self.run_id,
                request.image_id,
                item["item_revision"],
                changes,
            )
        )

    def _sync_standalone_commit(self, image_id):
        source = self.commit_store.read_item(image_id)
        target = self.run_store.read_item(self.run_id, image_id)
        digests = copy.deepcopy(target["digests"])
        digests.update(source["digests"])
        summary = copy.deepcopy(target["result_summary"])
        summary.update(source["result_summary"])
        self._track_item(
            self.run_store.update_item(
                self.run_id,
                image_id,
                target["item_revision"],
                {
                    "execution_status": source["execution_status"],
                    "failure_resolution": source["failure_resolution"],
                    "staged_commit_status": source["staged_commit_status"],
                    "source_commit_status": source["source_commit_status"],
                    "review_status": source["review_status"],
                    "commit_intent": copy.deepcopy(source["commit_intent"]),
                    "digests": digests,
                    "result_summary": summary,
                    "conflict": copy.deepcopy(source["conflict"]),
                },
            )
        )

    def _emit_progress(self, image_id):
        state = self.run_store.read_state(self.run_id)
        statuses = state["counts"]["execution_status"]
        summary = copy.deepcopy(self._live_summary or {})
        record = self.records_by_id.get(image_id)
        image_path = _record_value(record, "canonical_session_image_path")
        summary.update(
            processing_status=state.get("processing_status"),
            review_progress=state.get("review_progress"),
            image_id=image_id,
            current_filename=os.path.basename(image_path or ""),
            processed=len(self.queue_entries)
            - statuses.get("queued", 0)
            - statuses.get("running", 0),
            total=len(self.queue_entries),
            skipped=statuses.get("skipped", 0),
            failed=statuses.get("failed", 0),
        )
        self.progress_changed.emit(summary)

    @QtCore.pyqtSlot(int)
    def _on_runner_idle(self, generation):
        if self._finished or generation != self.runner.generation:
            return
        try:
            self.active_request = None
            if self.hard_error is not None:
                self._finalize("failed")
            elif self.control_intent in {"CLOSE", "STOP"}:
                self._finalize("stopped")
            elif self.control_intent == "PAUSE":
                self._set_run_state(
                    processing_status="PAUSED",
                    phase="PAUSED",
                    control_intent="PAUSE",
                )
                self._set_phase("PAUSED")
            elif self.waiting_image_id is None:
                self._dispatch_next()
        except Exception as exc:  # noqa: B902
            self._hard_fail(
                getattr(exc, "code", "controller_idle_failed"), str(exc)
            )

    @QtCore.pyqtSlot(object)
    def _on_runner_error(self, error):
        self._hard_fail("prediction_runner_error", str(error))

    def _hard_fail(self, code, detail=""):
        if self.hard_error is None:
            self.hard_error = {"code": code, "message": str(detail or "")}
            self.runner.request_stop()
            try:
                self._set_run_state(
                    processing_status="FAILED",
                    phase="FINALIZING",
                    last_error=self.hard_error,
                )
            except Exception:
                pass
        if self.runner.is_idle():
            self._finalize("failed")

    def request_pause(self):
        if self._finished:
            return False
        previous = self.control_intent
        self._raise_intent("PAUSE")
        if self.control_intent != "PAUSE":
            return False
        if previous == "PAUSE":
            return True
        self.runner.request_pause()
        try:
            self._set_run_state(control_intent=self.control_intent)
        except Exception as exc:  # noqa: B902
            self._hard_fail("control_checkpoint_failed", str(exc))
            return False
        if self.runner.is_idle() and self.waiting_image_id is None:
            self._on_runner_idle(self.runner.generation)
        return True

    def resume(self, dirty_resolution="CLEAN"):
        if self._finished or self.control_intent != "PAUSE":
            return False
        if dirty_resolution == "KEEP_PAUSED":
            return False
        if dirty_resolution not in {"CLEAN", "SAVED", "DISCARDED"}:
            raise FastControllerError("invalid_dirty_resume_resolution")
        if not self.runner.resume():
            return False
        self.control_intent = "NONE"
        try:
            if self.waiting_image_id is not None:
                self._set_run_state(
                    processing_status="RUNNING",
                    phase="WAITING_ERROR",
                    control_intent="NONE",
                )
                self._publish_waiting_error()
            else:
                self._set_run_state(
                    processing_status="RUNNING",
                    phase="INFERENCING",
                    control_intent="NONE",
                )
                self._set_phase("INFERENCING")
                self._dispatch_next()
        except Exception as exc:  # noqa: B902
            self._hard_fail("control_checkpoint_failed", str(exc))
            return False
        return True

    def retry_current(self):
        if self._finished or self.waiting_image_id is None:
            return False
        if not self.runner.is_idle():
            return False
        image_id = self.waiting_image_id
        self.waiting_image_id = None
        self.waiting_error_details = None
        try:
            self._set_run_state(phase="INFERENCING")
            self._set_phase("INFERENCING")
            item = self.run_store.read_item(self.run_id, image_id)
            self._dispatch_item(item)
        except Exception as exc:  # noqa: B902
            self._hard_fail("retry_checkpoint_failed", str(exc))
            return False
        return True

    def skip_current(self):
        if self._finished or self.waiting_image_id is None:
            return False
        if not self.runner.is_idle():
            return False
        image_id = self.waiting_image_id
        try:
            item = self.run_store.read_item(self.run_id, image_id)
            summary = copy.deepcopy(item["result_summary"])
            summary["skip_reason"] = "model_error"
            self._track_item(
                self.run_store.update_item(
                    self.run_id,
                    image_id,
                    item["item_revision"],
                    {
                        "execution_status": "skipped",
                        "failure_resolution": "explicit_skip",
                        "result_summary": summary,
                    },
                )
            )
            self.waiting_image_id = None
            self.waiting_error_details = None
            self._set_run_state(phase="INFERENCING")
            self._set_phase("INFERENCING")
            self._emit_progress(image_id)
            self._dispatch_next()
        except Exception as exc:  # noqa: B902
            self._hard_fail("explicit_skip_checkpoint_failed", str(exc))
            return False
        return True

    def request_stop(self):
        if self._finished:
            return False
        previous = self.control_intent
        self._raise_intent("STOP")
        if self.control_intent != "STOP":
            return False
        if previous == "STOP":
            return True
        self.runner.request_stop()
        try:
            self._set_run_state(control_intent=self.control_intent)
        except Exception as exc:  # noqa: B902
            self._hard_fail("control_checkpoint_failed", str(exc))
            return False
        if self.runner.is_idle():
            self._finalize("stopped")
        return True

    def request_close(self):
        if self._safe_generation is not None:
            return self._safe_generation
        self._raise_intent("CLOSE")
        if not self._finished:
            try:
                self._set_run_state(control_intent="CLOSE")
            except Exception as exc:  # noqa: B902
                self._hard_fail("control_checkpoint_failed", str(exc))
        self._safe_generation = self.runner.request_safe_shutdown(
            "continuous_fast_close"
        )
        if not self._finished and self.runner.is_idle():
            self._finalize("stopped")
        return self._safe_generation

    def request_safe_shutdown(self, _reason):
        return self.request_close()

    def is_idle(self):
        return self.runner.is_idle()

    def requires_safe_shutdown(self):
        return self.runner.requires_safe_shutdown()

    def _raise_intent(self, requested):
        if _INTENT_PRIORITY[requested] > _INTENT_PRIORITY[self.control_intent]:
            self.control_intent = requested

    @QtCore.pyqtSlot(int)
    def _on_safe_to_close(self, generation):
        if generation == self._safe_generation:
            self.safe_to_close.emit(generation)

    def _finalize(self, reason):
        if self._finished:
            return
        items = self.run_store.list_items(self.run_id)
        summary = build_fast_run_summary_v1(items)
        if reason == "failed" or self.hard_error is not None:
            status = "FAILED"
        elif reason == "stopped":
            safe_results = (
                summary["succeeded"]
                + summary["failed_input"]
                + summary["explicit_error_skips"]
                + summary["skipped_existing"]
                + summary["skipped_outside_range"]
                + summary["host_prepare_failed"]
            )
            status = "CANCELLED" if safe_results == 0 else "PARTIAL"
        elif (
            summary["remaining"]
            or summary["model_failed_unresolved"]
            or summary["conflicts"]
        ):
            status = "PARTIAL"
        else:
            status = "COMPLETED"
        review_progress = (
            "NOT_STARTED" if summary["pending_review"] else "COMPLETED"
        )
        try:
            final_state = self._set_run_state(
                processing_status=status,
                review_progress=review_progress,
                phase="FINISHED",
                control_intent=self.control_intent,
                last_error=self.hard_error,
            )
            summary = build_fast_run_summary_v1(items, final_state)
        except Exception as exc:
            status = "FAILED"
            self.hard_error = {
                "code": "final_state_checkpoint_failed",
                "message": str(exc),
            }
            summary["processing_status"] = status
            summary["review_progress"] = review_progress
        self._finished = True
        self._set_phase("FINISHED")
        with self._registry_lock:
            self._active_manager_ids.discard(self._manager_key)
        if self.control_intent != "CLOSE":
            self.runner.shutdown_when_idle()
        self.finished.emit(summary)


class FastCanvasStateGuard:
    """Keep background Fast work independent from the visible canvas."""

    def __init__(self, labeling_widget):
        self.widget = labeling_widget
        self.filename = getattr(labeling_widget, "filename", None)
        file_list = getattr(labeling_widget, "file_list_widget", None)
        self.file_row = file_list.currentRow() if file_list is not None else -1
        self.zoom_mode = getattr(labeling_widget, "zoom_mode", None)
        zoom_widget = getattr(labeling_widget, "zoom_widget", None)
        self.zoom_value = _control_value(zoom_widget, "value")
        bars = getattr(labeling_widget, "scroll_bars", {})
        self.vertical_scroll = _control_value(
            bars.get(QtCore.Qt.Vertical), "value"
        )
        self.horizontal_scroll = _control_value(
            bars.get(QtCore.Qt.Horizontal), "value"
        )
        self.current_image_id = None
        context = getattr(labeling_widget, "auto_labeling_host_context", None)
        if context is not None and self.filename is not None:
            record = context.image_records_by_path.get(
                _canonical(self.filename)
            )
            self.current_image_id = _record_value(record, "image_id")
        elif self.filename is not None:
            self.current_image_id = standalone_image_id_v1(self.filename)
        self._file_list_enabled = (
            file_list.isEnabled() if file_list is not None else None
        )
        self._widget_states = {}
        for name in ("label_list", "unique_label_list", "flag_widget"):
            control = getattr(labeling_widget, name, None)
            if control is not None:
                self._widget_states[name] = control.isEnabled()
        auto_widget = getattr(labeling_widget, "auto_labeling_widget", None)
        if auto_widget is not None:
            self._widget_states["auto_labeling_widget"] = (
                auto_widget.isEnabled()
            )
        self._action_states = {}
        actions = getattr(labeling_widget, "actions", None)
        for name in (
            "open",
            "close",
            "delete_file",
            "delete_image_file",
            "save",
            "save_as",
            "run_all_images",
            "delete",
            "edit",
            "duplicate",
            "copy",
            "paste",
            "undo_last_point",
            "undo",
            "remove_point",
            "create_mode",
            "edit_mode",
            "create_rectangle_mode",
            "create_pose_mode",
            "create_rotation_mode",
            "create_circle_mode",
            "create_line_mode",
            "create_point_mode",
            "create_line_strip_mode",
            "open_next_image",
            "open_prev_image",
            "open_next_unchecked_image",
            "open_prev_unchecked_image",
        ):
            action = getattr(actions, name, None)
            if action is not None:
                self._action_states[name] = action.isEnabled()

    def _set_interaction_locked(self, locked):
        setattr(self.widget, "fast_auto_labeling_edit_locked", bool(locked))
        file_list = getattr(self.widget, "file_list_widget", None)
        if file_list is not None:
            file_list.setEnabled(
                False if locked else bool(self._file_list_enabled)
            )
        for name, enabled in self._widget_states.items():
            control = getattr(self.widget, name, None)
            if name == "auto_labeling_widget":
                control = getattr(self.widget, name, None)
            if control is not None:
                control.setEnabled(False if locked else enabled)
        actions = getattr(self.widget, "actions", None)
        for name, enabled in self._action_states.items():
            action = getattr(actions, name, None)
            if action is not None:
                action.setEnabled(False if locked else enabled)
        canvas = getattr(self.widget, "canvas", None)
        setter = getattr(canvas, "set_sequence_edit_locked", None)
        if callable(setter):
            setter(locked)

    def set_running(self, running):
        self._set_interaction_locked(bool(running))
        setattr(self.widget, "fast_auto_labeling_active", bool(running))

    def set_paused(self, paused):
        self._set_interaction_locked(not paused)
        auto_widget = getattr(self.widget, "auto_labeling_widget", None)
        if paused and auto_widget is not None:
            auto_widget.setEnabled(False)
        setattr(self.widget, "fast_auto_labeling_active", True)

    def restore(self, modified_image_ids):
        self.set_running(False)
        should_reload = (
            self.current_image_id in set(modified_image_ids)
            or getattr(self.widget, "filename", None) != self.filename
        )
        if should_reload and self.filename is not None:
            self.widget.load_file(self.filename)
        file_list = getattr(self.widget, "file_list_widget", None)
        if file_list is not None and self.file_row >= 0:
            file_list.setCurrentRow(self.file_row)
        if self.zoom_mode is not None:
            self.widget.zoom_mode = self.zoom_mode
        if self.zoom_value is not None:
            setter = getattr(self.widget, "set_zoom", None)
            if callable(setter):
                setter(self.zoom_value)
        bars = getattr(self.widget, "scroll_bars", {})
        if self.vertical_scroll is not None and QtCore.Qt.Vertical in bars:
            bars[QtCore.Qt.Vertical].setValue(self.vertical_scroll)
        if self.horizontal_scroll is not None and QtCore.Qt.Horizontal in bars:
            bars[QtCore.Qt.Horizontal].setValue(self.horizontal_scroll)
        canvas = getattr(self.widget, "canvas", None)
        if canvas is not None:
            selected = getattr(canvas, "selected_shapes", None)
            if type(selected) is list:
                selected.clear()

    def release_for_close(self):
        """Unlock without reloading before the parent's SAVE/DISCARD barrier."""

        self.set_running(False)


def _control_value(control, method):
    function = getattr(control, method, None)
    return function() if callable(function) else None
