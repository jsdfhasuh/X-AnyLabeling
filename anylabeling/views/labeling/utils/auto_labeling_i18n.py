"""Localized text for the unified auto-labeling workflow."""

from types import MappingProxyType

from PyQt5 import QtCore


TRANSLATION_CONTEXT_V1 = "AutoLabelingPhase8"

AUTO_LABELING_TEXT_V1 = MappingProxyType(
    {
        "continuous_entry": "Continuous auto-labeling...",
        "continuous_title": "Continuous auto-labeling",
        "all_images": "All images",
        "current_to_end": "Current image to end",
        "base_range": "Base range",
        "all": "All",
        "only_without_valid_annotation": (
            "Only images without a valid annotation"
        ),
        "filter": "Filter",
        "skip_existing": "Skip existing annotations",
        "force_replace": "Force replace",
        "replace_confirmation_title": "Confirm replacement",
        "replace_confirmation": (
            "Replace may overwrite {count} valid existing annotation(s) "
            "in this scope. This choice is used only for this operation. "
            "Continue?"
        ),
        "write_policy": "Write policy",
        "seconds_suffix": " s",
        "dwell_time": "Dwell time",
        "model_and_parameters": "Model and parameters",
        "workset_count": "Workset images",
        "existing_annotation_count": "Existing annotations",
        "host_prepare_failure": "Host preparation failures",
        "fast_mode": "Fast",
        "fast_description": (
            "Fast: infer and save each image in the background without "
            "switching the main canvas."
        ),
        "start_fast": "Start Fast auto-labeling",
        "visible_mode": "Visible",
        "visible_description": (
            "Visible: display each result after a verified save, then "
            "advance after the configured dwell time."
        ),
        "start_visible": "Start Visible auto-labeling",
        "progress_title": "Continuous auto-labeling progress",
        "preparing": "Preparing",
        "configured_dwell_time": "Configured dwell time",
        "remaining_dwell_time": "Remaining dwell time",
        "workset_total": "Workset total",
        "selected_by_range": "Selected by range",
        "eligible_for_inference": "Eligible for inference",
        "succeeded": "Safely saved",
        "zero_target": "Zero target",
        "skipped_existing": "Skipped existing",
        "skipped_outside_range": "Skipped outside range",
        "host_prepare_failed": "Host preparation failed",
        "failed_input": "Failed input",
        "model_failed_unresolved": "Unresolved model failures",
        "explicit_error_skips": "Explicit error skips",
        "conflicts": "Conflicts",
        "pending_review": "Pending review",
        "pending_review_count": "Pending review {count}",
        "labels_tab": "Labels",
        "auto_labeling_tab": "Auto labeling",
        "audit_tab": "Review ({count})",
        "audit_empty": "No pending image in this session.",
        "remaining": "Remaining",
        "processing_status": "Processing status",
        "completed_with_errors": "Completed with errors",
        "pause": "Pause and edit",
        "pause_tooltip": (
            "Finish the current image safely, then unlock drawing and saving."
        ),
        "stop": "Stop",
        "retry_current": "Retry current image",
        "skip_current": "Skip current image",
        "review_now": "Review now",
        "review_later": "Review later",
        "phase_loading": "Preparing the image",
        "phase_inferencing": "Running inference",
        "phase_committing": "Saving safely",
        "phase_presenting": "Saved result displayed",
        "phase_waiting_error": "Waiting for error resolution",
        "phase_paused": "Paused - editing enabled",
        "phase_pausing": (
            "Pausing safely after the current image is saved..."
        ),
        "phase_finished": "Run finished",
        "continue": "Continue",
        "paused_remaining": (
            "Paused - editing enabled ({remaining:.1f} s remaining)"
        ),
        "nothing_to_process": "Nothing to process",
        "run_result": "Run result: {status}",
        "close": "Close",
        "start_continuous": "Start continuous auto-labeling",
        "dirty_current_image": "The current image has unsaved changes.",
        "continue_continuous": "Continue continuous auto-labeling",
        "stop_continuous": "Stop continuous auto-labeling",
        "cannot_continue": "Cannot continue continuous auto-labeling",
        "cannot_start": "Cannot start continuous auto-labeling",
        "fast_tooltip": (
            "Fast uses a 0.0 s delay and does not switch the main canvas "
            "for each image."
        ),
        "already_running": "Continuous auto-labeling is already running.",
        "legacy_no_audit": (
            "Legacy Batch: this model does not support Fast or Visible "
            "continuous auto-labeling. Legacy Batch does not create a "
            "unified audit queue."
        ),
        "sequence_unavailable_no_fallback": (
            "Fast/Visible continuous auto-labeling is unavailable. "
            "Legacy Batch was not started automatically."
        ),
        "audit_title": "Continuous auto-labeling audit",
        "previous": "Previous",
        "next_pending": "Next pending review",
        "mark_needs_fix": "Mark needs fix",
        "save_approve_next": "Save, approve, and next",
        "approve_next": "Approve and next",
        "finish_audit": "Finish audit",
        "audit_identity": (
            "{filename}  |  image_id={image_id}  |  #{sequence}"
        ),
        "audit_status": "Review status: {status}  |  dirty={dirty}",
        "audit_detail": (
            "staged={staged}  |  source={source}  |  "
            "target_count={target_count}  |  zero_target={zero_target}"
        ),
        "audit_summary": (
            "Session approved {staged_approved}  |  source approved "
            "{source_approved}  |  needs fix {needs_fix}  |  pending "
            "{pending}  |  stale {stale}"
        ),
        "audit_action_failed": "Audit action failed",
        "audit_unsaved_title": "Unsaved annotation",
        "audit_unsaved_prompt": (
            "Save or discard the current changes before switching review images."
        ),
        "status_not_applicable": "Not applicable",
        "status_pending": "Pending",
        "status_staged_approved": "Staged approved",
        "status_approved": "Approved",
        "status_needs_fix": "Needs fix",
        "status_stale": "Stale",
        "status_conflict": "Conflict",
        "status_none": "None",
        "status_prepared": "Prepared",
        "status_committed": "Committed",
        "status_abandoned": "Abandoned",
        "status_queued": "Queued",
        "status_running": "Running",
        "status_succeeded": "Succeeded",
        "status_skipped": "Skipped",
        "status_failed": "Failed",
        "status_paused": "Paused",
        "status_completed": "Completed",
        "status_partial": "Partial",
        "status_cancelled": "Cancelled",
        "status_recovery_required": "Recovery required",
        "status_archived": "Archived",
        "status_not_started": "Not started",
        "status_in_progress": "In progress",
        "yes": "Yes",
        "no": "No",
        "failure": "Failure",
        "reconciliation_pending": "Reconciliation pending",
        "integrity_refresh_required": (
            "Annotation integrity refresh required"
        ),
    }
)


_REVIEW_STATUS_KEYS = {
    "not_applicable": "status_not_applicable",
    "pending": "status_pending",
    "staged_approved": "status_staged_approved",
    "approved": "status_approved",
    "needs_fix": "status_needs_fix",
    "stale": "status_stale",
    "conflict": "status_conflict",
}

_COMMIT_STATUS_KEYS = {
    "not_applicable": "status_not_applicable",
    "none": "status_none",
    "prepared": "status_prepared",
    "committed": "status_committed",
    "abandoned": "status_abandoned",
    "pending": "status_pending",
    "conflict": "status_conflict",
}

_PROCESSING_STATUS_KEYS = {
    "PREPARED": "status_prepared",
    "RUNNING": "status_running",
    "PAUSED": "status_paused",
    "COMPLETED": "status_completed",
    "PARTIAL": "status_partial",
    "CANCELLED": "status_cancelled",
    "FAILED": "status_failed",
    "RECOVERY_REQUIRED": "status_recovery_required",
    "ARCHIVED": "status_archived",
}

_REVIEW_PROGRESS_KEYS = {
    "NOT_STARTED": "status_not_started",
    "IN_PROGRESS": "status_in_progress",
    "COMPLETED": "status_completed",
}


def auto_labeling_text_v1(key, **values):
    source = AUTO_LABELING_TEXT_V1[key]
    translated = QtCore.QCoreApplication.translate(
        TRANSLATION_CONTEXT_V1,
        source,
    )
    return translated.format(**values) if values else translated


def auto_labeling_status_text_v1(value, domain):
    maps = {
        "review": _REVIEW_STATUS_KEYS,
        "commit": _COMMIT_STATUS_KEYS,
        "processing": _PROCESSING_STATUS_KEYS,
        "review_progress": _REVIEW_PROGRESS_KEYS,
    }
    key = maps[domain].get(str(value))
    return auto_labeling_text_v1(key) if key else str(value)


def auto_labeling_boolean_text_v1(value):
    return auto_labeling_text_v1("yes" if bool(value) else "no")


def auto_labeling_error_category_v1(error_code):
    normalized = str(error_code or "").lower()
    if "reconcil" in normalized:
        return auto_labeling_text_v1("reconciliation_pending")
    if "recover" in normalized or "integrity" in normalized:
        return auto_labeling_text_v1("status_recovery_required")
    if "conflict" in normalized or "mismatch" in normalized:
        return auto_labeling_text_v1("status_conflict")
    return auto_labeling_text_v1("failure")
