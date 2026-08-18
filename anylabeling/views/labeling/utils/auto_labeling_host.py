"""Duck-typed image-centric host boundary for embedded auto labeling."""

import os
import re
import unicodedata
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

from .auto_labeling_audit import StagedAuditClientProtocolV1


_CREDENTIAL_KEY = re.compile(
    r"(api[_-]?key|token|secret|authorization|password)",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_active_host_context = None


class HostContextValidationError(ValueError):
    pass


@runtime_checkable
class AnnotationItemStoreProtocolV1(Protocol):
    def read_item(self, image_id):
        pass

    def recover_commit(
        self,
        image_id,
        current_document_digest,
        current_semantic_digest,
    ):
        pass


@runtime_checkable
class AutoLabelingHostContextProtocol(Protocol):
    project_id: str
    active_session_id: str
    image_records_by_path: dict
    annotation_item_store: AnnotationItemStoreProtocolV1
    model_fingerprint_provider: object
    annotation_commit_sink: object
    audit_client: object
    images_ready: bool
    workset_source: str
    annotation_session_lease: object
    session_purpose: str


@runtime_checkable
class StagedAuditHostContextProtocolV1(Protocol):
    staged_audit_client: StagedAuditClientProtocolV1


def _record_value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_credential_url(value):
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if not parsed.scheme or not parsed.netloc:
        return False
    if parsed.username is not None or parsed.password is not None:
        return True
    return any(
        _CREDENTIAL_KEY.search(key)
        for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
    )


def _validate_context_header(context):
    required = (
        "project_id",
        "active_session_id",
        "image_records_by_path",
        "annotation_item_store",
        "model_fingerprint_provider",
        "annotation_commit_sink",
        "audit_client",
        "images_ready",
        "workset_source",
        "annotation_session_lease",
        "session_purpose",
    )
    missing = [field for field in required if not hasattr(context, field)]
    if missing:
        raise HostContextValidationError(
            "host_context_missing_fields:" + ",".join(missing)
        )
    if type(context.project_id) is not str or not context.project_id.strip():
        raise HostContextValidationError("invalid_host_project_id")
    if (
        type(context.active_session_id) is not str
        or not context.active_session_id.strip()
    ):
        raise HostContextValidationError("invalid_host_session_id")
    if context.workset_source != "SESSION_WORKSET":
        raise HostContextValidationError("invalid_host_workset_source")
    if type(context.images_ready) is not bool:
        raise HostContextValidationError("invalid_host_images_ready")
    if context.session_purpose not in {"labeling", "review"}:
        raise HostContextValidationError("invalid_host_session_purpose")


def _validate_context_services(context):
    if not isinstance(
        context.annotation_item_store,
        AnnotationItemStoreProtocolV1,
    ):
        raise HostContextValidationError("invalid_annotation_item_store")
    if context.model_fingerprint_provider is not None and not callable(
        context.model_fingerprint_provider
    ):
        raise HostContextValidationError("invalid_model_fingerprint_provider")
    sink = context.annotation_commit_sink
    if sink is None or not all(
        callable(getattr(sink, name, None))
        for name in ("prepare", "checkpoint")
    ):
        raise HostContextValidationError("invalid_annotation_commit_sink")
    audit_client = context.audit_client
    if audit_client is not None and not isinstance(
        audit_client,
        StagedAuditClientProtocolV1,
    ):
        raise HostContextValidationError("invalid_staged_audit_client")
    lease = context.annotation_session_lease
    if (
        lease is None
        or getattr(lease, "resource", None) != "annotation_session"
        or getattr(lease, "mode", None) != "write"
        or not callable(getattr(lease, "release", None))
        or bool(getattr(lease, "released", False))
    ):
        raise HostContextValidationError("invalid_annotation_session_lease")


def _validate_record(key, record):
    image_id = _record_value(record, "image_id")
    image_path = _record_value(record, "canonical_session_image_path")
    label_path = _record_value(record, "canonical_session_label_path")
    source_image_digest = _record_value(record, "source_image_digest")
    source_label_digest = _record_value(
        record,
        "source_label_file_sha256",
    )
    sequence = _record_value(record, "manifest_sequence")
    if type(image_id) is not str or not image_id.strip():
        raise HostContextValidationError("invalid_host_record_image_id")
    if type(image_path) is not str or _canonical(image_path) != image_path:
        raise HostContextValidationError("invalid_host_record_image_path")
    if type(label_path) is not str or _canonical(label_path) != label_path:
        raise HostContextValidationError("invalid_host_record_label_path")
    if key != image_path:
        raise HostContextValidationError("host_record_mapping_key_mismatch")
    if not os.path.isfile(image_path) or os.path.islink(image_path):
        raise HostContextValidationError("host_record_image_not_regular")
    if os.path.lexists(label_path) and (
        not os.path.isfile(label_path) or os.path.islink(label_path)
    ):
        raise HostContextValidationError("host_record_label_not_regular")
    if type(sequence) is not int or sequence < 0:
        raise HostContextValidationError("invalid_host_manifest_sequence")
    if (
        type(source_image_digest) is not str
        or _SHA256.fullmatch(source_image_digest) is None
    ):
        raise HostContextValidationError("invalid_host_source_image_digest")
    if type(source_label_digest) is not str or (
        source_label_digest not in {"", "MISSING"}
        and _SHA256.fullmatch(source_label_digest) is None
    ):
        raise HostContextValidationError("invalid_host_source_label_digest")
    return (
        unicodedata.normalize("NFC", image_id),
        image_path,
        label_path,
        sequence,
    )


def validate_auto_labeling_host_context(context):
    _validate_context_header(context)
    _validate_context_services(context)
    records = context.image_records_by_path
    if type(records) is not dict:
        raise HostContextValidationError("invalid_host_image_records")
    image_ids = set()
    image_paths = set()
    label_paths = set()
    sequences = set()
    session_root = getattr(context, "_session_root", None)
    if (
        type(session_root) is not str
        or _canonical(session_root) != session_root
    ):
        raise HostContextValidationError("invalid_host_session_root")
    if not os.path.isdir(session_root) or os.path.islink(session_root):
        raise HostContextValidationError("invalid_host_session_root")
    for key, record in records.items():
        normalized_image_id, image_path, label_path, sequence = (
            _validate_record(key, record)
        )
        if normalized_image_id in image_ids:
            raise HostContextValidationError("duplicate_host_image_id")
        if image_path in image_paths:
            raise HostContextValidationError("duplicate_host_image_path")
        if label_path in label_paths:
            raise HostContextValidationError("duplicate_host_label_path")
        if sequence in sequences:
            raise HostContextValidationError(
                "duplicate_host_manifest_sequence"
            )
        try:
            image_contained = (
                os.path.commonpath((image_path, session_root)) == session_root
            )
            label_contained = (
                os.path.commonpath((label_path, session_root)) == session_root
            )
        except ValueError:
            image_contained = False
            label_contained = False
        if not image_contained or not label_contained:
            raise HostContextValidationError("host_record_path_escape")
        image_ids.add(normalized_image_id)
        image_paths.add(image_path)
        label_paths.add(label_path)
        sequences.add(sequence)
    return context


def validate_model_fingerprint_payload(payload):
    if type(payload) is not dict:
        raise HostContextValidationError("invalid_model_fingerprint")

    def inspect(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if _CREDENTIAL_KEY.search(str(key)):
                    raise HostContextValidationError(
                        f"credential_field_in_fingerprint:{path}.{key}"
                    )
                inspect(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")
        elif isinstance(value, str) and _is_credential_url(value):
            raise HostContextValidationError(
                f"credential_url_in_fingerprint:{path}"
            )

    inspect(payload, "$fingerprint")
    return payload


def resolve_model_fingerprint(context):
    context = validate_auto_labeling_host_context(context)
    provider = context.model_fingerprint_provider
    if provider is None:
        raise HostContextValidationError("model_fingerprint_unavailable")
    return validate_model_fingerprint_payload(provider())


def resolve_resume_sequence_spec_v1(_context):
    """Legacy API retained as an explicit tombstone."""

    raise HostContextValidationError("persistent_auto_labeling_resume_removed")


def set_auto_labeling_host_context(context):
    global _active_host_context
    _active_host_context = validate_auto_labeling_host_context(context)
    return _active_host_context


def clear_auto_labeling_host_context(context=None):
    global _active_host_context
    if context is None or context is _active_host_context:
        _active_host_context = None


def get_auto_labeling_host_context():
    return _active_host_context
