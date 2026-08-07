"""Frozen V5 auto-labeling contracts used by Phase 0 tests.

This module intentionally contains no runner, persistence, or UI behavior.
It provides strict validators and deterministic reference helpers for the
version-one data exchanged with the host application.
"""

import base64
import hashlib
import json
import math
import ntpath
import os
import posixpath
import re
import unicodedata
from datetime import datetime, timezone
from uuid import UUID


HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
VERSIONED_DIGEST_RE = re.compile(r"^(aldoc1|alsem1):[0-9a-f]{64}$")
WORKSET_DIGEST_RE = re.compile(r"^alworkset1:[0-9a-f]{64}$")

WORKSET_SOURCES_V1 = frozenset(
    {"SESSION_WORKSET", "CURRENT_FILE_LIST_SNAPSHOT"}
)
WORKSET_RANGES_V1 = frozenset({"ALL_IMAGES", "CURRENT_TO_END"})
IMAGE_INPUT_SOURCES_V1 = frozenset(
    {"SESSION_MANIFEST", "STANDALONE_FILE_LIST"}
)

SEMANTIC_EXCLUDED_TOP_LEVEL_FIELDS_V1 = frozenset(
    {
        "version",
        "imagePath",
        "imageData",
        "imageHeight",
        "imageWidth",
    }
)

ANNOTATION_PRESENCE_MISSING = "MISSING"
ANNOTATION_PRESENCE_VALID_EMPTY = "VALID_EMPTY"
ANNOTATION_PRESENCE_VALID_NONEMPTY = "VALID_NONEMPTY"
ANNOTATION_PRESENCE_INVALID = "INVALID"


WORKSET_SNAPSHOT_V1_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorksetSnapshotV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "workset_schema_version",
        "source",
        "search_filter_applied",
        "created_at",
        "current_anchor_image_id",
        "entries",
    ],
    "properties": {
        "workset_schema_version": {"const": 1},
        "source": {"enum": sorted(WORKSET_SOURCES_V1)},
        "search_filter_applied": {"type": "boolean"},
        "created_at": {"type": "string", "format": "date-time"},
        "current_anchor_image_id": {"type": ["string", "null"]},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sequence",
                    "image_id",
                    "canonical_image_path",
                    "canonical_label_path",
                    "expected_image_sha256",
                    "manifest_sequence",
                ],
                "properties": {
                    "sequence": {"type": "integer", "minimum": 0},
                    "image_id": {"type": "string", "minLength": 1},
                    "canonical_image_path": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "canonical_label_path": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "expected_image_sha256": {
                        "type": ["string", "null"],
                        "pattern": "^[0-9a-f]{64}$",
                    },
                    "manifest_sequence": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                    },
                },
            },
        },
    },
}


IMAGE_INPUT_SNAPSHOT_V1_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ImageInputSnapshotV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "snapshot_schema_version",
        "image_id",
        "requested_path",
        "canonical_path",
        "expected_sha256",
        "actual_sha256",
        "file_size",
        "mtime_ns",
        "decoded_width",
        "decoded_height",
        "pixel_format",
        "exif_orientation",
        "source",
    ],
    "properties": {
        "snapshot_schema_version": {"const": 1},
        "image_id": {"type": "string", "minLength": 1},
        "requested_path": {"type": "string", "minLength": 1},
        "canonical_path": {"type": "string", "minLength": 1},
        "expected_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "actual_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "file_size": {"type": "integer", "minimum": 0},
        "mtime_ns": {"type": "integer", "minimum": 0},
        "decoded_width": {"type": "integer", "minimum": 1},
        "decoded_height": {"type": "integer", "minimum": 1},
        "pixel_format": {"const": "RGB888"},
        "exif_orientation": {"enum": [1, None]},
        "source": {"enum": sorted(IMAGE_INPUT_SOURCES_V1)},
    },
}


ANNOTATION_COMMIT_EVENT_V1_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AnnotationCommitEventV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_schema_version",
        "event_id",
        "project_id",
        "session_id",
        "run_id",
        "image_id",
        "attempt_id",
        "writer_kind",
        "commit_scope",
        "mutation_mode",
        "document_digest",
        "semantic_digest",
        "source_image_digest",
        "base_item_revision",
        "created_at",
    ],
    "properties": {
        "event_schema_version": {"const": 1},
        "event_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "project_id": {"type": ["string", "null"]},
        "session_id": {"type": ["string", "null"]},
        "run_id": {"type": ["string", "null"]},
        "image_id": {"type": "string", "minLength": 1},
        "attempt_id": {"type": ["string", "null"]},
        "writer_kind": {
            "enum": [
                "CONTINUOUS",
                "MANUAL_SAVE",
                "AUTO_SAVE",
                "DELETE_LABEL",
                "SESSION_SOURCE_SYNC",
            ]
        },
        "commit_scope": {"enum": ["STAGED", "SOURCE"]},
        "mutation_mode": {"enum": ["NOTIFY_ONLY", "APPLY_MANUAL_REVISION"]},
        "document_digest": {
            "type": "string",
            "pattern": "^(aldoc1:[0-9a-f]{64}|MISSING)$",
        },
        "semantic_digest": {
            "type": "string",
            "pattern": "^(alsem1:[0-9a-f]{64}|MISSING)$",
        },
        "source_image_digest": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "base_item_revision": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "created_at": {"type": "string", "format": "date-time"},
    },
}


ANNOTATION_CANONICALIZATION_V1 = {
    "document_digest_schema_version": 1,
    "semantic_digest_schema_version": 1,
    "document_prefix": "ALDOC1\\0",
    "semantic_prefix": "ALSEM1\\0",
    "semantic_excluded_top_level_fields": sorted(
        SEMANTIC_EXCLUDED_TOP_LEVEL_FIELDS_V1
    ),
    "shape_order_significant": True,
    "unicode_normalization": "NFC",
    "path_fields": ["imagePath"],
}


class ContractValidationError(ValueError):
    """Raised when a value violates a frozen V5 contract."""

    def __init__(self, code, path="$", detail=""):
        self.code = str(code)
        self.path = str(path)
        self.detail = str(detail)
        message = f"{self.code}:{self.path}"
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


def _fail(code, path="$", detail=""):
    raise ContractValidationError(code, path, detail)


def _require_mapping(value, path):
    if type(value) is not dict:
        _fail("expected_object", path)
    return value


def _require_exact_fields(value, required, path):
    required = set(required)
    actual = set(value)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing:
        _fail("missing_fields", path, ",".join(missing))
    if unexpected:
        _fail("unexpected_fields", path, ",".join(unexpected))


def _require_nonempty_string(value, path):
    if type(value) is not str or not value.strip():
        _fail("expected_nonempty_string", path)
    return value


def _require_nullable_string(value, path):
    if value is not None:
        _require_nonempty_string(value, path)
    return value


def _require_integer(value, path, minimum=None, positive=False):
    if type(value) is not int:
        _fail("expected_integer", path)
    if positive and value <= 0:
        _fail("expected_positive_integer", path)
    if minimum is not None and value < minimum:
        _fail("integer_below_minimum", path, str(minimum))
    return value


def _require_number(value, path, minimum=None, maximum=None):
    if type(value) not in {int, float}:
        _fail("expected_number", path)
    if not math.isfinite(value):
        _fail("non_finite_number", path)
    if minimum is not None and value < minimum:
        _fail("number_below_minimum", path, str(minimum))
    if maximum is not None and value > maximum:
        _fail("number_above_maximum", path, str(maximum))
    return value


def _require_enum(value, allowed, path):
    if value not in allowed:
        _fail("invalid_enum", path, repr(value))
    return value


def _require_sha256(value, path, nullable=False):
    if nullable and value is None:
        return value
    if type(value) is not str or HEX_64_RE.fullmatch(value) is None:
        _fail("invalid_sha256", path)
    return value


def _require_versioned_digest(value, kind, path, allow_missing=False):
    if allow_missing and value == "MISSING":
        return value
    prefix = "aldoc1:" if kind == "document" else "alsem1:"
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or VERSIONED_DIGEST_RE.fullmatch(value) is None
    ):
        _fail(f"invalid_{kind}_digest", path)
    return value


def _require_uuid(value, path):
    _require_nonempty_string(value, path)
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError):
        _fail("invalid_uuid", path)
    if str(parsed) != value.lower():
        _fail("noncanonical_uuid", path)
    return value


def _require_utc_timestamp(value, path):
    _require_nonempty_string(value, path)
    if not value.endswith("Z"):
        _fail("timestamp_not_utc", path)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail("invalid_timestamp", path)
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail("timestamp_not_utc", path)
    return value


def canonical_path_identity(path):
    """Return the V5 realpath/normcase identity for a local path."""

    _require_nonempty_string(path, "path")
    if not os.path.isabs(path):
        _fail("path_not_absolute", "path")
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _require_absolute_path(value, path, canonical=False):
    _require_nonempty_string(value, path)
    if not os.path.isabs(value):
        _fail("path_not_absolute", path)
    if canonical and value != canonical_path_identity(value):
        _fail("path_not_canonical", path)
    return value


def validate_workset_snapshot_v1(snapshot):
    """Validate and return a WorksetSnapshotV1 mapping."""

    snapshot = _require_mapping(snapshot, "$workset")
    required = WORKSET_SNAPSHOT_V1_SCHEMA["required"]
    _require_exact_fields(snapshot, required, "$workset")
    if snapshot["workset_schema_version"] != 1:
        _fail("unsupported_schema_version", "workset_schema_version")
    source = _require_enum(snapshot["source"], WORKSET_SOURCES_V1, "source")
    if type(snapshot["search_filter_applied"]) is not bool:
        _fail("expected_boolean", "search_filter_applied")
    if source == "SESSION_WORKSET" and snapshot["search_filter_applied"]:
        _fail("session_workset_cannot_apply_search", "search_filter_applied")
    _require_utc_timestamp(snapshot["created_at"], "created_at")
    _require_nullable_string(
        snapshot["current_anchor_image_id"], "current_anchor_image_id"
    )
    entries = snapshot["entries"]
    if type(entries) is not list:
        _fail("expected_array", "entries")
    required_entry = WORKSET_SNAPSHOT_V1_SCHEMA["properties"]["entries"][
        "items"
    ]["required"]
    seen_sequences = set()
    seen_image_ids = set()
    seen_image_paths = set()
    seen_label_paths = set()
    previous_sequence = -1
    for index, entry in enumerate(entries):
        item_path = f"entries[{index}]"
        entry = _require_mapping(entry, item_path)
        _require_exact_fields(entry, required_entry, item_path)
        sequence = _require_integer(
            entry["sequence"], f"{item_path}.sequence", minimum=0
        )
        if sequence <= previous_sequence:
            _fail("sequence_not_strictly_increasing", item_path)
        previous_sequence = sequence
        if sequence in seen_sequences:
            _fail("duplicate_sequence", item_path)
        seen_sequences.add(sequence)
        image_id = _require_nonempty_string(
            entry["image_id"], f"{item_path}.image_id"
        )
        if image_id in seen_image_ids:
            _fail("duplicate_image_id", item_path)
        seen_image_ids.add(image_id)
        image_path = _require_absolute_path(
            entry["canonical_image_path"],
            f"{item_path}.canonical_image_path",
            canonical=True,
        )
        label_path = _require_absolute_path(
            entry["canonical_label_path"],
            f"{item_path}.canonical_label_path",
            canonical=True,
        )
        if image_path in seen_image_paths:
            _fail("duplicate_image_path", item_path)
        if label_path in seen_label_paths:
            _fail("duplicate_label_path", item_path)
        seen_image_paths.add(image_path)
        seen_label_paths.add(label_path)
        _require_sha256(
            entry["expected_image_sha256"],
            f"{item_path}.expected_image_sha256",
            nullable=True,
        )
        manifest_sequence = entry["manifest_sequence"]
        if manifest_sequence is not None:
            _require_integer(
                manifest_sequence,
                f"{item_path}.manifest_sequence",
                minimum=0,
            )
    return snapshot


def create_session_workset_snapshot_v1(
    records,
    session_root,
    created_at,
    current_anchor_image_id=None,
):
    """Build a Session workset only from manifest record order.

    Search text and visible widget rows are deliberately absent from this API.
    """

    if type(records) is not list:
        _fail("expected_array", "records")
    root = canonical_path_identity(str(session_root))
    entries = []
    for manifest_sequence, record in enumerate(records):
        record = _require_mapping(record, f"records[{manifest_sequence}]")
        if record.get("image_copy_succeeded") is not True:
            continue
        image_id = _require_nonempty_string(
            record.get("image_id"),
            f"records[{manifest_sequence}].image_id",
        )
        image_relpath = _require_nonempty_string(
            record.get("session_image_path"),
            f"records[{manifest_sequence}].session_image_path",
        )
        label_relpath = _require_nonempty_string(
            record.get("session_label_path"),
            f"records[{manifest_sequence}].session_label_path",
        )
        entries.append(
            {
                "sequence": len(entries),
                "image_id": image_id,
                "canonical_image_path": canonical_path_identity(
                    os.path.join(root, image_relpath)
                ),
                "canonical_label_path": canonical_path_identity(
                    os.path.join(root, label_relpath)
                ),
                "expected_image_sha256": record.get("source_image_sha256")
                or None,
                "manifest_sequence": manifest_sequence,
            }
        )
    snapshot = {
        "workset_schema_version": 1,
        "source": "SESSION_WORKSET",
        "search_filter_applied": False,
        "created_at": created_at,
        "current_anchor_image_id": current_anchor_image_id,
        "entries": entries,
    }
    return validate_workset_snapshot_v1(snapshot)


def select_workset_entries_v1(snapshot, range_mode):
    """Select ALL_IMAGES or CURRENT_TO_END from a frozen workset."""

    validate_workset_snapshot_v1(snapshot)
    _require_enum(range_mode, WORKSET_RANGES_V1, "range")
    entries = list(snapshot["entries"])
    if range_mode == "ALL_IMAGES":
        return entries
    anchor = snapshot["current_anchor_image_id"]
    if not anchor:
        _fail("current_anchor_missing", "current_anchor_image_id")
    matches = [
        index
        for index, entry in enumerate(entries)
        if entry["image_id"] == anchor
    ]
    if len(matches) != 1:
        _fail("current_anchor_not_unique", "current_anchor_image_id")
    return entries[matches[0] :]


def validate_image_input_snapshot_v1(snapshot):
    """Validate and return an ImageInputSnapshotV1 mapping."""

    snapshot = _require_mapping(snapshot, "$image_input")
    required = IMAGE_INPUT_SNAPSHOT_V1_SCHEMA["required"]
    _require_exact_fields(snapshot, required, "$image_input")
    if snapshot["snapshot_schema_version"] != 1:
        _fail("unsupported_schema_version", "snapshot_schema_version")
    _require_nonempty_string(snapshot["image_id"], "image_id")
    _require_absolute_path(snapshot["requested_path"], "requested_path")
    _require_absolute_path(
        snapshot["canonical_path"], "canonical_path", canonical=True
    )
    if snapshot["canonical_path"] != canonical_path_identity(
        snapshot["requested_path"]
    ):
        _fail("canonical_path_mismatch", "canonical_path")
    _require_sha256(
        snapshot["expected_sha256"], "expected_sha256", nullable=True
    )
    _require_sha256(snapshot["actual_sha256"], "actual_sha256")
    if (
        snapshot["expected_sha256"] is not None
        and snapshot["expected_sha256"] != snapshot["actual_sha256"]
    ):
        _fail("conflict_image_changed", "actual_sha256")
    _require_integer(snapshot["file_size"], "file_size", minimum=0)
    _require_integer(snapshot["mtime_ns"], "mtime_ns", minimum=0)
    _require_integer(snapshot["decoded_width"], "decoded_width", positive=True)
    _require_integer(
        snapshot["decoded_height"], "decoded_height", positive=True
    )
    if snapshot["pixel_format"] != "RGB888":
        _fail("invalid_pixel_format", "pixel_format")
    if snapshot["exif_orientation"] not in {None, 1}:
        _fail("exif_not_normalized", "exif_orientation")
    _require_enum(snapshot["source"], IMAGE_INPUT_SOURCES_V1, "source")
    return snapshot


def _validate_shape_structure_v1(shape, path):
    if type(shape) is not dict:
        _fail("invalid_shape_structure", path)
    if type(shape.get("label")) is not str:
        _fail("invalid_shape_structure", f"{path}.label")
    points = shape.get("points")
    if type(points) is not list or not points:
        _fail("invalid_shape_structure", f"{path}.points")
    for point_index, point in enumerate(points):
        if type(point) is not list or len(point) != 2:
            _fail("invalid_shape_structure", f"{path}.points[{point_index}]")
        for coordinate_index, coordinate in enumerate(point):
            coordinate_path = (
                f"{path}.points[{point_index}][{coordinate_index}]"
            )
            if type(coordinate) not in {int, float}:
                _fail("invalid_shape_structure", coordinate_path)
            if not math.isfinite(coordinate):
                _fail("non_finite_number", coordinate_path)
    shape_type = shape.get("shape_type", "polygon")
    if shape_type not in {
        "polygon",
        "rectangle",
        "rotation",
        "point",
        "line",
        "circle",
        "linestrip",
    }:
        _fail("invalid_shape_structure", f"{path}.shape_type")
    if "flags" in shape and type(shape["flags"]) is not dict:
        _fail("invalid_shape_structure", f"{path}.flags")


def _has_valid_shape_structure_v1(shape):
    try:
        _validate_shape_structure_v1(shape, "$shape")
    except ContractValidationError:
        return False
    return True


def classify_annotation_document_v1(value):
    """Classify missing, valid empty, valid nonempty, and invalid labels."""

    if value is None:
        return ANNOTATION_PRESENCE_MISSING
    if type(value) in {str, bytes, bytearray}:
        try:
            value = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return ANNOTATION_PRESENCE_INVALID
    if type(value) is not dict or type(value.get("shapes")) is not list:
        return ANNOTATION_PRESENCE_INVALID
    if not all(
        _has_valid_shape_structure_v1(shape) for shape in value["shapes"]
    ):
        return ANNOTATION_PRESENCE_INVALID
    if value["shapes"]:
        return ANNOTATION_PRESENCE_VALID_NONEMPTY
    return ANNOTATION_PRESENCE_VALID_EMPTY


def _normalize_image_path(value, path):
    if type(value) is not str:
        _fail("image_path_not_string", path)
    value = unicodedata.normalize("NFC", value)
    if ntpath.isabs(value) or posixpath.isabs(value):
        _fail("image_path_absolute", path)
    drive, _tail = ntpath.splitdrive(value)
    if drive:
        _fail("image_path_absolute", path)
    normalized_parts = []
    for part in value.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                _fail("image_path_escape", path)
            normalized_parts.pop()
            continue
        normalized_parts.append(part)
    return "/".join(normalized_parts) or "."


def _canonicalize_value(value, path, top_level=False):
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail("non_finite_number", path)
        if value == 0.0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if type(value) is str:
        return unicodedata.normalize("NFC", value)
    if type(value) is list:
        return [
            _canonicalize_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        normalized = {}
        for original_key, item in value.items():
            if type(original_key) is not str:
                _fail("non_string_key", path)
            key = unicodedata.normalize("NFC", original_key)
            if key in normalized:
                _fail("normalized_key_collision", path, key)
            item_path = f"{path}.{key}"
            if top_level and key == "imagePath":
                normalized[key] = _normalize_image_path(item, item_path)
            elif top_level and key == "imageData" and item is not None:
                if type(item) is not str:
                    _fail("image_data_not_base64", item_path)
                try:
                    raw = base64.b64decode(item, validate=True)
                except (ValueError, TypeError):
                    _fail("image_data_not_base64", item_path)
                normalized[key] = {
                    "__image_data_sha256__": hashlib.sha256(raw).hexdigest()
                }
            else:
                normalized[key] = _canonicalize_value(item, item_path)
        return normalized
    _fail("non_json_type", path, type(value).__name__)


def canonicalize_annotation_document_v1(document, semantic=False):
    """Return the canonical tree for a LabelMe document."""

    if classify_annotation_document_v1(document) in {
        ANNOTATION_PRESENCE_MISSING,
        ANNOTATION_PRESENCE_INVALID,
    }:
        if (
            type(document) is not dict
            or type(document.get("shapes")) is not list
        ):
            _fail("invalid_annotation_document", "$document")
        for index, shape in enumerate(document["shapes"]):
            _validate_shape_structure_v1(shape, f"$document.shapes[{index}]")
        _fail("invalid_annotation_document", "$document")
    source = document
    if semantic:
        source = {
            key: value
            for key, value in document.items()
            if unicodedata.normalize("NFC", key)
            not in SEMANTIC_EXCLUDED_TOP_LEVEL_FIELDS_V1
        }
    return _canonicalize_value(source, "$document", top_level=True)


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_document_digest_v1(document):
    tree = canonicalize_annotation_document_v1(document, semantic=False)
    digest = hashlib.sha256(b"ALDOC1\0" + _canonical_json_bytes(tree))
    return "aldoc1:" + digest.hexdigest()


def semantic_annotation_digest_v1(document):
    tree = canonicalize_annotation_document_v1(document, semantic=True)
    digest = hashlib.sha256(b"ALSEM1\0" + _canonical_json_bytes(tree))
    return "alsem1:" + digest.hexdigest()


def raw_file_sha256_v1(raw_bytes):
    if type(raw_bytes) not in {bytes, bytearray}:
        _fail("raw_digest_requires_bytes", "$raw")
    return hashlib.sha256(bytes(raw_bytes)).hexdigest()


_COMMIT_EVENT_ID_FIELDS = (
    "project_id",
    "session_id",
    "run_id",
    "image_id",
    "attempt_id",
    "writer_kind",
    "commit_scope",
    "mutation_mode",
    "document_digest",
    "semantic_digest",
)


def annotation_commit_event_id_v1(event):
    payload = {field: event.get(field) for field in _COMMIT_EVENT_ID_FIELDS}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def validate_annotation_commit_event_v1(event):
    """Validate and return an AnnotationCommitEventV1 mapping."""

    event = _require_mapping(event, "$commit_event")
    required = ANNOTATION_COMMIT_EVENT_V1_SCHEMA["required"]
    _require_exact_fields(event, required, "$commit_event")
    if event["event_schema_version"] != 1:
        _fail("unsupported_schema_version", "event_schema_version")
    _require_sha256(event["event_id"], "event_id")
    for field in ("project_id", "session_id", "run_id", "attempt_id"):
        _require_nullable_string(event[field], field)
    _require_nonempty_string(event["image_id"], "image_id")
    _require_enum(
        event["writer_kind"],
        {
            "CONTINUOUS",
            "MANUAL_SAVE",
            "AUTO_SAVE",
            "DELETE_LABEL",
            "SESSION_SOURCE_SYNC",
        },
        "writer_kind",
    )
    _require_enum(event["commit_scope"], {"STAGED", "SOURCE"}, "commit_scope")
    _require_enum(
        event["mutation_mode"],
        {"NOTIFY_ONLY", "APPLY_MANUAL_REVISION"},
        "mutation_mode",
    )
    _require_versioned_digest(
        event["document_digest"],
        "document",
        "document_digest",
        allow_missing=True,
    )
    _require_versioned_digest(
        event["semantic_digest"],
        "semantic",
        "semantic_digest",
        allow_missing=True,
    )
    _require_sha256(
        event["source_image_digest"],
        "source_image_digest",
        nullable=True,
    )
    if event["base_item_revision"] is not None:
        _require_integer(
            event["base_item_revision"],
            "base_item_revision",
            minimum=0,
        )
    _require_utc_timestamp(event["created_at"], "created_at")
    expected_id = annotation_commit_event_id_v1(event)
    if event["event_id"] != expected_id:
        _fail("event_id_mismatch", "event_id")
    return event


def validate_workset_digest_v1(value):
    if type(value) is not str or WORKSET_DIGEST_RE.fullmatch(value) is None:
        _fail("invalid_workset_digest", "$workset_digest")
    return value


def validate_document_digest_v1(value, allow_missing=False):
    return _require_versioned_digest(
        value, "document", "$document_digest", allow_missing=allow_missing
    )


def validate_semantic_digest_v1(value, allow_missing=False):
    return _require_versioned_digest(
        value, "semantic", "$semantic_digest", allow_missing=allow_missing
    )


def validate_raw_sha256_v1(value):
    return _require_sha256(value, "$raw_sha256")
