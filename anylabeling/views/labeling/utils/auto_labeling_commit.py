"""Pure Phase 1 label composition and atomic commit helpers."""

import base64
import copy
import io
import json
import math
import ntpath
import os
import posixpath
import stat
import tempfile
import threading
from dataclasses import dataclass

import PIL.Image
from PyQt5 import QtGui

from anylabeling.app_info import __version__

from .auto_labeling_contracts import (
    ANNOTATION_PRESENCE_INVALID,
    ANNOTATION_PRESENCE_MISSING,
    ANNOTATION_PRESENCE_VALID_EMPTY,
    ANNOTATION_PRESENCE_VALID_NONEMPTY,
    ContractValidationError,
    canonical_document_digest_v1,
    canonical_path_identity,
    classify_annotation_document_v1,
    raw_file_sha256_v1,
    semantic_annotation_digest_v1,
    validate_document_digest_v1,
    validate_image_input_snapshot_v1,
)
from .image import decode_image_for_labeling


WRITE_POLICIES_V1 = frozenset(
    {
        "SKIP_EXISTING",
        "FORCE_REPLACE",
    }
)
MAX_GROUP_ID_V1 = (2**31) - 1
_LABEL_BASE_FIELDS = frozenset(
    {
        "version",
        "flags",
        "shapes",
        "imagePath",
        "imageData",
        "imageHeight",
        "imageWidth",
    }
)


class AutoLabelingCommitError(RuntimeError):
    """Base error carrying a stable Phase 1 code."""

    def __init__(self, code, detail=""):
        self.code = code
        self.detail = str(detail)
        message = code if not self.detail else f"{code}: {self.detail}"
        super().__init__(message)


class LabelConflictError(AutoLabelingCommitError):
    """Raised when an existing file or output identity cannot be overwritten."""


class LabelWriteError(AutoLabelingCommitError):
    """Raised when an atomic label write cannot be completed and verified."""


class PredictionOutcomeError(AutoLabelingCommitError):
    """Raised when a prediction is not an explicit successful outcome."""


class ImageInputSnapshotError(AutoLabelingCommitError):
    """Raised when immutable image input cannot be created safely."""


@dataclass(frozen=True)
class ExistingLabelResolution:
    presence: str
    document: object
    raw_bytes: object
    raw_file_sha256: str
    document_digest: str
    semantic_digest: str
    error_detail: object = None


@dataclass(frozen=True)
class NormalizedPredictionOutcomeV1:
    status: str
    shapes: tuple
    replace: object
    description: str
    zero_target: bool
    error_code: object


@dataclass(frozen=True)
class CompositionResultV1:
    action: str
    effective_policy: str
    document: object
    document_digest: str
    semantic_digest: str
    semantic_change: bool
    target_count: int
    zero_target: bool
    skip_reason: object = None


@dataclass(frozen=True)
class AtomicWriteResultV1:
    document: dict
    raw_file_sha256: str
    document_digest: str
    semantic_digest: str


@dataclass(frozen=True)
class ImageInputSnapshotV1:
    image_id: str
    requested_path: str
    canonical_path: str
    expected_sha256: object
    actual_sha256: str
    file_size: int
    mtime_ns: int
    decoded_width: int
    decoded_height: int
    pixel_format: str
    exif_orientation: object
    source: str
    rgb888: bytes

    def contract_fields(self):
        """Return the persisted metadata fields without caching pixel data."""

        return {
            "snapshot_schema_version": 1,
            "image_id": self.image_id,
            "requested_path": self.requested_path,
            "canonical_path": self.canonical_path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "file_size": self.file_size,
            "mtime_ns": self.mtime_ns,
            "decoded_width": self.decoded_width,
            "decoded_height": self.decoded_height,
            "pixel_format": self.pixel_format,
            "exif_orientation": self.exif_orientation,
            "source": self.source,
        }

    def to_qimage(self):
        """Build a detached QImage from the frozen RGB888 bytes."""

        image = QtGui.QImage(
            self.rgb888,
            self.decoded_width,
            self.decoded_height,
            self.decoded_width * 3,
            QtGui.QImage.Format_RGB888,
        )
        return image.copy()

    def to_rgb_array(self):
        """Build a writable model array without re-reading the image path."""

        import numpy as np

        return (
            np.frombuffer(self.rgb888, dtype=np.uint8)
            .reshape(self.decoded_height, self.decoded_width, 3)
            .copy()
        )


@dataclass(frozen=True)
class ImageCommitResultV1:
    composition: CompositionResultV1
    write_result: object
    item: dict
    event: object


@dataclass(frozen=True)
class CommitRecoveryResultV1:
    action: str
    item: dict
    current_document_digest: str


_label_mutexes = {}
_label_mutexes_guard = threading.Lock()


def _label_mutex(path):
    identity = canonical_path_identity(path)
    with _label_mutexes_guard:
        return _label_mutexes.setdefault(identity, threading.RLock())


def _fail_contract(code, path="$", detail=""):
    suffix = f":{detail}" if detail else ""
    raise ContractValidationError(f"{code}:{path}{suffix}")


def _plain_json_value(value, path="$"):
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail_contract("non_finite_number", path)
        return value
    if type(value) in {list, tuple}:
        return [
            _plain_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                _fail_contract("non_string_key", path)
            result[key] = _plain_json_value(item, f"{path}.{key}")
        return result
    _fail_contract("non_json_type", path, type(value).__name__)


def validate_label_document(document):
    """Return an independent plain-JSON LabelMe document or raise."""

    normalized = _plain_json_value(document, "$document")
    if type(normalized) is not dict:
        _fail_contract("invalid_annotation_document", "$document")
    if type(normalized.get("shapes")) is not list:
        _fail_contract("invalid_annotation_document", "$document.shapes")
    if "flags" in normalized and type(normalized["flags"]) is not dict:
        _fail_contract("invalid_annotation_flags", "$document.flags")
    if (
        "description" in normalized
        and type(normalized["description"]) is not str
    ):
        _fail_contract(
            "invalid_annotation_description", "$document.description"
        )
    if "version" in normalized and normalized["version"] is not None:
        if type(normalized["version"]) is not str:
            _fail_contract("invalid_annotation_version", "$document.version")
    for field in ("imageHeight", "imageWidth"):
        value = normalized.get(field)
        if value is not None and (type(value) is not int or value <= 0):
            _fail_contract("invalid_image_dimension", f"$document.{field}")
    canonical_document_digest_v1(normalized)
    semantic_annotation_digest_v1(normalized)
    return normalized


def resolve_existing_label(label_path):
    """Resolve a label into an explicit missing/valid/invalid state."""

    path = os.fspath(label_path)
    if not os.path.lexists(path):
        return ExistingLabelResolution(
            presence=ANNOTATION_PRESENCE_MISSING,
            document=None,
            raw_bytes=None,
            raw_file_sha256="MISSING",
            document_digest="MISSING",
            semantic_digest="MISSING",
        )
    if os.path.islink(path):
        raise LabelConflictError("conflict_output_symlink", path)
    try:
        file_stat = os.stat(path)
    except OSError as exc:
        raise LabelConflictError("conflict_label_unreadable", exc) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise LabelConflictError("conflict_output_not_regular_file", path)
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError as exc:
        raise LabelConflictError("conflict_label_unreadable", exc) from exc
    raw_digest = raw_file_sha256_v1(raw)
    try:
        parsed = json.loads(raw.decode("utf-8"))
        document = validate_label_document(parsed)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ContractValidationError,
    ) as exc:
        return ExistingLabelResolution(
            presence=ANNOTATION_PRESENCE_INVALID,
            document=None,
            raw_bytes=raw,
            raw_file_sha256=raw_digest,
            document_digest="INVALID",
            semantic_digest="INVALID",
            error_detail=str(exc),
        )
    presence = classify_annotation_document_v1(document)
    if presence not in {
        ANNOTATION_PRESENCE_VALID_EMPTY,
        ANNOTATION_PRESENCE_VALID_NONEMPTY,
    }:
        return ExistingLabelResolution(
            presence=ANNOTATION_PRESENCE_INVALID,
            document=None,
            raw_bytes=raw,
            raw_file_sha256=raw_digest,
            document_digest="INVALID",
            semantic_digest="INVALID",
            error_detail="invalid_annotation_document",
        )
    return ExistingLabelResolution(
        presence=presence,
        document=document,
        raw_bytes=raw,
        raw_file_sha256=raw_digest,
        document_digest=canonical_document_digest_v1(document),
        semantic_digest=semantic_annotation_digest_v1(document),
    )


def normalize_prediction_outcome_v1(outcome):
    """Normalize the Phase 1 commit input without treating None as success."""

    if type(outcome) is not dict:
        raise PredictionOutcomeError("invalid_prediction_outcome")
    required = {"status", "shapes", "replace", "description", "zero_target"}
    missing = sorted(required.difference(outcome))
    if missing:
        raise PredictionOutcomeError(
            "invalid_prediction_outcome", ",".join(missing)
        )
    status = outcome["status"]
    if status not in {"succeeded", "failed", "cancelled"}:
        raise PredictionOutcomeError("invalid_prediction_status", status)
    error_code = outcome.get("error_code")
    if status != "succeeded":
        if type(error_code) is not str or not error_code:
            raise PredictionOutcomeError("missing_prediction_error_code")
        return NormalizedPredictionOutcomeV1(
            status=status,
            shapes=(),
            replace=None,
            description="",
            zero_target=False,
            error_code=error_code,
        )
    if type(outcome["replace"]) is not bool:
        raise PredictionOutcomeError("invalid_prediction_replace")
    if type(outcome["description"]) is not str:
        raise PredictionOutcomeError("invalid_prediction_description")
    if type(outcome["zero_target"]) is not bool:
        raise PredictionOutcomeError("invalid_prediction_zero_target")
    if type(outcome["shapes"]) not in {list, tuple}:
        raise PredictionOutcomeError("invalid_prediction_shapes")
    shapes = _plain_json_value(outcome["shapes"], "$prediction.shapes")
    validate_label_document({"shapes": shapes})
    if outcome["zero_target"] != (len(shapes) == 0):
        raise PredictionOutcomeError("prediction_zero_target_mismatch")
    if error_code is not None:
        raise PredictionOutcomeError("successful_prediction_has_error")
    return NormalizedPredictionOutcomeV1(
        status=status,
        shapes=tuple(copy.deepcopy(shapes)),
        replace=outcome["replace"],
        description=outcome["description"],
        zero_target=outcome["zero_target"],
        error_code=None,
    )


def _validate_group_id(group_id):
    if group_id is None:
        return
    if type(group_id) is int:
        return
    if type(group_id) is str and group_id:
        return
    raise LabelConflictError("conflict_unsupported_group_id")


def _group_key(group_id):
    return (type(group_id).__name__, group_id)


def _validate_pose_groups(incoming_shapes, pose_config):
    if pose_config is None:
        return
    classes = pose_config.get("classes") if type(pose_config) is dict else None
    if type(classes) is not dict or not classes:
        raise LabelConflictError("invalid_pose_group_structure")
    grouped = {}
    for shape in incoming_shapes:
        group_id = shape.get("group_id")
        if group_id is not None:
            grouped.setdefault(_group_key(group_id), []).append(shape)
    for shapes in grouped.values():
        rectangles = [
            shape for shape in shapes if shape.get("shape_type") == "rectangle"
        ]
        if len(rectangles) != 1:
            raise LabelConflictError("invalid_pose_group_structure")
        class_name = rectangles[0].get("label")
        keypoints = classes.get(class_name)
        if type(keypoints) is not list:
            raise LabelConflictError("invalid_pose_group_structure")
        allowed = set(keypoints)
        seen = set()
        for shape in shapes:
            if shape is rectangles[0]:
                continue
            if shape.get("shape_type") != "point":
                raise LabelConflictError("invalid_pose_group_structure")
            label = shape.get("label")
            if label not in allowed or label in seen:
                raise LabelConflictError("invalid_pose_group_structure")
            if len(shape.get("points", [])) != 1:
                raise LabelConflictError("invalid_pose_group_structure")
            seen.add(label)


def remap_incoming_group_ids_v1(
    existing_shapes,
    incoming_shapes,
    pose_config=None,
    kie_linking_adapter=None,
):
    """Deep-copy and deterministically remap incoming Merge group IDs."""

    existing = _plain_json_value(existing_shapes, "$existing.shapes")
    incoming = _plain_json_value(incoming_shapes, "$incoming.shapes")
    for shape in existing + incoming:
        if type(shape) is not dict:
            raise LabelConflictError("invalid_shape_structure")
        _validate_group_id(shape.get("group_id"))
    _validate_pose_groups(incoming, pose_config)
    has_linking = any(bool(shape.get("kie_linking")) for shape in incoming)
    if has_linking and kie_linking_adapter is None:
        raise LabelConflictError("conflict_unsupported_kie_merge")
    used = {
        shape.get("group_id")
        for shape in existing
        if type(shape.get("group_id")) is int and shape.get("group_id") >= 0
    }
    next_group_id = max(used, default=-1) + 1
    mapping = {}
    for shape in incoming:
        group_id = shape.get("group_id")
        if group_id is None:
            continue
        key = _group_key(group_id)
        if key not in mapping:
            while next_group_id in used and next_group_id <= MAX_GROUP_ID_V1:
                next_group_id += 1
            if next_group_id > MAX_GROUP_ID_V1:
                raise LabelConflictError("conflict_group_id_exhausted")
            mapping[key] = next_group_id
            used.add(next_group_id)
            next_group_id += 1
        shape["group_id"] = mapping[key]
    if has_linking:
        incoming = kie_linking_adapter(copy.deepcopy(incoming), dict(mapping))
        incoming = _plain_json_value(incoming, "$incoming.shapes")
    return incoming


def _normalized_writer_image_path(label_path, image_path):
    value = os.fspath(image_path)
    if ntpath.isabs(value) or posixpath.isabs(value):
        value = os.path.relpath(
            value, os.path.dirname(os.path.abspath(label_path))
        )
    value = value.replace("\\", "/")
    normalized = posixpath.normpath(value)
    if normalized == ".." or normalized.startswith("../"):
        normalized = posixpath.basename(normalized)
    if normalized in {"", "."}:
        normalized = posixpath.basename(value)
    return normalized


def _encoded_image_data(image_data):
    if image_data is None:
        return None
    if type(image_data) in {bytes, bytearray}:
        return base64.b64encode(bytes(image_data)).decode("ascii")
    if type(image_data) is str:
        return image_data
    _fail_contract("invalid_image_data", "$imageData")


def _combined_description(existing, incoming, effective_policy):
    if not incoming:
        return existing
    if effective_policy == "FORCE_REPLACE":
        return incoming
    if not existing:
        return incoming
    if existing == incoming:
        return existing
    return f"{existing}\n\n{incoming}"


def compose_final_label_document(
    existing_label,
    prediction_outcome,
    write_policy,
    *,
    label_path,
    image_path,
    image_height,
    image_width,
    image_data=None,
    flags=None,
    other_data=None,
    version=__version__,
    pose_config=None,
    kie_linking_adapter=None,
):
    """Apply the frozen V5 write truth table without mutating inputs."""

    if not isinstance(existing_label, ExistingLabelResolution):
        raise TypeError("existing_label must be ExistingLabelResolution")
    if write_policy not in WRITE_POLICIES_V1:
        raise ValueError(f"unsupported write policy: {write_policy}")
    if existing_label.presence == ANNOTATION_PRESENCE_INVALID:
        raise LabelConflictError("invalid_existing_label")
    prediction = normalize_prediction_outcome_v1(prediction_outcome)
    if prediction.status != "succeeded":
        raise PredictionOutcomeError(prediction.error_code)
    existing_is_valid = existing_label.presence in {
        ANNOTATION_PRESENCE_VALID_EMPTY,
        ANNOTATION_PRESENCE_VALID_NONEMPTY,
    }
    if write_policy == "SKIP_EXISTING" and existing_is_valid:
        return CompositionResultV1(
            action="skipped",
            effective_policy="SKIP_EXISTING",
            document=copy.deepcopy(existing_label.document),
            document_digest=existing_label.document_digest,
            semantic_digest=existing_label.semantic_digest,
            semantic_change=False,
            target_count=0,
            zero_target=prediction.zero_target,
            skip_reason="existing_annotation",
        )
    effective_policy = "FORCE_REPLACE"
    existing_document = (
        validate_label_document(existing_label.document)
        if existing_is_valid
        else None
    )
    incoming_shapes = [copy.deepcopy(shape) for shape in prediction.shapes]
    for shape in incoming_shapes:
        _validate_group_id(shape.get("group_id"))
    final_shapes = incoming_shapes
    supplied_other = _plain_json_value(other_data or {}, "$other_data")
    if type(supplied_other) is not dict:
        _fail_contract("invalid_other_data", "$other_data")
    if any(key in _LABEL_BASE_FIELDS for key in supplied_other):
        _fail_contract("other_data_field_collision", "$other_data")
    preserved_other = {}
    if existing_document is not None:
        preserved_other = {
            key: copy.deepcopy(value)
            for key, value in existing_document.items()
            if key not in _LABEL_BASE_FIELDS
        }
    for key, value in supplied_other.items():
        preserved_other.setdefault(key, copy.deepcopy(value))
    existing_description = preserved_other.pop("description", "")
    if type(existing_description) is not str:
        _fail_contract(
            "invalid_annotation_description", "$document.description"
        )
    description = _combined_description(
        existing_description,
        prediction.description,
        effective_policy,
    )
    if existing_document is not None:
        final_flags = copy.deepcopy(existing_document.get("flags", {}))
    else:
        final_flags = _plain_json_value(flags or {}, "$flags")
    document = {
        "version": version,
        "flags": final_flags,
        "shapes": final_shapes,
        "imagePath": _normalized_writer_image_path(label_path, image_path),
        "imageData": _encoded_image_data(image_data),
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
    document.update(preserved_other)
    document["description"] = description
    document = validate_label_document(document)
    document_digest = canonical_document_digest_v1(document)
    semantic_digest = semantic_annotation_digest_v1(document)
    return CompositionResultV1(
        action="write",
        effective_policy=effective_policy,
        document=document,
        document_digest=document_digest,
        semantic_digest=semantic_digest,
        semantic_change=semantic_digest != existing_label.semantic_digest,
        target_count=len(prediction.shapes),
        zero_target=prediction.zero_target,
    )


def _within_root(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _same_file(first, second):
    if not os.path.exists(first) or not os.path.exists(second):
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _validate_output_location(label_path, requested_root, root):
    parent = os.path.dirname(label_path)
    if not os.path.isdir(parent):
        raise LabelConflictError("conflict_output_parent_missing", parent)
    if os.path.lexists(label_path) and os.path.islink(label_path):
        raise LabelConflictError("conflict_output_symlink", label_path)
    if root is not None and not _within_root(
        os.path.realpath(label_path), root
    ):
        raise LabelConflictError("conflict_output_path_escape", label_path)
    if requested_root is not None:
        relative = os.path.relpath(label_path, requested_root)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            raise LabelConflictError("conflict_output_path_escape", label_path)
        current = requested_root
        for part in relative.split(os.sep):
            current = os.path.join(current, part)
            if os.path.lexists(current) and os.path.islink(current):
                raise LabelConflictError("conflict_output_symlink", current)
    if os.path.exists(label_path) and not os.path.isfile(label_path):
        raise LabelConflictError(
            "conflict_output_not_regular_file", label_path
        )


def validate_output_paths_v1(
    entries,
    *,
    allowed_root=None,
    require_unique_basename=False,
    reserved_paths=(),
):
    """Reject basename, canonical, samefile, symlink, and root collisions."""

    if type(entries) not in {list, tuple}:
        raise TypeError("entries must be a sequence")
    root = None
    requested_root = None
    if allowed_root is not None:
        requested_root = os.path.abspath(os.fspath(allowed_root))
        root = os.path.realpath(requested_root)
        if not os.path.isdir(root):
            raise LabelConflictError("conflict_output_root_invalid", root)
    reserved = [os.path.abspath(os.fspath(path)) for path in reserved_paths]
    reserved_identities = {canonical_path_identity(path) for path in reserved}
    labels = []
    label_identities = set()
    basenames = set()
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            raise TypeError(f"entries[{index}] must be a mapping")
        image_path = os.path.abspath(os.fspath(entry["image_path"]))
        label_path = os.path.abspath(os.fspath(entry["label_path"]))
        _validate_output_location(label_path, requested_root, root)
        identity = canonical_path_identity(label_path)
        if identity in label_identities or identity in reserved_identities:
            raise LabelConflictError(
                "conflict_output_path_collision", label_path
            )
        if identity == canonical_path_identity(image_path):
            raise LabelConflictError(
                "conflict_output_image_collision", label_path
            )
        basename = os.path.normcase(os.path.basename(label_path))
        if require_unique_basename and basename in basenames:
            raise LabelConflictError(
                "conflict_output_basename_collision", basename
            )
        for prior in labels + reserved:
            if _same_file(label_path, prior):
                raise LabelConflictError(
                    "conflict_output_samefile_collision", label_path
                )
        if _same_file(label_path, image_path):
            raise LabelConflictError(
                "conflict_output_samefile_collision", label_path
            )
        labels.append(label_path)
        label_identities.add(identity)
        basenames.add(basename)
    return tuple(canonical_path_identity(path) for path in labels)


def atomic_write_label_document(
    label_path,
    document,
    *,
    pre_document_digest,
    allowed_root=None,
    image_source_path=None,
    reserved_paths=(),
    after_replace_hook=None,
):
    """Atomically replace and reread a label under a document-digest CAS."""

    path = os.path.abspath(os.fspath(label_path))
    writer_document = _plain_json_value(document, "$document")
    if type(writer_document) is not dict or "imagePath" not in writer_document:
        _fail_contract("invalid_annotation_document", "$document.imagePath")
    writer_document["imagePath"] = _normalized_writer_image_path(
        path, writer_document["imagePath"]
    )
    validated_document = validate_label_document(writer_document)
    validate_document_digest_v1(pre_document_digest, allow_missing=True)
    intended_digest = canonical_document_digest_v1(validated_document)
    intended_semantic_digest = semantic_annotation_digest_v1(
        validated_document
    )
    image_path = image_source_path or os.path.join(
        os.path.dirname(path), validated_document["imagePath"]
    )
    validate_output_paths_v1(
        [{"image_path": image_path, "label_path": path}],
        allowed_root=allowed_root or os.path.dirname(path),
        reserved_paths=reserved_paths,
    )
    temporary_path = None
    with _label_mutex(path):
        current = resolve_existing_label(path)
        if current.presence == ANNOTATION_PRESENCE_INVALID:
            raise LabelConflictError("invalid_existing_label")
        if current.document_digest != pre_document_digest:
            raise LabelConflictError("conflict_pre_document_digest")
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".auto-labeling-",
                suffix=".tmp",
                dir=os.path.dirname(path),
            )
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(
                    validated_document,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            latest = resolve_existing_label(path)
            if latest.presence == ANNOTATION_PRESENCE_INVALID:
                raise LabelConflictError("invalid_existing_label")
            if latest.document_digest != pre_document_digest:
                raise LabelConflictError("conflict_pre_document_digest")
            os.replace(temporary_path, path)
            temporary_path = None
        except (ContractValidationError, LabelConflictError):
            raise
        except Exception as exc:
            raise LabelWriteError("label_write_failed", exc) from exc
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        if after_replace_hook is not None:
            after_replace_hook()
        reread = resolve_existing_label(path)
        if reread.presence not in {
            ANNOTATION_PRESENCE_VALID_EMPTY,
            ANNOTATION_PRESENCE_VALID_NONEMPTY,
        }:
            raise LabelWriteError("post_write_verification_failed")
        if reread.document_digest != intended_digest:
            raise LabelWriteError("post_write_document_digest_mismatch")
        if reread.semantic_digest != intended_semantic_digest:
            raise LabelWriteError("post_write_semantic_digest_mismatch")
        return AtomicWriteResultV1(
            document=copy.deepcopy(reread.document),
            raw_file_sha256=reread.raw_file_sha256,
            document_digest=reread.document_digest,
            semantic_digest=reread.semantic_digest,
        )


def atomic_delete_label_document(
    label_path,
    *,
    pre_document_digest,
    allowed_root=None,
):
    """Delete one authoritative label under the shared per-path CAS mutex."""

    path = os.path.abspath(os.fspath(label_path))
    validate_document_digest_v1(pre_document_digest, allow_missing=True)
    if allowed_root is not None:
        root = canonical_path_identity(os.fspath(allowed_root))
        identity = canonical_path_identity(path)
        if not _within_root(identity, root):
            raise LabelConflictError("conflict_output_path_escape", path)
    with _label_mutex(path):
        current = resolve_existing_label(path)
        if current.presence == ANNOTATION_PRESENCE_INVALID:
            raise LabelConflictError("invalid_existing_label", path)
        if current.document_digest != pre_document_digest:
            raise LabelConflictError("conflict_pre_document_digest", path)
        if current.presence == ANNOTATION_PRESENCE_MISSING:
            return current
        if not os.path.isfile(path) or os.path.islink(path):
            raise LabelConflictError("conflict_output_not_regular_file", path)
        try:
            os.unlink(path)
        except OSError as exc:
            raise LabelWriteError("label_delete_failed", exc) from exc
        reread = resolve_existing_label(path)
        if reread.presence != ANNOTATION_PRESENCE_MISSING:
            raise LabelWriteError("post_delete_verification_failed")
        return reread


def _reject_session_symlink_components(path, root):
    requested = os.path.abspath(path)
    root_path = os.path.abspath(root)
    try:
        relative = os.path.relpath(requested, root_path)
    except ValueError as exc:
        raise ImageInputSnapshotError("session_image_path_escape") from exc
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ImageInputSnapshotError("session_image_path_escape")
    current = root_path
    for part in relative.split(os.sep):
        current = os.path.join(current, part)
        if os.path.lexists(current) and os.path.islink(current):
            raise ImageInputSnapshotError("session_image_symlink")


def _exif_orientation(raw_bytes):
    try:
        with PIL.Image.open(io.BytesIO(raw_bytes)) as image:
            orientation = image.getexif().get(0x0112)
    except PIL.UnidentifiedImageError:
        return None
    except Exception as exc:
        raise ImageInputSnapshotError("failed_input", exc) from exc
    if orientation is None:
        return None
    try:
        return int(orientation)
    except (TypeError, ValueError) as exc:
        raise ImageInputSnapshotError("failed_input", exc) from exc


def decode_image_input_snapshot_v1(
    *,
    image_id,
    requested_path,
    source,
    expected_sha256=None,
    session_images_root=None,
):
    """Read one target file once and freeze its decoded RGB888 contents."""

    path = os.fspath(requested_path)
    if not os.path.isabs(path):
        raise ImageInputSnapshotError("image_path_not_absolute")
    absolute_path = os.path.abspath(path)
    if source not in {"SESSION_MANIFEST", "STANDALONE_FILE_LIST"}:
        raise ImageInputSnapshotError("invalid_image_input_source")
    if source == "SESSION_MANIFEST":
        if session_images_root is None:
            raise ImageInputSnapshotError("session_images_root_required")
        root = os.path.realpath(os.path.abspath(session_images_root))
        real_path = os.path.realpath(absolute_path)
        if not _within_root(real_path, root):
            raise ImageInputSnapshotError("session_image_path_escape")
        _reject_session_symlink_components(absolute_path, session_images_root)
    elif session_images_root is not None:
        raise ImageInputSnapshotError("standalone_session_root_forbidden")
    try:
        with open(absolute_path, "rb") as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ImageInputSnapshotError("failed_input_not_regular_file")
            raw = stream.read()
    except ImageInputSnapshotError:
        raise
    except OSError as exc:
        raise ImageInputSnapshotError("failed_input", exc) from exc
    actual_sha256 = raw_file_sha256_v1(raw)
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise ImageInputSnapshotError("conflict_image_changed")
    orientation = _exif_orientation(raw)
    if orientation not in {None, 1}:
        raise ImageInputSnapshotError("exif_not_normalized")
    try:
        decoded = decode_image_for_labeling(raw)
    except (TypeError, ValueError) as exc:
        raise ImageInputSnapshotError("failed_input", exc) from exc
    rgb = decoded.convertToFormat(QtGui.QImage.Format_RGB888)
    width = rgb.width()
    height = rgb.height()
    if width <= 0 or height <= 0:
        raise ImageInputSnapshotError("failed_input")
    bits = rgb.bits()
    bits.setsize(rgb.byteCount())
    padded = bytes(bits)
    row_size = width * 3
    rgb_bytes = b"".join(
        padded[row * rgb.bytesPerLine() : row * rgb.bytesPerLine() + row_size]
        for row in range(height)
    )
    contract = {
        "snapshot_schema_version": 1,
        "image_id": image_id,
        "requested_path": absolute_path,
        "canonical_path": canonical_path_identity(absolute_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "file_size": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "decoded_width": width,
        "decoded_height": height,
        "pixel_format": "RGB888",
        "exif_orientation": orientation,
        "source": source,
    }
    validate_image_input_snapshot_v1(contract)
    return ImageInputSnapshotV1(
        **{
            key: value
            for key, value in contract.items()
            if key != "snapshot_schema_version"
        },
        rgb888=rgb_bytes,
    )


def _fault(store, point):
    fault = getattr(store, "fault", None)
    if fault is not None:
        fault(point)


def _mark_conflict(store, image_id, conflict_code):
    item = store.read_item(image_id)
    return store.mark_commit_conflict(
        image_id,
        conflict_code,
        expected_item_revision=item["item_revision"],
    )


def _is_v2_commit_sink(event_sink):
    return event_sink is not None and all(
        callable(getattr(event_sink, name, None))
        for name in ("prepare", "checkpoint")
    )


def _publish_v1_commit_event(
    *,
    v2_sink,
    current_event,
    event_sink,
    store,
    project_id,
    session_id,
    run_id,
    image_id,
    attempt_id,
    document_digest,
    semantic_digest,
    source_image_digest,
    base_item_revision,
    item,
):
    if v2_sink:
        return current_event, item
    if event_sink is None and all(
        value is None for value in (project_id, session_id, run_id)
    ):
        return None, item
    from .auto_labeling_run_store import build_annotation_commit_event_v1

    event = build_annotation_commit_event_v1(
        project_id=project_id,
        session_id=session_id,
        run_id=run_id,
        image_id=image_id,
        attempt_id=attempt_id,
        writer_kind="CONTINUOUS",
        commit_scope="STAGED",
        mutation_mode="NOTIFY_ONLY",
        document_digest=document_digest,
        semantic_digest=semantic_digest,
        source_image_digest=source_image_digest,
        base_item_revision=base_item_revision,
    )
    if event_sink is not None:
        event_sink.publish(event)
        item = store.read_item(image_id)
    return event, item


def _recover_v2_committed(
    event_sink,
    annotation_item_store,
    image_id,
    existing,
):
    if not _is_v2_commit_sink(event_sink):
        return False
    recover = getattr(annotation_item_store, "recover_commit", None)
    if not callable(recover):
        raise LabelConflictError("annotation_item_store_recovery_unavailable")
    action, _authority_item = recover(
        image_id,
        existing.document_digest,
        existing.semantic_digest,
    )
    if action not in {"NO_OP", "CHECKPOINTED"}:
        raise LabelConflictError("conflict_session_item_changed")
    return True


def _already_committed_result(
    *,
    store,
    item,
    existing,
    image_id,
    attempt_id,
    write_policy,
    event_sink,
    project_id,
    session_id,
    run_id,
    source_image_digest,
    annotation_item_store,
):
    staged_document_digest = item["digests"]["staged_document_digest"]
    if existing.document_digest != staged_document_digest:
        _mark_conflict(store, image_id, "conflict_committed_document_changed")
        raise LabelConflictError("conflict_committed_document_changed")
    summary = item["result_summary"]
    composition = CompositionResultV1(
        action="already_committed",
        effective_policy=write_policy,
        document=copy.deepcopy(existing.document),
        document_digest=existing.document_digest,
        semantic_digest=existing.semantic_digest,
        semantic_change=summary["semantic_change"],
        target_count=summary["target_count"],
        zero_target=summary["zero_target"],
    )
    v2_sink = _recover_v2_committed(
        event_sink,
        annotation_item_store,
        image_id,
        existing,
    )
    event, item = _publish_v1_commit_event(
        v2_sink=v2_sink,
        current_event=None,
        event_sink=event_sink,
        store=store,
        project_id=project_id,
        session_id=session_id,
        run_id=run_id,
        image_id=image_id,
        attempt_id=attempt_id,
        document_digest=existing.document_digest,
        semantic_digest=existing.semantic_digest,
        source_image_digest=source_image_digest,
        base_item_revision=item["item_revision"],
        item=item,
    )
    return ImageCommitResultV1(composition, None, item, event)


def _prepare_v2_commit_event(
    *,
    event_sink,
    annotation_item_store,
    image_id,
    project_id,
    session_id,
    existing,
    composition,
    model_fingerprint,
    label_path,
):
    if not _is_v2_commit_sink(event_sink):
        return None, False
    if annotation_item_store is None:
        raise LabelConflictError("annotation_item_store_unavailable")
    authority_item = annotation_item_store.read_item(image_id)
    if (
        authority_item["staged_document_digest"] != existing.document_digest
        or authority_item["staged_semantic_digest"] != existing.semantic_digest
    ):
        raise LabelConflictError("conflict_session_item_changed")
    from .auto_labeling_run_store import build_annotation_commit_event_v2

    event = build_annotation_commit_event_v2(
        project_id=project_id,
        session_id=session_id,
        image_id=image_id,
        base_session_item_revision=authority_item["revision"],
        writer="AUTOMATIC",
        pre_document_digest=existing.document_digest,
        pre_semantic_digest=existing.semantic_digest,
        intended_document_digest=composition.document_digest,
        intended_semantic_digest=composition.semantic_digest,
        model_fingerprint=model_fingerprint,
    )
    event_sink.prepare(copy.deepcopy(event), label_path=label_path)
    return event, True


def commit_label_for_image_v1(
    *,
    store,
    image_id,
    attempt_id,
    label_path,
    image_path,
    image_height,
    image_width,
    prediction_outcome,
    write_policy,
    image_data=None,
    flags=None,
    other_data=None,
    version=__version__,
    pose_config=None,
    kie_linking_adapter=None,
    allowed_root=None,
    reserved_paths=(),
    event_sink=None,
    project_id=None,
    session_id=None,
    run_id=None,
    source_image_digest=None,
    annotation_item_store=None,
    model_fingerprint=None,
):
    """Run the sole image intent -> label -> checkpoint transaction."""

    from .auto_labeling_run_store import CommitStoreProtocolV1

    if not isinstance(store, CommitStoreProtocolV1):
        raise TypeError("store does not implement CommitStoreProtocolV1")
    item = store.read_item(image_id)
    if item["latest_attempt_id"] != attempt_id:
        raise LabelConflictError("stale_attempt")
    existing = resolve_existing_label(label_path)
    if (
        item["staged_commit_status"] == "committed"
        and item["execution_status"] == "succeeded"
    ):
        return _already_committed_result(
            store=store,
            item=item,
            existing=existing,
            image_id=image_id,
            attempt_id=attempt_id,
            write_policy=write_policy,
            event_sink=event_sink,
            project_id=project_id,
            session_id=session_id,
            run_id=run_id,
            source_image_digest=source_image_digest,
            annotation_item_store=annotation_item_store,
        )
    if item["staged_commit_status"] == "prepared":
        raise LabelConflictError("prepared_commit_requires_recovery")
    try:
        composition = compose_final_label_document(
            existing,
            prediction_outcome,
            write_policy,
            label_path=label_path,
            image_path=image_path,
            image_height=image_height,
            image_width=image_width,
            image_data=image_data,
            flags=flags,
            other_data=other_data,
            version=version,
            pose_config=pose_config,
            kie_linking_adapter=kie_linking_adapter,
        )
    except LabelConflictError as exc:
        _mark_conflict(store, image_id, exc.code)
        raise
    if composition.action == "skipped":
        skipped = store.mark_existing_annotation_skipped(
            image_id,
            attempt_id,
            expected_item_revision=item["item_revision"],
        )
        store.rebuild_summary()
        return ImageCommitResultV1(composition, None, skipped, None)
    intent = {
        "attempt_id": attempt_id,
        "pre_file_sha256": existing.raw_file_sha256,
        "pre_document_digest": existing.document_digest,
        "pre_semantic_digest": existing.semantic_digest,
        "intended_document_digest": composition.document_digest,
        "intended_semantic_digest": composition.semantic_digest,
        "phase": "prepared",
    }
    _fault(store, "before_intent")
    event, v2_sink = _prepare_v2_commit_event(
        event_sink=event_sink,
        annotation_item_store=annotation_item_store,
        image_id=image_id,
        project_id=project_id,
        session_id=session_id,
        existing=existing,
        composition=composition,
        model_fingerprint=model_fingerprint,
        label_path=label_path,
    )
    prepared = store.write_commit_intent(
        image_id,
        attempt_id,
        intent,
        expected_item_revision=item["item_revision"],
        result_summary={
            "target_count": composition.target_count,
            "zero_target": composition.zero_target,
            "semantic_change": composition.semantic_change,
        },
    )
    _fault(store, "after_intent_before_label_write")
    try:
        write_result = atomic_write_label_document(
            label_path,
            composition.document,
            pre_document_digest=existing.document_digest,
            allowed_root=allowed_root,
            image_source_path=image_path,
            reserved_paths=reserved_paths,
            after_replace_hook=lambda: _fault(
                store, "after_label_replace_before_checkpoint"
            ),
        )
    except LabelConflictError as exc:
        _mark_conflict(store, image_id, exc.code)
        raise
    checkpoint = store.checkpoint_staged_commit(
        image_id,
        attempt_id,
        expected_item_revision=prepared["item_revision"],
        staged_document_digest=write_result.document_digest,
        staged_annotation_digest=write_result.semantic_digest,
        target_count=composition.target_count,
        zero_target=composition.zero_target,
        semantic_change=composition.semantic_change,
    )
    if v2_sink:
        event_sink.checkpoint(copy.deepcopy(event), label_path=label_path)
    _fault(store, "after_checkpoint_before_summary")
    store.rebuild_summary()
    event, checkpoint = _publish_v1_commit_event(
        v2_sink=v2_sink,
        current_event=event,
        event_sink=event_sink,
        store=store,
        project_id=project_id,
        session_id=session_id,
        run_id=run_id,
        image_id=image_id,
        attempt_id=attempt_id,
        document_digest=write_result.document_digest,
        semantic_digest=write_result.semantic_digest,
        source_image_digest=source_image_digest,
        base_item_revision=checkpoint["item_revision"],
        item=checkpoint,
    )
    return ImageCommitResultV1(composition, write_result, checkpoint, event)


def recover_image_commit_v1(*, store, image_id, label_path):
    """Recover a crash using document digest as the sole state decision."""

    item = store.read_item(image_id)
    intent = item["commit_intent"]
    current = resolve_existing_label(label_path)
    current_digest = current.document_digest
    if intent is None:
        if item["staged_commit_status"] == "committed":
            if current_digest != item["digests"]["staged_document_digest"]:
                conflict = store.mark_commit_conflict(
                    image_id,
                    "conflict_committed_document_changed",
                    item["item_revision"],
                )
                return CommitRecoveryResultV1(
                    "CONFLICT", conflict, current_digest
                )
            store.rebuild_summary()
            return CommitRecoveryResultV1(
                "CHECKPOINT_PRESENT", item, current_digest
            )
        if item["execution_status"] == "running":
            reset = store.reset_interrupted_item(
                image_id, item["item_revision"]
            )
            return CommitRecoveryResultV1(
                "RETURNED_TO_QUEUE", reset, current_digest
            )
        return CommitRecoveryResultV1("NO_ACTION", item, current_digest)
    attempt_id = intent["attempt_id"]
    if current_digest == intent["intended_document_digest"]:
        summary = item["result_summary"]
        checkpoint = store.checkpoint_staged_commit(
            image_id,
            attempt_id,
            expected_item_revision=item["item_revision"],
            staged_document_digest=intent["intended_document_digest"],
            staged_annotation_digest=intent["intended_semantic_digest"],
            target_count=summary["target_count"],
            zero_target=summary["zero_target"],
            semantic_change=summary["semantic_change"],
        )
        store.rebuild_summary()
        return CommitRecoveryResultV1(
            "COMPLETED_CHECKPOINT", checkpoint, current_digest
        )
    if current_digest == intent["pre_document_digest"]:
        rolled_back = store.rollback_prepared_commit(
            image_id, attempt_id, item["item_revision"]
        )
        return CommitRecoveryResultV1(
            "RETURNED_TO_QUEUE", rolled_back, current_digest
        )
    conflict = store.mark_commit_conflict(
        image_id,
        "conflict_label_changed_during_commit",
        item["item_revision"],
    )
    return CommitRecoveryResultV1("CONFLICT", conflict, current_digest)
