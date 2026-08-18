"""Continuous sequence options, capability registry, and model identity."""

import copy
import hashlib
import importlib.resources as pkg_resources
import math
import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from urllib.parse import parse_qsl, urlsplit

import anylabeling.configs as auto_labeling_configs


FAST_RANGES_V1 = frozenset({"ALL_IMAGES", "CURRENT_TO_END"})
FAST_FILTERS_V1 = frozenset({"ALL", "ONLY_WITHOUT_VALID_ANNOTATION"})
FAST_WRITE_POLICIES_V1 = frozenset(
    {
        "SKIP_EXISTING",
        "FORCE_REPLACE",
    }
)
FAST_WORKSET_SOURCES_V1 = frozenset(
    {"SESSION_WORKSET", "CURRENT_FILE_LIST_SNAPSHOT"}
)
SEQUENCE_RANGES_V1 = FAST_RANGES_V1
SEQUENCE_FILTERS_V1 = FAST_FILTERS_V1
SEQUENCE_WRITE_POLICIES_V1 = FAST_WRITE_POLICIES_V1
SEQUENCE_WORKSET_SOURCES_V1 = FAST_WORKSET_SOURCES_V1
_CREDENTIAL_KEY = re.compile(
    r"(api[_-]?key|token|secret|authorization|password)", re.IGNORECASE
)


class FastSequenceContractError(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        self.detail = str(detail or "")
        super().__init__(code if not self.detail else f"{code}: {self.detail}")


class ModelFingerprintError(FastSequenceContractError):
    pass


def _credential_url(value):
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


def _plain_json(value, path="$value"):
    if value is None or type(value) in {bool, int}:
        return copy.deepcopy(value)
    if type(value) is str:
        if _credential_url(value):
            raise FastSequenceContractError("credential_url_forbidden", path)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise FastSequenceContractError("non_finite_value", path)
        return value
    if type(value) in {list, tuple}:
        return [
            _plain_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) in {dict, MappingProxyType}:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise FastSequenceContractError("non_string_key", path)
            if _CREDENTIAL_KEY.search(key):
                raise FastSequenceContractError(
                    "credential_field_forbidden", key
                )
            result[key] = _plain_json(item, f"{path}.{key}")
        return result
    raise FastSequenceContractError(
        "non_json_value", f"{path}:{type(value).__name__}"
    )


def _freeze_json(value):
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def thaw_json(value):
    if isinstance(value, MappingProxyType):
        return {key: thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [thaw_json(item) for item in value]
    return copy.deepcopy(value)


def validate_model_fingerprint_v1(value):
    try:
        fingerprint = _plain_json(value, "$model_fingerprint")
    except FastSequenceContractError as exc:
        raise ModelFingerprintError(exc.code, exc.detail) from exc
    required = {
        "fingerprint_schema_version",
        "model_type",
        "config_digest",
        "artifact_digests",
        "remote_endpoint_identity",
        "adapter_version",
        "resume_supported",
    }
    if type(fingerprint) is not dict or set(fingerprint) != required:
        raise ModelFingerprintError("model_fingerprint_unavailable")
    if fingerprint["fingerprint_schema_version"] != 1:
        raise ModelFingerprintError("unsupported_model_fingerprint_schema")
    if (
        type(fingerprint["model_type"]) is not str
        or not fingerprint["model_type"].strip()
        or type(fingerprint["adapter_version"]) is not str
        or not fingerprint["adapter_version"].strip()
        or type(fingerprint["resume_supported"]) is not bool
        or type(fingerprint["config_digest"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint["config_digest"]) is None
    ):
        raise ModelFingerprintError("model_fingerprint_unavailable")
    artifacts = fingerprint["artifact_digests"]
    if type(artifacts) is not list or not artifacts:
        raise ModelFingerprintError("model_fingerprint_unavailable")
    fields = set()
    for artifact in artifacts:
        if type(artifact) is not dict or set(artifact) != {
            "field",
            "sha256",
            "size",
        }:
            raise ModelFingerprintError("model_fingerprint_unavailable")
        field = artifact["field"]
        if (
            type(field) is not str
            or not field
            or field in fields
            or type(artifact["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None
            or type(artifact["size"]) is not int
            or artifact["size"] < 0
        ):
            raise ModelFingerprintError("model_fingerprint_unavailable")
        fields.add(field)
    endpoint = fingerprint["remote_endpoint_identity"]
    if endpoint is not None and (
        type(endpoint) is not str or _credential_url(endpoint)
    ):
        raise ModelFingerprintError("credential_url_forbidden")
    return fingerprint


def validate_sequence_delay_seconds_v1(value):
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise FastSequenceContractError("invalid_sequence_delay_seconds")
    normalized = float(value)
    if normalized < 0.0 or normalized > 60.0:
        raise FastSequenceContractError("invalid_sequence_delay_seconds")
    doubled = normalized * 2.0
    if not doubled.is_integer():
        raise FastSequenceContractError("invalid_sequence_delay_step")
    return normalized


def _normalize_run_options(instance, error_prefix):
    if instance.range not in SEQUENCE_RANGES_V1:
        raise FastSequenceContractError(f"invalid_{error_prefix}_range")
    if instance.filter not in SEQUENCE_FILTERS_V1:
        raise FastSequenceContractError(f"invalid_{error_prefix}_filter")
    if instance.write_policy not in SEQUENCE_WRITE_POLICIES_V1:
        raise FastSequenceContractError(f"invalid_{error_prefix}_write_policy")
    if instance.workset_source not in SEQUENCE_WORKSET_SOURCES_V1:
        raise FastSequenceContractError(
            f"invalid_{error_prefix}_workset_source"
        )
    if instance.range == "CURRENT_TO_END" and (
        type(instance.current_anchor_image_id) is not str
        or not instance.current_anchor_image_id
    ):
        raise FastSequenceContractError("current_anchor_image_id_required")
    fingerprint = validate_model_fingerprint_v1(instance.model_fingerprint)
    parameters = _plain_json(
        instance.parameter_snapshot, "$parameter_snapshot"
    )
    if type(fingerprint) is not dict or type(parameters) is not dict:
        raise FastSequenceContractError(
            f"invalid_{error_prefix}_options_payload"
        )
    object.__setattr__(
        instance, "model_fingerprint", _freeze_json(fingerprint)
    )
    object.__setattr__(
        instance, "parameter_snapshot", _freeze_json(parameters)
    )


@dataclass(frozen=True)
class SequenceRunOptionsV1:
    delay_seconds: float
    range: str
    filter: str
    write_policy: str
    current_anchor_image_id: str | None
    workset_source: str
    model_fingerprint: object
    parameter_snapshot: object

    def __post_init__(self):
        delay = validate_sequence_delay_seconds_v1(self.delay_seconds)
        object.__setattr__(self, "delay_seconds", delay)
        _normalize_run_options(self, "sequence")

    @property
    def execution_mode(self):
        return "FAST" if self.delay_seconds == 0.0 else "VISIBLE"


@dataclass(frozen=True)
class FastRunOptionsV1(SequenceRunOptionsV1):
    def __post_init__(self):
        if (
            type(self.delay_seconds) not in {int, float}
            or isinstance(self.delay_seconds, bool)
            or self.delay_seconds != 0.0
        ):
            raise FastSequenceContractError(
                "visible_mode_not_available_phase4"
            )
        object.__setattr__(self, "delay_seconds", 0.0)
        _normalize_run_options(self, "fast")


@dataclass(frozen=True)
class AutoLabelingSequenceCapabilities:
    supports_fast_sequence: bool
    supports_visible_sequence: bool
    single_image_independent: bool
    requires_frozen_text_prompt: bool
    uses_existing_shapes_input: bool
    stateful_across_images: bool
    owns_replace_policy: bool
    supports_safe_shutdown: bool
    adapter_version: str


LEGACY_SEQUENCE_CAPABILITIES = AutoLabelingSequenceCapabilities(
    supports_fast_sequence=False,
    supports_visible_sequence=False,
    single_image_independent=False,
    requires_frozen_text_prompt=False,
    uses_existing_shapes_input=False,
    stateful_across_images=True,
    owns_replace_policy=False,
    supports_safe_shutdown=False,
    adapter_version="legacy-batch",
)


_YOLO_DETECT_SEQUENCE_TYPES = frozenset(
    {
        "yolov5",
        "yolov6",
        "yolov7",
        "yolov8",
        "yolov9",
        "yolov10",
        "yolo11",
        "yolo12",
        "yolo26",
    }
)
_YOLO_POSE_SEQUENCE_TYPES = frozenset({"yolov8_pose", "yolo11_pose"})
_YOLO_FAST_TYPES = _YOLO_DETECT_SEQUENCE_TYPES | _YOLO_POSE_SEQUENCE_TYPES

_INTERACTIVE_SAM_TYPES = frozenset(
    {
        "edge_sam",
        "efficientvit_sam",
        "grounding_sam",
        "grounding_sam2",
        "sam_hq",
        "sam_med2d",
        "segment_anything",
        "segment_anything_2",
        "yolov5_sam",
        "yolov8_sam2",
    }
)
_VIDEO_SEQUENCE_TYPES = frozenset(
    {
        "segment_anything_2_video",
        "sam2_video",
        "sam3_video",
    }
)
_REMOTE_MULTI_IMAGE_TYPES = frozenset({"remote_server"})

UNIFIED_SEQUENCE_ROUTE_V1 = "UNIFIED_SEQUENCE"
LEGACY_BATCH_ROUTE_V1 = "LEGACY_BATCH"
_YOLO_FAST_CAPABILITIES = AutoLabelingSequenceCapabilities(
    supports_fast_sequence=True,
    supports_visible_sequence=True,
    single_image_independent=True,
    requires_frozen_text_prompt=False,
    uses_existing_shapes_input=False,
    stateful_across_images=False,
    owns_replace_policy=True,
    supports_safe_shutdown=True,
    adapter_version="phase4-yolo-v1",
)


@dataclass(frozen=True)
class SequenceCapabilityDecisionV1:
    model_family: str
    route: str
    reason_code: str | None
    creates_unified_audit_queue: bool
    capabilities: AutoLabelingSequenceCapabilities


def resolve_sequence_capabilities(model_config):
    if type(model_config) is not dict:
        return LEGACY_SEQUENCE_CAPABILITIES
    model_type = model_config.get("type")
    if model_type in _YOLO_FAST_TYPES:
        return _YOLO_FAST_CAPABILITIES
    return LEGACY_SEQUENCE_CAPABILITIES


def resolve_sequence_capability_decision_v1(model_config):
    """Return the explicit Fast/Visible or Legacy routing decision."""

    capabilities = resolve_sequence_capabilities(model_config)
    model_type = (
        model_config.get("type") if type(model_config) is dict else None
    )
    if model_type in _YOLO_DETECT_SEQUENCE_TYPES:
        family = "YOLO_DETECT"
    elif model_type in _YOLO_POSE_SEQUENCE_TYPES:
        family = "YOLO_POSE"
    elif type(model_type) is str and model_type.endswith("_track"):
        family = "TRACKER"
    elif model_type in _VIDEO_SEQUENCE_TYPES:
        family = "VIDEO"
    elif model_type in _INTERACTIVE_SAM_TYPES:
        family = "INTERACTIVE_SAM"
    elif model_type in _REMOTE_MULTI_IMAGE_TYPES:
        family = "REMOTE_MULTI_IMAGE"
    elif type(model_type) is str and (
        model_type.endswith("_seg") or "_seg_" in model_type
    ):
        family = "SEGMENTATION"
    elif type(model_type) is str and (
        model_type.endswith("_obb") or "_obb_" in model_type
    ):
        family = "OBB"
    else:
        family = "UNVERIFIED"

    if capabilities.supports_fast_sequence:
        return SequenceCapabilityDecisionV1(
            model_family=family,
            route=UNIFIED_SEQUENCE_ROUTE_V1,
            reason_code=None,
            creates_unified_audit_queue=True,
            capabilities=capabilities,
        )

    reasons = {
        "SEGMENTATION": "capability_not_validated",
        "OBB": "capability_not_validated",
        "TRACKER": "stateful_across_images",
        "VIDEO": "stateful_across_images",
        "INTERACTIVE_SAM": "interactive_input_required",
        "REMOTE_MULTI_IMAGE": "remote_multi_image_unverified",
        "UNVERIFIED": "model_not_registered",
    }
    return SequenceCapabilityDecisionV1(
        model_family=family,
        route=LEGACY_BATCH_ROUTE_V1,
        reason_code=reasons[family],
        creates_unified_audit_queue=False,
        capabilities=capabilities,
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _config_content(model_config):
    config_file = model_config.get("config_file")
    if type(config_file) is not str or not config_file:
        raise ModelFingerprintError("model_fingerprint_unavailable")
    try:
        if config_file.startswith(":/"):
            resource = pkg_resources.files(auto_labeling_configs).joinpath(
                "auto_labeling", config_file[2:]
            )
            return resource.read_bytes()
        with open(config_file, "rb") as stream:
            return stream.read()
    except (OSError, FileNotFoundError) as exc:
        raise ModelFingerprintError(
            "model_fingerprint_unavailable", "config"
        ) from exc


def build_model_fingerprint_v1(model_config, artifact_resolver=None):
    capabilities = resolve_sequence_capabilities(model_config)
    if not capabilities.supports_fast_sequence:
        raise ModelFingerprintError("model_fingerprint_unavailable")
    model = model_config.get("model")
    if artifact_resolver is None:
        resolver = getattr(model, "get_model_abs_path", None)

        def artifact_resolver(field):
            if callable(resolver):
                return resolver(model_config, field)
            value = model_config.get(field)
            return value if type(value) is str else None

    artifact_fields = sorted(
        key
        for key, value in model_config.items()
        if type(value) is str
        and (key == "model_path" or key.endswith("_model_path"))
    )
    artifacts = []
    for field in artifact_fields:
        try:
            path = artifact_resolver(field)
        except Exception as exc:
            raise ModelFingerprintError(
                "model_fingerprint_unavailable", field
            ) from exc
        if type(path) is not str or not os.path.isfile(path):
            raise ModelFingerprintError("model_fingerprint_unavailable", field)
        digest, size = _sha256_file(path)
        artifacts.append({"field": field, "sha256": digest, "size": size})
    if not artifacts:
        raise ModelFingerprintError(
            "model_fingerprint_unavailable", "artifacts"
        )
    return {
        "fingerprint_schema_version": 1,
        "model_type": model_config["type"],
        "config_digest": hashlib.sha256(
            _config_content(model_config)
        ).hexdigest(),
        "artifact_digests": artifacts,
        "remote_endpoint_identity": None,
        "adapter_version": capabilities.adapter_version,
        "resume_supported": True,
    }


def _control_value(control, method):
    function = getattr(control, method, None)
    return function() if callable(function) else None


def build_parameter_snapshot_v1(model_config, auto_widget=None):
    if type(model_config) is not dict or model_config.get("model") is None:
        raise FastSequenceContractError("prediction_model_not_loaded")
    model = model_config["model"]
    snapshot = {
        "model_type": model_config.get("type"),
        "model_name": model_config.get("display_name")
        or model_config.get("name")
        or model_config.get("type"),
        "confidence_threshold": None,
        "iou_threshold": None,
        "keypoint_threshold": None,
        "output_mode": getattr(model, "output_mode", None),
        "preserve_existing_annotations": None,
        "replace": getattr(model, "replace", None),
        "skip_detection": False,
        "cropping_mode": None,
        "mask_fineness": None,
    }
    for output_field, names in (
        ("confidence_threshold", ("conf_thres", "conf_threshold")),
        ("iou_threshold", ("iou_thres", "iou_threshold")),
        ("keypoint_threshold", ("kpt_thres", "kpt_threshold")),
    ):
        for name in names:
            value = getattr(model, name, None)
            if type(value) in {int, float} and not isinstance(value, bool):
                snapshot[output_field] = float(value)
                break
    if auto_widget is not None:
        conf = _control_value(getattr(auto_widget, "edit_conf", None), "value")
        iou = _control_value(getattr(auto_widget, "edit_iou", None), "value")
        preserve = _control_value(
            getattr(
                auto_widget,
                "toggle_preserve_existing_annotations",
                None,
            ),
            "isChecked",
        )
        skip = _control_value(
            getattr(auto_widget, "button_skip_detection", None),
            "isChecked",
        )
        cropping = _control_value(
            getattr(auto_widget, "button_cropping", None), "isChecked"
        )
        fineness = _control_value(
            getattr(auto_widget, "mask_fineness_slider", None), "value"
        )
        if type(conf) in {int, float}:
            snapshot["confidence_threshold"] = float(conf)
        if type(iou) in {int, float}:
            snapshot["iou_threshold"] = float(iou)
        if type(preserve) is bool:
            snapshot["preserve_existing_annotations"] = preserve
            snapshot["replace"] = not preserve
        if type(skip) is bool:
            snapshot["skip_detection"] = skip
        if type(cropping) is bool:
            snapshot["cropping_mode"] = cropping
        if type(fineness) in {int, float}:
            snapshot["mask_fineness"] = float(fineness)
    return _plain_json(snapshot, "$parameter_snapshot")
