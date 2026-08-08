import copy
import json
import math
import os
from dataclasses import dataclass

from PyQt5.QtCore import QPointF


DELIVERY_MODES = {"RETURN_ONLY", "LEGACY_CANVAS"}
OUTCOME_STATUSES = {"succeeded", "failed", "cancelled"}


class PredictionContractError(ValueError):
    """Raised when a prediction value violates the frozen V5 contract."""


def _contract_error(code, detail=None):
    message = code if detail is None else f"{code}: {detail}"
    raise PredictionContractError(message)


def _plain_value(value, path="value"):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        value = float(value)
        if not math.isfinite(value):
            _contract_error("prediction_payload_not_finite", path)
        return value
    if isinstance(value, QPointF):
        return [float(value.x()), float(value.y())]

    # NumPy scalars expose item(), without making NumPy a runtime dependency.
    value_type = type(value)
    if value_type.__module__.split(".", 1)[0] == "numpy" and hasattr(
        value, "item"
    ):
        return _plain_value(value.item(), path)

    if isinstance(value, dict):
        plain = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _contract_error("prediction_payload_key_not_string", path)
            plain[key] = _plain_value(item, f"{path}.{key}")
        return plain
    if isinstance(value, (list, tuple)):
        return [
            _plain_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    _contract_error(
        "prediction_payload_unsupported_type",
        f"{path}={type(value).__name__}",
    )


def _plain_copy(value, path):
    plain = _plain_value(value, path)
    try:
        json.dumps(plain, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        _contract_error("prediction_payload_not_serializable", str(exc))
    return copy.deepcopy(plain)


def _required_string(value, field):
    if not isinstance(value, str) or not value:
        _contract_error("prediction_context_invalid", field)
    return value


@dataclass(frozen=True)
class PredictionRequest:
    run_id: str
    job_id: str
    attempt_id: str
    image_id: str
    canonical_image_path: str
    generation: int
    parameter_snapshot: dict
    existing_shapes_input: list
    delivery_mode: str

    def __post_init__(self):
        for field in (
            "run_id",
            "job_id",
            "attempt_id",
            "image_id",
            "canonical_image_path",
        ):
            object.__setattr__(
                self, field, _required_string(getattr(self, field), field)
            )
        if not os.path.isabs(self.canonical_image_path):
            _contract_error(
                "prediction_context_invalid", "canonical_image_path"
            )
        if isinstance(self.generation, bool) or not isinstance(
            self.generation, int
        ):
            _contract_error("prediction_context_invalid", "generation")
        if self.generation < 0:
            _contract_error("prediction_context_invalid", "generation")
        if self.delivery_mode not in DELIVERY_MODES:
            _contract_error("prediction_delivery_mode_invalid")

        parameters = _plain_copy(self.parameter_snapshot, "parameter_snapshot")
        existing = _plain_copy(
            self.existing_shapes_input, "existing_shapes_input"
        )
        if not isinstance(parameters, dict):
            _contract_error("prediction_context_invalid", "parameter_snapshot")
        if not isinstance(existing, list):
            _contract_error(
                "prediction_context_invalid", "existing_shapes_input"
            )
        object.__setattr__(self, "parameter_snapshot", parameters)
        object.__setattr__(self, "existing_shapes_input", existing)

    @property
    def identity(self):
        return (
            self.run_id,
            self.job_id,
            self.attempt_id,
            self.image_id,
            self.generation,
        )


@dataclass(frozen=True)
class AutoLabelingPayload:
    shapes: list
    replace: bool
    description: str

    def __post_init__(self):
        if not isinstance(self.replace, bool):
            _contract_error("prediction_replace_not_bool")
        if not isinstance(self.description, str):
            _contract_error("prediction_description_not_string")
        shapes = _plain_copy(self.shapes, "shapes")
        if not isinstance(shapes, list):
            _contract_error("prediction_shapes_not_list")
        if any(not isinstance(shape, dict) for shape in shapes):
            _contract_error("prediction_shape_not_object")
        object.__setattr__(self, "shapes", shapes)


@dataclass(frozen=True)
class PredictionOutcome:
    context: PredictionRequest
    status: str
    result: AutoLabelingPayload | None
    zero_target: bool
    error_code: str | None
    error_message: str | None

    def __post_init__(self):
        if not isinstance(self.context, PredictionRequest):
            _contract_error("prediction_outcome_context_invalid")
        if self.status not in OUTCOME_STATUSES:
            _contract_error("prediction_outcome_status_invalid")
        if not isinstance(self.zero_target, bool):
            _contract_error("prediction_zero_target_not_bool")

        if self.status == "succeeded":
            if not isinstance(self.result, AutoLabelingPayload):
                _contract_error("prediction_result_missing")
            if self.zero_target != (len(self.result.shapes) == 0):
                _contract_error("prediction_zero_target_mismatch")
            if self.error_code is not None or self.error_message is not None:
                _contract_error("prediction_success_has_error")
            return

        if self.result is not None:
            _contract_error("prediction_failure_has_result")
        if self.zero_target:
            _contract_error("prediction_failure_zero_target")
        _required_string(self.error_code, "error_code")
        _required_string(self.error_message, "error_message")

    @classmethod
    def succeeded(cls, context, result):
        if not isinstance(result, AutoLabelingPayload):
            _contract_error("prediction_result_missing")
        return cls(
            context=context,
            status="succeeded",
            result=result,
            zero_target=len(result.shapes) == 0,
            error_code=None,
            error_message=None,
        )

    @classmethod
    def failed(cls, context, error_code, error_message):
        return cls(
            context=context,
            status="failed",
            result=None,
            zero_target=False,
            error_code=error_code,
            error_message=error_message,
        )

    @classmethod
    def cancelled(cls, context, error_code, error_message):
        return cls(
            context=context,
            status="cancelled",
            result=None,
            zero_target=False,
            error_code=error_code,
            error_message=error_message,
        )


def normalize_auto_labeling_result(result):
    """Convert a legacy AutoLabelingResult to a plain V5 payload."""

    if result is None:
        _contract_error("prediction_result_missing")
    if not hasattr(result, "replace"):
        _contract_error("prediction_replace_missing")
    if not isinstance(result.replace, bool):
        _contract_error("prediction_replace_not_bool")
    if not hasattr(result, "description"):
        _contract_error("prediction_description_missing")
    if not isinstance(result.description, str):
        _contract_error("prediction_description_not_string")
    if not hasattr(result, "shapes") or not isinstance(result.shapes, list):
        _contract_error("prediction_shapes_not_list")

    shapes = []
    for index, shape in enumerate(result.shapes):
        if isinstance(shape, dict):
            shape_value = shape
        elif hasattr(shape, "to_dict") and callable(shape.to_dict):
            try:
                shape_value = shape.to_dict()
            except Exception as exc:
                _contract_error("prediction_shape_conversion_failed", str(exc))
        else:
            _contract_error("prediction_shape_unsupported", f"shapes[{index}]")
        shapes.append(shape_value)

    return AutoLabelingPayload(
        shapes=shapes,
        replace=result.replace,
        description=result.description,
    )
