import os
import unittest

import numpy as np
from PyQt5 import QtCore

from anylabeling.services.auto_labeling.prediction_job import (
    AutoLabelingPayload,
    PredictionContractError,
    PredictionOutcome,
    PredictionRequest,
    normalize_auto_labeling_result,
)
from anylabeling.services.auto_labeling.types import AutoLabelingResult


class _Shape:
    def __init__(self, label="box"):
        self.label = label

    def to_dict(self):
        return {
            "label": self.label,
            "points": [QtCore.QPointF(1, 2), [np.float32(3), 4]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }


def _request(**overrides):
    values = {
        "run_id": "run-a",
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "image_id": "image-a",
        "canonical_image_path": os.path.abspath(__file__),
        "generation": 1,
        "parameter_snapshot": {"conf": 0.5},
        "existing_shapes_input": [],
        "delivery_mode": "RETURN_ONLY",
    }
    values.update(overrides)
    return PredictionRequest(**values)


class PredictionJobContractTests(unittest.TestCase):
    def test_successful_nonempty_payload_is_plain_and_deep_copied(self):
        source = AutoLabelingResult([_Shape("person")], False, "model")
        payload = normalize_auto_labeling_result(source)
        source.shapes[0].label = "changed"

        self.assertIsInstance(payload, AutoLabelingPayload)
        self.assertEqual(payload.shapes[0]["label"], "person")
        self.assertEqual(payload.shapes[0]["points"], [[1.0, 2.0], [3.0, 4]])
        self.assertIs(payload.replace, False)
        self.assertEqual(payload.description, "model")
        outcome = PredictionOutcome.succeeded(_request(), payload)
        self.assertEqual(outcome.status, "succeeded")
        self.assertFalse(outcome.zero_target)

    def test_successful_empty_payload_is_not_none(self):
        payload = normalize_auto_labeling_result(
            AutoLabelingResult([], True, "")
        )
        outcome = PredictionOutcome.succeeded(_request(), payload)
        self.assertEqual(payload.shapes, [])
        self.assertTrue(outcome.zero_target)
        self.assertIs(outcome.result, payload)

    def test_none_cannot_represent_zero_target_success(self):
        with self.assertRaisesRegex(
            PredictionContractError, "prediction_result_missing"
        ):
            normalize_auto_labeling_result(None)
        with self.assertRaises(PredictionContractError):
            PredictionOutcome(
                context=_request(),
                status="succeeded",
                result=None,
                zero_target=True,
                error_code=None,
                error_message=None,
            )

    def test_replace_must_be_an_explicit_bool(self):
        for value in (None, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    PredictionContractError, "prediction_replace_not_bool"
                ):
                    normalize_auto_labeling_result(
                        AutoLabelingResult([], value, "")
                    )

    def test_nan_infinity_and_unsupported_qt_objects_are_rejected(self):
        for value in (float("nan"), float("inf"), QtCore.QObject()):
            with self.subTest(value=type(value).__name__):
                result = AutoLabelingResult([_Shape()], True, "")
                result.shapes[0].to_dict = lambda value=value: {
                    "label": "bad",
                    "points": [[value, 1]],
                    "shape_type": "point",
                    "flags": {},
                }
                with self.assertRaises(PredictionContractError):
                    normalize_auto_labeling_result(result)

    def test_request_inputs_are_copied_and_context_identity_is_exact(self):
        parameters = {"nested": [1]}
        existing = [{"label": "old", "points": [[1, 2]]}]
        request = _request(
            parameter_snapshot=parameters,
            existing_shapes_input=existing,
        )
        parameters["nested"].append(2)
        existing[0]["label"] = "changed"
        self.assertEqual(request.parameter_snapshot, {"nested": [1]})
        self.assertEqual(request.existing_shapes_input[0]["label"], "old")
        self.assertNotEqual(
            request.identity,
            _request(job_id="job-b").identity,
        )
        with self.assertRaisesRegex(
            PredictionContractError, "canonical_image_path"
        ):
            _request(canonical_image_path="relative.jpg")

    def test_failed_and_cancelled_outcomes_never_carry_results(self):
        request = _request()
        failed = PredictionOutcome.failed(request, "model_failed", "boom")
        cancelled = PredictionOutcome.cancelled(
            request, "cancelled_before_model_start", "stopped"
        )
        self.assertIsNone(failed.result)
        self.assertIsNone(cancelled.result)
        self.assertFalse(failed.zero_target)
        self.assertFalse(cancelled.zero_target)
        with self.assertRaises(PredictionContractError):
            PredictionOutcome(
                context=request,
                status="failed",
                result=AutoLabelingPayload([], True, ""),
                zero_target=False,
                error_code="model_failed",
                error_message="boom",
            )

    def test_payload_copy_is_independent_of_caller_mutation(self):
        shapes = [_Shape("first").to_dict()]
        payload = AutoLabelingPayload(shapes, True, "")
        shapes[0]["label"] = "changed"
        self.assertEqual(payload.shapes[0]["label"], "first")
        self.assertEqual(payload.shapes[0]["points"], [[1.0, 2.0], [3.0, 4]])


if __name__ == "__main__":
    unittest.main()
