import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt5 import QtWidgets

from anylabeling.services.auto_labeling.prediction_job import (
    AutoLabelingPayload,
    PredictionOutcome,
)
from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    LabelConflictError,
    LabelWriteError,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    FastSequenceContractError,
    ModelFingerprintError,
    build_model_fingerprint_v1,
    build_parameter_snapshot_v1,
    resolve_sequence_capabilities,
    thaw_json,
    validate_model_fingerprint_v1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    FastAutoLabelingController,
    build_fast_run_summary_v1,
    create_standalone_fast_activation_v1,
    standalone_image_id_v1,
)
from tests.test_continuous_auto_labeling_phase4 import (
    _Manager,
    _fingerprint,
    _images,
    _options,
    _success,
    _valid_empty,
    _wait_until,
)


class FastSequenceContractTests(unittest.TestCase):
    def test_delay_is_fixed_and_payloads_are_deeply_immutable(self):
        parameters = {"confidence_threshold": 0.25, "nested": [1, {"x": 2}]}
        fingerprint = _fingerprint()
        options = _options(
            parameter_snapshot=parameters,
            model_fingerprint=fingerprint,
        )

        parameters["nested"][1]["x"] = 99
        fingerprint["artifact_digests"][0]["size"] = 99
        self.assertEqual(
            thaw_json(options.parameter_snapshot)["nested"], [1, {"x": 2}]
        )
        self.assertEqual(
            thaw_json(options.model_fingerprint)["artifact_digests"][0][
                "size"
            ],
            1,
        )
        with self.assertRaises(TypeError):
            options.parameter_snapshot["nested"] = ()
        with self.assertRaises(FastSequenceContractError) as raised:
            _options(delay_seconds=0.1)
        self.assertEqual(
            raised.exception.code, "visible_mode_not_available_phase4"
        )
        with self.assertRaises(FastSequenceContractError) as raised:
            _options(delay_seconds=False)
        self.assertEqual(
            raised.exception.code, "visible_mode_not_available_phase4"
        )

    def test_positive_capability_registry_excludes_stateful_models(self):
        for model_type in ("yolov8", "yolov8_pose", "yolo11_pose"):
            with self.subTest(model_type=model_type):
                capability = resolve_sequence_capabilities(
                    {"type": model_type, "model": object()}
                )
                self.assertTrue(capability.supports_fast_sequence)
                self.assertFalse(capability.supports_visible_sequence)
                self.assertTrue(capability.single_image_independent)
                self.assertFalse(capability.stateful_across_images)
                self.assertTrue(capability.supports_safe_shutdown)

        for model_type in (
            "yolov8_det_track",
            "sam2_hiera_large_video",
            "sam_hq_vit_b",
            "remote_server",
            "yolox",
        ):
            with self.subTest(model_type=model_type):
                capability = resolve_sequence_capabilities(
                    {"type": model_type, "model": object()}
                )
                self.assertFalse(capability.supports_fast_sequence)
                self.assertEqual(capability.adapter_version, "legacy-batch")

    def test_fingerprint_hashes_content_without_persisting_paths_or_secrets(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "model.yaml"
            artifact_path = root / "weights.onnx"
            config_path.write_bytes(
                b"type: yolov8\nmodel_path: weights.onnx\n"
            )
            artifact_path.write_bytes(b"model-weights")

            class Model:
                @staticmethod
                def get_model_abs_path(_config, _field):
                    return str(artifact_path)

            config = {
                "type": "yolov8",
                "model": Model(),
                "config_file": str(config_path),
                "model_path": "https://example.test/public-model.onnx",
            }
            fingerprint = build_model_fingerprint_v1(config)

            self.assertEqual(
                fingerprint["config_digest"],
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                fingerprint["artifact_digests"],
                [
                    {
                        "field": "model_path",
                        "sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                        "size": artifact_path.stat().st_size,
                    }
                ],
            )
            self.assertNotIn(str(root), json.dumps(fingerprint))
            self.assertIsNone(fingerprint["remote_endpoint_identity"])

        secret = _fingerprint()
        secret["remote_endpoint_identity"] = (
            "https://example.test/model?api_key=secret"
        )
        with self.assertRaises(ModelFingerprintError) as raised:
            validate_model_fingerprint_v1(secret)
        self.assertEqual(raised.exception.code, "credential_url_forbidden")

    def test_parameter_snapshot_copies_applicable_model_controls(self):
        class Control:
            def __init__(self, value):
                self._value = value

            def value(self):
                return self._value

            def isChecked(self):
                return self._value

        model = SimpleNamespace(
            conf_thres=0.2,
            iou_thres=0.4,
            kpt_thres=0.1,
            output_mode="rectangle",
            replace=True,
        )
        widget = SimpleNamespace(
            edit_conf=Control(0.35),
            edit_iou=Control(0.55),
            toggle_preserve_existing_annotations=Control(True),
            button_skip_detection=Control(False),
            button_cropping=Control(True),
            mask_fineness_slider=Control(7),
        )
        snapshot = build_parameter_snapshot_v1(
            {
                "type": "yolov8_pose",
                "display_name": "Pose model",
                "model": model,
            },
            widget,
        )
        self.assertEqual(snapshot["confidence_threshold"], 0.35)
        self.assertEqual(snapshot["iou_threshold"], 0.55)
        self.assertEqual(snapshot["keypoint_threshold"], 0.1)
        self.assertTrue(snapshot["preserve_existing_annotations"])
        self.assertTrue(snapshot["cropping_mode"])
        self.assertEqual(snapshot["mask_fineness"], 7.0)


class StandaloneActivationTests(unittest.TestCase):
    def test_current_to_end_and_annotation_filter_are_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, labels = _images(root, 3)
            (labels / "image-0001.json").write_text(
                json.dumps(_valid_empty("image-0001.png")), encoding="utf-8"
            )
            (labels / "image-0002.json").write_text(
                "{broken", encoding="utf-8"
            )
            options = _options(
                range="CURRENT_TO_END",
                current_anchor_image_id=standalone_image_id_v1(paths[1]),
                filter="ONLY_WITHOUT_VALID_ANNOTATION",
            )

            activation = create_standalone_fast_activation_v1(
                paths, options, output_dir=str(labels)
            )
            items = activation.run_store.list_items(activation.run_id)

            self.assertEqual(
                items[0]["result_summary"]["skip_reason"],
                "outside_selected_range",
            )
            self.assertEqual(
                items[1]["result_summary"]["skip_reason"],
                "existing_annotation",
            )
            self.assertEqual(items[2]["execution_status"], "conflict")
            self.assertEqual(
                items[2]["result_summary"]["conflict_code"],
                "invalid_existing_label",
            )

    def test_output_basename_collision_is_rejected_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a" / "same.png"
            second = root / "b" / "same.png"
            output = root / "labels"
            first.parent.mkdir()
            second.parent.mkdir()
            output.mkdir()
            from PIL import Image

            Image.new("RGB", (2, 2)).save(first)
            Image.new("RGB", (2, 2)).save(second)
            with self.assertRaises(Exception) as raised:
                create_standalone_fast_activation_v1(
                    [str(first), str(second)],
                    _options(),
                    output_dir=str(output),
                )
            self.assertEqual(
                getattr(raised.exception, "code", None),
                "conflict_output_path_collision",
            )

    def test_different_sibling_shadow_label_is_rejected_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, labels = _images(root, 1)
            sibling = Path(paths[0]).with_suffix(".json")
            sibling.write_text(
                json.dumps(_valid_empty(Path(paths[0]).name)),
                encoding="utf-8",
            )

            with self.assertRaises(LabelConflictError) as raised:
                create_standalone_fast_activation_v1(
                    paths,
                    _options(),
                    output_dir=str(labels),
                )

            self.assertEqual(raised.exception.code, "conflict_shadow_label")


class FastRunSummaryTests(unittest.TestCase):
    @staticmethod
    def _item(
        status,
        *,
        resolution="not_applicable",
        skip_reason=None,
        error_code=None,
        conflict_code=None,
        zero_target=None,
        review_status="not_applicable",
    ):
        return {
            "execution_status": status,
            "failure_resolution": resolution,
            "result_summary": {
                "skip_reason": skip_reason,
                "error_code": error_code,
                "conflict_code": conflict_code,
                "zero_target": zero_target,
            },
            "review_status": review_status,
        }

    def test_mixed_persistent_facts_produce_exact_summary(self):
        items = [
            self._item("skipped", skip_reason="outside_selected_range"),
            self._item("skipped", skip_reason="existing_annotation"),
            self._item(
                "failed",
                resolution="auto_skip",
                error_code="host_prepare_failed",
            ),
            self._item(
                "succeeded",
                zero_target=True,
                review_status="pending",
            ),
            self._item(
                "failed",
                resolution="auto_skip",
                error_code="failed_input",
            ),
            self._item(
                "failed",
                resolution="unresolved",
                error_code="model_prediction_failed",
            ),
            self._item(
                "skipped",
                resolution="explicit_skip",
                skip_reason="model_error",
            ),
            self._item(
                "conflict",
                resolution="unresolved",
                conflict_code="conflict_image_changed",
            ),
            self._item(
                "conflict",
                resolution="unresolved",
                conflict_code="invalid_existing_label",
            ),
            self._item("queued"),
        ]

        summary = build_fast_run_summary_v1(items)

        self.assertEqual(summary["workset_total"], 10)
        self.assertEqual(summary["selected_by_range"], 9)
        self.assertEqual(summary["eligible_for_inference"], 6)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["zero_target"], 1)
        self.assertEqual(summary["skipped_existing"], 1)
        self.assertEqual(summary["skipped_outside_range"], 1)
        self.assertEqual(summary["host_prepare_failed"], 1)
        self.assertEqual(summary["failed_input"], 1)
        self.assertEqual(summary["model_failed_unresolved"], 1)
        self.assertEqual(summary["explicit_error_skips"], 1)
        self.assertEqual(summary["conflicts"], 2)
        self.assertEqual(summary["pending_review"], 1)
        self.assertEqual(summary["remaining"], 1)
        self.assertTrue(summary["completed_with_errors"])


class FastControllerFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )

    @staticmethod
    def _controller(paths, labels, execute):
        manager = _Manager()
        runner = PredictionRunner(manager, request_executor=execute)
        controller = FastAutoLabelingController(
            runner,
            _options(),
            standalone_image_paths=paths,
            standalone_output_dir=str(labels),
        )
        return controller, runner, manager

    def test_explicit_error_skip_is_terminal_but_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            controller, runner, _manager = self._controller(
                paths,
                labels,
                lambda request, _lease: PredictionOutcome.failed(
                    request, "model_prediction_failed", "oom"
                ),
            )
            finished = []
            controller.finished.connect(finished.append)
            self.assertTrue(controller.start())
            self.assertTrue(
                _wait_until(lambda: controller.phase == "WAITING_ERROR")
            )
            self.assertTrue(controller.skip_current())
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

            item = controller.run_store.list_items(controller.run_id)[0]
            self.assertEqual(item["execution_status"], "skipped")
            self.assertEqual(item["failure_resolution"], "explicit_skip")
            self.assertEqual(finished[0]["explicit_error_skips"], 1)
            self.assertEqual(finished[0]["succeeded"], 0)
            self.assertEqual(finished[0]["processing_status"], "COMPLETED")
            self.assertTrue(finished[0]["completed_with_errors"])

    def test_invalid_label_conflict_does_not_block_later_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            (labels / "image-0000.json").write_text(
                "{broken", encoding="utf-8"
            )
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                return _success(request)

            controller, runner, _manager = self._controller(
                paths, labels, execute
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

            self.assertEqual(calls, [standalone_image_id_v1(paths[1])])
            self.assertEqual(finished[0]["conflicts"], 1)
            self.assertEqual(finished[0]["succeeded"], 1)
            self.assertEqual(finished[0]["processing_status"], "PARTIAL")
            self.assertEqual(
                (labels / "image-0000.json").read_text(encoding="utf-8"),
                "{broken",
            )

    def test_label_commit_failure_stops_before_next_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                return _success(request)

            controller, runner, _manager = self._controller(
                paths, labels, execute
            )
            finished = []
            controller.finished.connect(finished.append)
            with mock.patch(
                "anylabeling.views.labeling.utils.continuous_auto_labeling."
                "commit_label_for_image_v1",
                side_effect=RuntimeError("disk full"),
            ):
                controller.start()
                self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

            self.assertEqual(len(calls), 1)
            self.assertEqual(finished[0]["processing_status"], "FAILED")
            state = controller.run_store.read_state(controller.run_id)
            self.assertEqual(state["processing_status"], "FAILED")
            self.assertEqual(
                state["last_error"]["code"], "commit_handler_failed"
            )

    def test_post_write_reread_failure_stops_before_next_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                return _success(request)

            controller, runner, _manager = self._controller(
                paths, labels, execute
            )
            finished = []
            controller.finished.connect(finished.append)
            with mock.patch(
                "anylabeling.views.labeling.utils.continuous_auto_labeling."
                "commit_label_for_image_v1",
                side_effect=LabelWriteError("post_write_verification_failed"),
            ):
                controller.start()
                self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

            self.assertEqual(len(calls), 1)
            self.assertEqual(finished[0]["processing_status"], "FAILED")
            state = controller.run_store.read_state(controller.run_id)
            self.assertEqual(
                state["last_error"]["code"],
                "post_write_verification_failed",
            )

    def test_success_without_snapshot_digest_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                return PredictionOutcome.succeeded(
                    request,
                    AutoLabelingPayload(
                        [],
                        True,
                        "",
                        input_width=4,
                        input_height=3,
                    ),
                )

            controller, runner, _manager = self._controller(
                paths, labels, execute
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(len(calls), 1)
            self.assertEqual(finished[0]["processing_status"], "FAILED")
            state = controller.run_store.read_state(controller.run_id)
            self.assertEqual(
                state["last_error"]["code"],
                "prediction_input_metadata_missing",
            )
            self.assertFalse(any(labels.iterdir()))

    def test_run_store_checkpoint_failure_stops_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            entered = threading.Event()
            release = threading.Event()
            calls = []

            def execute(request, _lease):
                calls.append(request.image_id)
                entered.set()
                release.wait(5)
                return _success(request)

            controller, runner, _manager = self._controller(
                paths, labels, execute
            )
            finished = []
            controller.finished.connect(finished.append)
            controller.start()
            self.assertTrue(entered.wait(2))
            controller.run_store.update_item = mock.Mock(
                side_effect=RuntimeError("injected item CAS failure")
            )
            release.set()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))

            self.assertEqual(len(calls), 1)
            self.assertEqual(finished[0]["processing_status"], "FAILED")
            state = controller.run_store.read_state(controller.run_id)
            self.assertEqual(state["processing_status"], "FAILED")
            self.assertEqual(
                state["last_error"]["code"], "commit_handler_failed"
            )

    def test_stop_after_unresolved_first_failure_is_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            calls = []
            controller, runner, _manager = self._controller(
                paths,
                labels,
                lambda request, _lease: calls.append(request)
                or PredictionOutcome.failed(
                    request, "model_prediction_failed", "failed"
                ),
            )
            finished = []
            controller.finished.connect(finished.append)
            self.assertTrue(controller.start())
            self.assertTrue(
                _wait_until(lambda: controller.phase == "WAITING_ERROR")
            )
            self.assertTrue(controller.request_stop())
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(len(calls), 1)
            self.assertEqual(finished[0]["processing_status"], "CANCELLED")
            self.assertEqual(finished[0]["model_failed_unresolved"], 1)

    def test_explicit_skip_store_failure_is_terminal_infrastructure_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 1)
            controller, runner, _manager = self._controller(
                paths,
                labels,
                lambda request, _lease: PredictionOutcome.failed(
                    request, "model_prediction_failed", "oom"
                ),
            )
            finished = []
            controller.finished.connect(finished.append)
            self.assertTrue(controller.start())
            self.assertTrue(
                _wait_until(lambda: controller.phase == "WAITING_ERROR")
            )
            controller.run_store.update_item = mock.Mock(
                side_effect=RuntimeError("injected explicit skip failure")
            )

            self.assertFalse(controller.skip_current())
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
            self.assertEqual(finished[0]["processing_status"], "FAILED")
            state = controller.run_store.read_state(controller.run_id)
            self.assertEqual(
                state["last_error"]["code"],
                "explicit_skip_checkpoint_failed",
            )

    def test_close_has_priority_and_waits_for_safe_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, labels = _images(Path(tmp), 2)
            entered = threading.Event()
            release = threading.Event()

            def execute(request, _lease):
                entered.set()
                release.wait(5)
                return _success(request)

            controller, runner, _manager = self._controller(
                paths, labels, execute
            )
            finished = []
            safe = []
            controller.finished.connect(finished.append)
            controller.safe_to_close.connect(safe.append)
            controller.start()
            self.assertTrue(entered.wait(2))
            controller.request_pause()
            controller.request_stop()
            generation = controller.request_close()
            controller.request_pause()
            self.assertEqual(controller.control_intent, "CLOSE")
            self.assertEqual(safe, [])
            release.set()
            self.assertTrue(_wait_until(lambda: bool(finished)))
            self.assertTrue(_wait_until(lambda: safe == [generation]))
            self.assertIsNone(runner.worker_thread)
            self.assertEqual(finished[0]["processing_status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
