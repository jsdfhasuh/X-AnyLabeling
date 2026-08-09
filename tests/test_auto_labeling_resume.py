import copy
import unittest
from types import SimpleNamespace
from unittest import mock

from anylabeling.views.labeling.utils.auto_labeling_host import (
    HostContextValidationError,
    resolve_resume_sequence_spec_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    SequenceRunOptionsV1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    ContinuousAutoLabelingController,
    FastControllerError,
)
from anylabeling.views.labeling.widgets.auto_labeling_run_dialog import (
    FastRunUiSession,
)


RUN_ID = "11111111-1111-4111-8111-111111111111"
BINDING_ID = "22222222-2222-4222-8222-222222222222"


def _fingerprint():
    return {
        "fingerprint_schema_version": 1,
        "model_type": "yolov8",
        "config_digest": "a" * 64,
        "artifact_digests": [
            {"field": "model_path", "sha256": "b" * 64, "size": 12}
        ],
        "remote_endpoint_identity": None,
        "adapter_version": "phase4-yolo-v1",
        "resume_supported": True,
    }


def _config():
    return {
        "run_config_schema_version": 1,
        "run_id": RUN_ID,
        "run_kind": "AUTO_LABELING",
        "project_id": "project-a",
        "created_at": "2026-08-09T00:00:00Z",
        "created_by_app_version": "5.0.0",
        "workset_source": "SESSION_WORKSET",
        "workset_digest": "alworkset1:" + "c" * 64,
        "delay_seconds": 2.0,
        "range": "ALL_IMAGES",
        "filter": "ALL",
        "write_policy": "FORCE_MERGE",
        "model_fingerprint": _fingerprint(),
        "parameter_snapshot": {"confidence_threshold": 0.25},
        "label_path_policy": "HOST_CONTEXT",
        "document_digest_schema_version": 1,
        "semantic_digest_schema_version": 1,
        "config_digest": "d" * 64,
    }


def _binding():
    return {
        "binding_schema_version": 1,
        "binding_id": BINDING_ID,
        "project_id": "project-a",
        "run_id": RUN_ID,
        "session_id": "session-resume",
        "attempt_id": "attempt-resume",
        "purpose": "resume_remaining",
        "status": "committed",
        "created_at": "2026-08-09T00:01:00Z",
        "updated_at": "2026-08-09T00:01:01Z",
    }


def _options():
    config = _config()
    return SequenceRunOptionsV1(
        delay_seconds=config["delay_seconds"],
        range=config["range"],
        filter=config["filter"],
        write_policy=config["write_policy"],
        current_anchor_image_id=None,
        workset_source=config["workset_source"],
        model_fingerprint=config["model_fingerprint"],
        parameter_snapshot=config["parameter_snapshot"],
    )


class _Store:
    def __init__(self, config=None, binding=None):
        self.config = copy.deepcopy(config or _config())
        self.binding = copy.deepcopy(binding or _binding())

    def read_config(self, _run_id):
        return copy.deepcopy(self.config)

    def read_queue(self, _run_id):
        return {"entries": []}

    def read_state(self, _run_id):
        return {"binding": copy.deepcopy(self.binding)}

    def read_item(self, _run_id, _image_id):
        return {}

    def list_items(self, _run_id):
        return []

    def update_state(self, _run_id, _revision, _changes):
        return {}

    def update_item(self, _run_id, _image_id, _revision, _changes):
        return {}


def _context(config=None, binding=None):
    store = _Store(config, binding)
    spec = {
        "config": copy.deepcopy(store.config),
        "binding": copy.deepcopy(store.binding),
    }
    return SimpleNamespace(
        project_id="project-a",
        active_session_id="session-resume",
        active_run_id=RUN_ID,
        run_store=store,
        read_active_run_resume_spec=lambda: copy.deepcopy(spec),
        resume_sequence_run=mock.Mock(),
    )


class ResumeSpecTests(unittest.TestCase):
    def test_spec_requires_exact_persisted_config_and_resume_binding(self):
        context = _context()
        with mock.patch(
            "anylabeling.views.labeling.utils.auto_labeling_host."
            "validate_auto_labeling_host_context",
            return_value=context,
        ):
            self.assertEqual(
                resolve_resume_sequence_spec_v1(context),
                {
                    "config": context.run_store.config,
                    "binding": context.run_store.binding,
                },
            )

            context.run_store.config["write_policy"] = "FORCE_REPLACE"
            with self.assertRaisesRegex(
                HostContextValidationError,
                "resume_config_snapshot_mismatch",
            ):
                resolve_resume_sequence_spec_v1(context)

    def test_non_resume_binding_is_rejected(self):
        binding = _binding()
        binding["purpose"] = "review"
        context = _context(binding=binding)
        with mock.patch(
            "anylabeling.views.labeling.utils.auto_labeling_host."
            "validate_auto_labeling_host_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(
                HostContextValidationError,
                "resume_binding_identity_mismatch",
            ):
                resolve_resume_sequence_spec_v1(context)


class ResumeUiTests(unittest.TestCase):
    def _prepare(self, *, fingerprint=None, parameters=None):
        config = _config()
        session = SimpleNamespace()
        context = SimpleNamespace(
            image_records_by_path={
                "image": {
                    "image_id": "image-a",
                    "manifest_sequence": 0,
                }
            }
        )
        capability = SimpleNamespace(
            supports_fast_sequence=True,
            supports_visible_sequence=True,
        )
        with mock.patch(
            "anylabeling.views.labeling.widgets.auto_labeling_run_dialog."
            "resolve_resume_sequence_spec_v1",
            return_value={"config": config, "binding": _binding()},
        ):
            return FastRunUiSession._prepare_resume_options(
                session,
                context,
                ("image",),
                capability,
                fingerprint or _fingerprint(),
                parameters or {"confidence_threshold": 0.25},
            )

    def test_exact_fingerprint_and_parameters_reuse_immutable_options(self):
        options, paths = self._prepare()
        self.assertEqual(paths, ("image",))
        self.assertEqual(options.delay_seconds, 2.0)
        self.assertEqual(options.write_policy, "FORCE_MERGE")
        self.assertEqual(options.execution_mode, "VISIBLE")

    def test_fingerprint_or_parameter_change_fails_before_inference(self):
        changed = _fingerprint()
        changed["config_digest"] = "e" * 64
        with self.assertRaisesRegex(
            FastControllerError,
            "resume_model_fingerprint_mismatch",
        ):
            self._prepare(fingerprint=changed)
        with self.assertRaisesRegex(
            FastControllerError,
            "resume_parameter_snapshot_mismatch",
        ):
            self._prepare(parameters={"confidence_threshold": 0.5})


class ResumeControllerTests(unittest.TestCase):
    def test_existing_run_uses_resume_service_and_same_attempt_identity(self):
        context = _context()
        activation = SimpleNamespace(
            context=context,
            run_id=RUN_ID,
            commit_store=object(),
            annotation_commit_sink=object(),
        )
        context.resume_sequence_run.return_value = activation
        context.image_records_by_path = {"image": {"image_id": "image-a"}}
        context.images_ready = True
        context.activate_sequence_run = mock.Mock()
        controller = SimpleNamespace(
            host_context=context,
            execution_mode="VISIBLE",
            options=_options(),
            created_by_app_version="5.0.0",
            context_replacer=None,
            _resume_spec=None,
            _recover_resumed_items=mock.Mock(),
            _load_queue=mock.Mock(),
            _activate_presenter=mock.Mock(),
        )
        with mock.patch(
            "anylabeling.views.labeling.utils.continuous_auto_labeling."
            "resolve_resume_sequence_spec_v1",
            return_value={"config": _config(), "binding": _binding()},
        ):
            ContinuousAutoLabelingController._activate_run(controller)

        request = context.resume_sequence_run.call_args.args[0]
        self.assertEqual(request["run_id"], RUN_ID)
        self.assertEqual(request["session_attempt_id"], "attempt-resume")
        context.activate_sequence_run.assert_not_called()
        controller._recover_resumed_items.assert_called_once_with()

    def test_queue_skips_resolved_and_retries_unresolved_failure(self):
        items = {
            "done": {
                "image_id": "done",
                "execution_status": "succeeded",
                "failure_resolution": "not_applicable",
            },
            "retry": {
                "image_id": "retry",
                "execution_status": "failed",
                "failure_resolution": "unresolved",
            },
        }
        controller = SimpleNamespace(
            _finished=False,
            active_request=None,
            runner=SimpleNamespace(is_idle=lambda: True),
            queue_position=0,
            queue_entries=[{"image_id": "done"}, {"image_id": "retry"}],
            run_id=RUN_ID,
            run_store=SimpleNamespace(
                read_item=lambda _run_id, image_id: items[image_id]
            ),
            _dispatch_item=mock.Mock(),
            _finalize=mock.Mock(),
            _hard_fail=mock.Mock(),
        )
        ContinuousAutoLabelingController._dispatch_next(controller)
        controller._dispatch_item.assert_called_once_with(items["retry"])
        controller._finalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
