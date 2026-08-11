import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from anylabeling.views.labeling.utils.auto_labeling_host import (
    HostContextValidationError,
    resolve_resume_sequence_spec_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    FastSequenceContractError,
    SequenceRunOptionsV1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    create_host_sequence_activation_v1,
)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


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
        "resume_supported": False,
    }


def _options(write_policy="SKIP_EXISTING"):
    return SequenceRunOptionsV1(
        delay_seconds=0.0,
        range="ALL_IMAGES",
        filter="ALL",
        write_policy=write_policy,
        current_anchor_image_id=None,
        workset_source="SESSION_WORKSET",
        model_fingerprint=_fingerprint(),
        parameter_snapshot={"confidence_threshold": 0.25},
    )


class _ItemStore:
    def read_item(self, image_id):
        return {
            "image_id": image_id,
            "revision": 0,
            "staged_document_digest": "MISSING",
            "staged_semantic_digest": "MISSING",
        }

    def recover_commit(
        self,
        image_id,
        current_document_digest,
        current_semantic_digest,
    ):
        return "RETRY", self.read_item(image_id)


class _Sink:
    def prepare(self, event, *, label_path=None):
        return "PREPARED", event, label_path

    def checkpoint(self, event, *, label_path=None):
        return "CHECKPOINTED", event, label_path


class _Lease:
    resource = "annotation_session"
    mode = "write"
    released = False

    def release(self, _reason="completed"):
        self.released = True


def _context(root):
    session_root = root / "labeling_sessions" / "session-a"
    image_root = session_root / "images"
    label_root = session_root / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir()
    image_path = image_root / "image.jpg"
    image_path.write_bytes(b"image")
    canonical_image = _canonical(image_path)
    record = SimpleNamespace(
        image_id="image-a",
        canonical_session_image_path=canonical_image,
        canonical_session_label_path=_canonical(label_root / "image.json"),
        source_image_digest=hashlib.sha256(b"image").hexdigest(),
        source_label_file_sha256="MISSING",
        manifest_sequence=0,
    )
    return SimpleNamespace(
        project_id="project-a",
        active_session_id="session-a",
        image_records_by_path={canonical_image: record},
        annotation_item_store=_ItemStore(),
        annotation_commit_sink=_Sink(),
        audit_client=None,
        model_fingerprint_provider=_fingerprint,
        images_ready=True,
        workset_source="SESSION_WORKSET",
        annotation_session_lease=_Lease(),
        session_purpose="labeling",
        _session_root=_canonical(session_root),
        activate_sequence_run=mock.Mock(
            side_effect=AssertionError("persistent activation is forbidden")
        ),
        resume_sequence_run=mock.Mock(
            side_effect=AssertionError("persistent resume is forbidden")
        ),
    )


class RemovedPersistentResumeTests(unittest.TestCase):
    def test_resume_api_is_an_explicit_tombstone(self):
        with self.assertRaisesRegex(
            HostContextValidationError,
            "persistent_auto_labeling_resume_removed",
        ):
            resolve_resume_sequence_spec_v1(object())

    def test_retired_policies_are_rejected_by_sequence_contract(self):
        for policy in ("INHERIT_MODEL_POLICY", "FORCE_MERGE"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(
                    FastSequenceContractError,
                    "invalid_sequence_write_policy",
                ):
                    _options(policy)

    def test_each_start_builds_a_fresh_process_local_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _context(root)
            first = create_host_sequence_activation_v1(context, _options())
            first_item = first.run_store.read_item(first.run_id, "image-a")
            first.run_store.update_item(
                first.run_id,
                "image-a",
                first_item["item_revision"],
                {"execution_status": "succeeded"},
            )

            second = create_host_sequence_activation_v1(context, _options())

            self.assertNotEqual(first.run_id, second.run_id)
            self.assertIsNot(first.run_store, second.run_store)
            self.assertEqual(
                second.run_store.read_item(second.run_id, "image-a")[
                    "execution_status"
                ],
                "queued",
            )
            context.activate_sequence_run.assert_not_called()
            context.resume_sequence_run.assert_not_called()
            self.assertFalse(
                (root / ".annotation_state" / "auto_labeling_runs").exists()
            )


if __name__ == "__main__":
    unittest.main()
