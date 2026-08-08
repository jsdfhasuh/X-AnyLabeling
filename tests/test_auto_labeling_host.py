import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.auto_labeling_host import (
    AutoLabelingHostContextProtocol,
    HostContextValidationError,
    clear_auto_labeling_host_context,
    get_auto_labeling_host_context,
    resolve_model_fingerprint,
    set_auto_labeling_host_context,
    validate_auto_labeling_host_context,
    validate_model_fingerprint_payload,
)
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)


class _Store:
    def read_config(self, run_id):
        return {"run_id": run_id}

    def read_queue(self, run_id):
        return {"run_id": run_id}

    def read_state(self, run_id):
        return {"run_id": run_id}

    def read_item(self, run_id, image_id):
        return {"run_id": run_id, "image_id": image_id}

    def list_items(self, run_id):
        return []

    def update_state(self, run_id, expected_state_revision, changes):
        return changes

    def update_item(
        self,
        run_id,
        image_id,
        expected_item_revision,
        changes,
    ):
        return changes


class _Sink:
    def publish(self, event, *, label_path=None):
        return event, label_path


class _Lease:
    resource = "annotation_session"
    mode = "write"
    released = False

    def release(self, _reason="completed"):
        self.released = True


def _context(root):
    image_root = root / "labeling_sessions" / "session-a" / "images"
    label_root = root / "labeling_sessions" / "session-a" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir()
    image_path = os.path.normcase(os.path.realpath(image_root / "image.jpg"))
    label_path = os.path.normcase(os.path.realpath(label_root / "image.json"))
    Path(image_path).write_bytes(b"image")
    record = SimpleNamespace(
        image_id="image-a",
        canonical_session_image_path=image_path,
        canonical_session_label_path=label_path,
        source_image_digest="a" * 64,
        source_label_file_sha256="MISSING",
        manifest_sequence=0,
    )
    return SimpleNamespace(
        project_id="project-a",
        active_session_id="session-a",
        active_run_id=None,
        image_records_by_path={image_path: record},
        run_store=_Store(),
        model_fingerprint_provider=lambda: {
            "fingerprint_schema_version": 1,
            "model_digest": "b" * 64,
        },
        annotation_commit_sink=_Sink(),
        images_ready=False,
        workset_source="SESSION_WORKSET",
        annotation_session_lease=_Lease(),
    )


class AutoLabelingHostProtocolTests(unittest.TestCase):
    def tearDown(self):
        clear_auto_labeling_host_context()

    def test_parent_style_object_satisfies_protocol_without_parent_import(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            context = _context(Path(tmp))

            self.assertIs(
                validate_auto_labeling_host_context(context), context
            )
            self.assertIsInstance(context, AutoLabelingHostContextProtocol)
            source = Path(
                __import__(
                    "anylabeling.views.labeling.utils.auto_labeling_host",
                    fromlist=["__file__"],
                ).__file__
            ).read_text(encoding="utf-8")
            self.assertNotIn("project__func", source)
            self.assertNotIn("labeling_session", source)

    def test_set_clear_and_standalone_fallback_are_identity_safe(self):
        self.assertIsNone(get_auto_labeling_host_context())
        with tempfile.TemporaryDirectory() as tmp:
            context = _context(Path(tmp))
            other = SimpleNamespace()

            self.assertIs(set_auto_labeling_host_context(context), context)
            clear_auto_labeling_host_context(other)
            self.assertIs(get_auto_labeling_host_context(), context)
            clear_auto_labeling_host_context(context)
            self.assertIsNone(get_auto_labeling_host_context())

    def test_record_identity_digest_and_lease_validation_are_strict(self):
        mutations = (
            (
                lambda context: setattr(
                    next(iter(context.image_records_by_path.values())),
                    "source_image_digest",
                    "bad",
                ),
                "invalid_host_source_image_digest",
            ),
            (
                lambda context: setattr(
                    next(iter(context.image_records_by_path.values())),
                    "source_label_file_sha256",
                    "bad",
                ),
                "invalid_host_source_label_digest",
            ),
            (
                lambda context: setattr(
                    context.annotation_session_lease,
                    "released",
                    True,
                ),
                "invalid_annotation_session_lease",
            ),
        )
        for mutate, error in mutations:
            with (
                self.subTest(error=error),
                tempfile.TemporaryDirectory() as tmp,
            ):
                context = _context(Path(tmp))
                mutate(context)
                with self.assertRaisesRegex(HostContextValidationError, error):
                    validate_auto_labeling_host_context(context)

    def test_duplicate_ids_labels_sequences_and_mapping_keys_are_rejected(
        self,
    ):
        cases = (
            ("image_id", "duplicate_host_image_id"),
            ("canonical_session_label_path", "duplicate_host_label_path"),
            ("manifest_sequence", "duplicate_host_manifest_sequence"),
        )
        for field, error in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                context = _context(root)
                first = next(iter(context.image_records_by_path.values()))
                second_image = (
                    root
                    / "labeling_sessions"
                    / "session-a"
                    / "images"
                    / "two.jpg"
                )
                second_label = (
                    root
                    / "labeling_sessions"
                    / "session-a"
                    / "labels"
                    / "two.json"
                )
                second_image.write_bytes(b"two")
                second = copy.copy(first)
                second.image_id = "image-b"
                second.canonical_session_image_path = os.path.normcase(
                    os.path.realpath(second_image)
                )
                second.canonical_session_label_path = os.path.normcase(
                    os.path.realpath(second_label)
                )
                second.manifest_sequence = 1
                setattr(second, field, getattr(first, field))
                context.image_records_by_path[
                    second.canonical_session_image_path
                ] = second
                with self.assertRaisesRegex(HostContextValidationError, error):
                    validate_auto_labeling_host_context(context)

        with tempfile.TemporaryDirectory() as tmp:
            context = _context(Path(tmp))
            key, record = next(iter(context.image_records_by_path.items()))
            context.image_records_by_path = {key + ".other": record}
            with self.assertRaisesRegex(
                HostContextValidationError,
                "host_record_mapping_key_mismatch",
            ):
                validate_auto_labeling_host_context(context)

    def test_model_fingerprint_provider_rejects_credentials(self):
        safe = {
            "fingerprint_schema_version": 1,
            "source": "https://models.example.test/model.onnx",
        }
        self.assertEqual(validate_model_fingerprint_payload(safe), safe)
        unsafe = (
            {"api_key": "secret"},
            {"source": "https://user:pass@example.test/model"},
            {"source": "https://example.test/model?token=secret"},
        )
        for payload in unsafe:
            with self.subTest(payload=payload):
                with self.assertRaises(HostContextValidationError):
                    validate_model_fingerprint_payload(payload)

        with tempfile.TemporaryDirectory() as tmp:
            context = _context(Path(tmp))
            self.assertEqual(
                resolve_model_fingerprint(context),
                {
                    "fingerprint_schema_version": 1,
                    "model_digest": "b" * 64,
                },
            )
            context.model_fingerprint_provider = lambda: {"token": "secret"}
            with self.assertRaises(HostContextValidationError):
                resolve_model_fingerprint(context)
            context.model_fingerprint_provider = None
            with self.assertRaisesRegex(
                HostContextValidationError,
                "model_fingerprint_unavailable",
            ):
                resolve_model_fingerprint(context)

    def test_labeling_and_auto_widgets_forward_and_clear_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = _context(Path(tmp))
            auto_widget = SimpleNamespace(
                set_auto_labeling_host_context=mock.Mock(),
                clear_auto_labeling_host_context=mock.Mock(),
            )
            labeling_widget = SimpleNamespace(
                auto_labeling_host_context=None,
                auto_labeling_widget=auto_widget,
            )

            result = LabelingWidget.set_auto_labeling_host_context(
                labeling_widget,
                context,
            )
            self.assertIs(result, context)
            auto_widget.set_auto_labeling_host_context.assert_called_once_with(
                context
            )
            LabelingWidget.clear_auto_labeling_host_context(labeling_widget)
            self.assertIsNone(labeling_widget.auto_labeling_host_context)
            auto_widget.clear_auto_labeling_host_context.assert_called_once_with()

            holder = SimpleNamespace(auto_labeling_host_context=None)
            self.assertIs(
                AutoLabelingWidget.set_auto_labeling_host_context(
                    holder,
                    context,
                ),
                context,
            )
            AutoLabelingWidget.clear_auto_labeling_host_context(holder)
            self.assertIsNone(holder.auto_labeling_host_context)


if __name__ == "__main__":
    unittest.main()
