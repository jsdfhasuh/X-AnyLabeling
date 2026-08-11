import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from PyQt5 import QtWidgets

from anylabeling.views.labeling.label_file import LabelFile, LabelFileError
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.auto_labeling_commit_bridge import (
    AnnotationCommitBridgeError,
    AnnotationCommitBridgeV1,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    atomic_write_label_document,
    resolve_existing_label,
)
from anylabeling.views.labeling.utils.auto_labeling_run_store import (
    InMemoryAnnotationCommitSinkV1,
    InMemoryCommitStoreV1,
)


def _document(description=""):
    return {
        "version": "3.3.7",
        "flags": {},
        "shapes": [],
        "imagePath": "image.png",
        "imageData": None,
        "imageHeight": 6,
        "imageWidth": 8,
        "description": description,
    }


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


class _HostStore:
    def __init__(self):
        self.item = {"item_revision": 7, "latest_attempt_id": "attempt-a"}

    def read_item(self, run_id, image_id):
        if run_id != "run-a" or image_id != "image-a":
            raise AssertionError("unexpected identity")
        return dict(self.item)


class _ImageItemStore:
    def __init__(self, label_path):
        self.label_path = label_path

    def read_item(self, image_id):
        current = resolve_existing_label(self.label_path)
        return {
            "image_id": image_id,
            "revision": 4,
            "staged_document_digest": current.document_digest,
            "staged_semantic_digest": current.semantic_digest,
        }

    def recover_commit(
        self,
        image_id,
        current_document_digest,
        current_semantic_digest,
    ):
        return "RETRY", self.read_item(image_id)


class _StandaloneClient:
    def __init__(self, record, store, sink):
        self.record = record
        self.store = store
        self.sink = sink
        self.integrity_refresh = mock.Mock()

    def commit_binding(self, image_path):
        if os.path.normcase(os.path.realpath(image_path)) != (
            self.record.canonical_session_image_path
        ):
            return None
        item = self.store.read_item("run-a", self.record.image_id)
        return {
            "project_id": "project-a",
            "session_id": "session-a",
            "run_id": "run-a",
            "image_id": self.record.image_id,
            "attempt_id": item.get("latest_attempt_id"),
            "item_revision": item["item_revision"],
            "label_path": self.record.canonical_session_label_path,
            "source_image_digest": self.record.source_image_digest,
            "sink": self.sink,
        }


class AnnotationCommitBridgeTests(unittest.TestCase):
    def test_bound_session_rejects_image_deletion(self):
        widget = SimpleNamespace(
            auto_labeling_audit_active=False,
            auto_labeling_host_context=SimpleNamespace(
                active_session_id="session-a"
            ),
        )
        self.assertFalse(LabelingWidget.delete_image_file(widget))

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "image.png"
        self.label_path = self.root / "image.json"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(self.image_path)
        _write(self.label_path, _document())
        self.sink = SimpleNamespace(publish=mock.Mock(return_value="APPLIED"))
        self.record = SimpleNamespace(
            image_id="image-a",
            canonical_session_image_path=os.path.normcase(
                os.path.realpath(self.image_path)
            ),
            canonical_session_label_path=os.path.normcase(
                os.path.realpath(self.label_path)
            ),
            source_image_digest="a" * 64,
        )
        self.standalone_client = _StandaloneClient(
            self.record,
            _HostStore(),
            self.sink,
        )
        self.widget = SimpleNamespace(
            auto_labeling_host_context=None,
            _standalone_auto_labeling_audit_client=self.standalone_client,
            _sequence_presentation_session=None,
            filename=str(self.image_path),
            auto_labeling_commit_blocked=False,
            auto_labeling_commit_error=None,
            dirty=False,
        )
        self.bridge = AnnotationCommitBridgeV1(self.widget)
        self.bridge.record_loaded_label(self.label_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _v2_bridge(self):
        sink = mock.Mock()
        sink.prepare.return_value = "PREPARED"
        sink.checkpoint.return_value = "CHECKPOINTED"
        context = SimpleNamespace(
            project_id="project-a",
            active_session_id="session-a",
            image_records_by_path={
                self.record.canonical_session_image_path: self.record
            },
            annotation_item_store=_ImageItemStore(self.label_path),
            annotation_commit_sink=sink,
        )
        self.widget.auto_labeling_host_context = context
        bridge = AnnotationCommitBridgeV1(self.widget)
        bridge.record_loaded_label(self.label_path)
        return bridge, sink

    def test_v2_manual_auto_save_and_delete_use_prepare_then_checkpoint(self):
        for writer in ("MANUAL_SAVE", "AUTO_SAVE"):
            with self.subTest(writer=writer):
                _write(self.label_path, _document())
                bridge, sink = self._v2_bridge()
                current = resolve_existing_label(self.label_path)
                document = _document(f"saved by {writer}")
                bridge.prepare_saved_label(
                    writer,
                    self.image_path,
                    self.label_path,
                    document,
                    current,
                )
                atomic_write_label_document(
                    self.label_path,
                    document,
                    pre_document_digest=current.document_digest,
                    allowed_root=self.root,
                )
                self.assertEqual(
                    bridge.publish_saved_label(
                        writer,
                        self.image_path,
                        self.label_path,
                    ),
                    "CHECKPOINTED",
                )
                self.assertEqual(
                    [call[0] for call in sink.method_calls],
                    ["prepare", "checkpoint"],
                )
                event = sink.prepare.call_args.args[0]
                self.assertEqual(event["event_schema_version"], 2)
                self.assertEqual(event["writer"], writer)
                self.assertNotIn("run_id", event)

        _write(self.label_path, _document())
        bridge, sink = self._v2_bridge()
        self.assertEqual(
            bridge.delete_label(self.image_path, self.label_path),
            "CHECKPOINTED",
        )
        self.assertFalse(self.label_path.exists())
        self.assertEqual(
            [call[0] for call in sink.method_calls],
            ["prepare", "checkpoint"],
        )
        self.assertEqual(
            sink.prepare.call_args.args[0]["writer"], "DELETE_LABEL"
        )

    def _configure_save_surface(self, *, auto_save=False):
        save_action = SimpleNamespace(setEnabled=mock.Mock())
        self.widget.actions = SimpleNamespace(
            undo=SimpleNamespace(setEnabled=mock.Mock()),
            save=save_action,
        )
        self.widget.canvas = SimpleNamespace(is_shape_restorable=False)
        self.widget._config = {
            "auto_save": auto_save,
            "store_data": False,
        }
        self.widget.image_path = str(self.image_path)
        self.widget.output_dir = None
        self.widget.output_file = None
        self.widget.image_data = None
        self.widget.image = SimpleNamespace(
            isNull=mock.Mock(return_value=False),
            height=mock.Mock(return_value=6),
            width=mock.Mock(return_value=8),
        )
        self.widget.label_list = []
        self.widget.flag_widget = SimpleNamespace(
            count=mock.Mock(return_value=0)
        )
        self.widget.other_data = {"description": "saved through widget"}
        self.widget.file_list_widget = SimpleNamespace(
            findItems=mock.Mock(return_value=[])
        )
        self.widget.error_message = mock.Mock()
        self.widget.tr = lambda value: value
        self.widget.annotation_commit_bridge = self.bridge
        self.widget.add_recent_file = mock.Mock()

        def set_clean():
            self.widget.dirty = False

        self.widget.set_clean = mock.Mock(side_effect=set_clean)
        self.widget.save_labels = lambda filename, writer_kind="MANUAL_SAVE": (
            LabelingWidget.save_labels(
                self.widget,
                filename,
                writer_kind,
            )
        )
        self.widget._save_file = lambda filename: LabelingWidget._save_file(
            self.widget,
            filename,
        )
        return save_action

    def test_manual_and_auto_save_publish_one_reread_event(self):
        self.assertEqual(
            self.bridge.authoritative_label_path(self.image_path),
            os.path.normcase(os.path.realpath(self.label_path)),
        )
        self.assertEqual(
            self.bridge.publish_saved_label(
                "MANUAL_SAVE",
                self.image_path,
                self.label_path,
            ),
            "APPLIED",
        )
        event = self.sink.publish.call_args.args[0]
        self.assertEqual(event["writer_kind"], "MANUAL_SAVE")
        self.assertEqual(event["commit_scope"], "STAGED")
        self.assertEqual(event["mutation_mode"], "APPLY_MANUAL_REVISION")
        self.assertEqual(event["base_item_revision"], 7)
        self.assertEqual(self.sink.publish.call_count, 1)

        self.bridge.publish_saved_label(
            "AUTO_SAVE",
            self.image_path,
            self.label_path,
        )
        self.assertEqual(
            self.sink.publish.call_args.args[0]["writer_kind"],
            "AUTO_SAVE",
        )
        self.assertEqual(self.sink.publish.call_count, 2)

    def test_presentation_never_emits_manual_event(self):
        self.widget._sequence_presentation_session = {"token": "active"}
        self.assertEqual(
            self.bridge.publish_saved_label(
                "MANUAL_SAVE",
                self.image_path,
                self.label_path,
            ),
            "PRESENTATION_NO_EVENT",
        )
        self.sink.publish.assert_not_called()

    def test_sink_failure_preserves_disk_and_sets_integrity_blocker(self):
        changed = _document("saved to disk")
        _write(self.label_path, changed)
        self.sink.publish.side_effect = RuntimeError("checkpoint failed")
        with self.assertRaisesRegex(
            AnnotationCommitBridgeError,
            "manual_commit_sink_failed",
        ):
            self.bridge.publish_saved_label(
                "MANUAL_SAVE",
                self.image_path,
                self.label_path,
            )
        self.assertEqual(json.loads(self.label_path.read_text()), changed)
        self.assertTrue(self.widget.auto_labeling_commit_blocked)
        self.assertEqual(
            self.widget.auto_labeling_commit_error["code"],
            "manual_commit_sink_failed",
        )
        self.assertTrue(self.widget.dirty)
        pending = self.bridge.pending_commit
        current = resolve_existing_label(self.label_path)
        self.assertEqual(
            pending["event"]["event_id"],
            self.widget.auto_labeling_commit_error["event_id"],
        )
        self.assertEqual(pending["raw_file_sha256"], current.raw_file_sha256)
        self.assertEqual(pending["document_digest"], current.document_digest)
        self.assertEqual(pending["semantic_digest"], current.semantic_digest)

    def test_integrity_refresh_replays_original_event_without_audit_refresh(
        self,
    ):
        _write(self.label_path, _document("saved to disk"))
        self.sink.publish.side_effect = [
            RuntimeError("response lost"),
            "IDEMPOTENT_NO_OP",
        ]
        with self.assertRaises(AnnotationCommitBridgeError):
            self.bridge.publish_saved_label(
                "MANUAL_SAVE",
                self.image_path,
                self.label_path,
            )
        pending = self.bridge.pending_commit
        first_event = self.sink.publish.call_args_list[0].args[0]

        self.assertEqual(
            self.bridge.integrity_refresh(),
            "IDEMPOTENT_NO_OP",
        )

        replayed_event = self.sink.publish.call_args_list[1].args[0]
        self.assertEqual(replayed_event, first_event)
        self.assertEqual(
            replayed_event["event_id"], pending["event"]["event_id"]
        )
        self.assertIsNone(self.bridge.pending_commit)
        self.assertFalse(self.widget.auto_labeling_commit_blocked)
        self.assertIsNone(self.widget.auto_labeling_commit_error)
        self.standalone_client.integrity_refresh.assert_not_called()

    def test_response_loss_replay_is_exactly_once_in_commit_store(self):
        store = InMemoryCommitStoreV1()
        store.create_item("image-a", "attempt-a")
        delegate = InMemoryAnnotationCommitSinkV1(store)

        class ResponseLossSink:
            def __init__(self):
                self.first = True

            def publish(inner_self, event, *, label_path=None):
                result = delegate.publish(event, label_path=label_path)
                if inner_self.first:
                    inner_self.first = False
                    raise RuntimeError("response lost after commit")
                return result

        self.standalone_client.sink = ResponseLossSink()
        _write(self.label_path, _document("saved to store"))
        with self.assertRaises(AnnotationCommitBridgeError):
            self.bridge.publish_saved_label(
                "MANUAL_SAVE",
                self.image_path,
                self.label_path,
            )
        event_id = self.bridge.pending_commit["event"]["event_id"]

        self.assertEqual(
            self.bridge.integrity_refresh(),
            "IDEMPOTENT_NO_OP",
        )

        item = store.read_item("image-a")
        self.assertEqual(item["last_commit_event_id"], event_id)
        self.assertEqual(item["recent_commit_event_ids"].count(event_id), 1)

    def test_integrity_refresh_rejects_changed_disk_and_keeps_pending(self):
        _write(self.label_path, _document("saved to disk"))
        self.sink.publish.side_effect = RuntimeError("checkpoint failed")
        with self.assertRaises(AnnotationCommitBridgeError):
            self.bridge.publish_saved_label(
                "MANUAL_SAVE",
                self.image_path,
                self.label_path,
            )
        pending = self.bridge.pending_commit
        _write(self.label_path, _document("external rewrite"))
        self.sink.publish.side_effect = None

        with self.assertRaisesRegex(
            AnnotationCommitBridgeError,
            "manual_commit_pending_disk_mismatch",
        ):
            self.bridge.integrity_refresh()

        self.assertEqual(self.sink.publish.call_count, 1)
        self.assertEqual(self.bridge.pending_commit, pending)
        self.assertTrue(self.widget.auto_labeling_commit_blocked)

    def test_ctrl_s_retry_replays_pending_before_new_save(self):
        self._configure_save_surface()
        self.widget.dirty = True
        self.sink.publish.side_effect = [
            RuntimeError("response lost"),
            "IDEMPOTENT_NO_OP",
            "APPLIED",
        ]

        self.assertFalse(LabelingWidget.save_file(self.widget))
        self.assertTrue(self.widget.dirty)
        self.assertTrue(self.widget.auto_labeling_commit_blocked)
        self.widget.set_clean.assert_not_called()

        self.assertTrue(LabelingWidget.save_file(self.widget))

        first_event = self.sink.publish.call_args_list[0].args[0]
        replayed_event = self.sink.publish.call_args_list[1].args[0]
        self.assertEqual(replayed_event, first_event)
        self.assertEqual(replayed_event["event_id"], first_event["event_id"])
        self.assertEqual(self.sink.publish.call_count, 3)
        self.widget.set_clean.assert_called_once_with()
        self.assertFalse(self.widget.dirty)
        self.assertFalse(self.widget.auto_labeling_commit_blocked)

    def test_auto_save_failure_stays_dirty_until_original_event_replays(self):
        self._configure_save_surface(auto_save=True)
        self.widget.dirty = False
        self.sink.publish.side_effect = [
            RuntimeError("response lost"),
            "IDEMPOTENT_NO_OP",
        ]

        LabelingWidget.set_dirty(self.widget)

        self.assertTrue(self.widget.dirty)
        self.assertTrue(self.widget.auto_labeling_commit_blocked)
        self.widget.set_clean.assert_not_called()
        first_event = self.sink.publish.call_args_list[0].args[0]
        self.assertEqual(first_event["writer_kind"], "AUTO_SAVE")

        self.assertEqual(
            self.bridge.integrity_refresh(),
            "IDEMPOTENT_NO_OP",
        )

        self.assertEqual(
            self.sink.publish.call_args_list[1].args[0],
            first_event,
        )
        self.assertFalse(self.widget.auto_labeling_commit_blocked)
        self.assertTrue(self.widget.dirty)

    def test_delete_is_cas_guarded_and_publishes_missing(self):
        self.assertEqual(
            self.bridge.delete_label(self.image_path, self.label_path),
            "APPLIED",
        )
        self.assertFalse(self.label_path.exists())
        event = self.sink.publish.call_args.args[0]
        self.assertEqual(event["writer_kind"], "DELETE_LABEL")
        self.assertEqual(event["document_digest"], "MISSING")
        self.assertEqual(event["semantic_digest"], "MISSING")

    def test_delete_failure_replays_original_missing_event(self):
        self.sink.publish.side_effect = [
            RuntimeError("response lost"),
            "IDEMPOTENT_NO_OP",
        ]
        with self.assertRaises(AnnotationCommitBridgeError):
            self.bridge.delete_label(self.image_path, self.label_path)

        self.assertFalse(self.label_path.exists())
        self.assertTrue(self.widget.dirty)
        self.assertTrue(self.widget.auto_labeling_commit_blocked)
        pending = self.bridge.pending_commit
        self.assertEqual(pending["raw_file_sha256"], "MISSING")
        self.assertEqual(pending["document_digest"], "MISSING")
        self.assertEqual(pending["semantic_digest"], "MISSING")
        first_event = self.sink.publish.call_args_list[0].args[0]

        self.assertEqual(
            self.bridge.integrity_refresh(),
            "IDEMPOTENT_NO_OP",
        )

        replayed_event = self.sink.publish.call_args_list[1].args[0]
        self.assertEqual(replayed_event, first_event)
        self.assertEqual(replayed_event["writer_kind"], "DELETE_LABEL")
        self.assertEqual(replayed_event["document_digest"], "MISSING")
        self.assertFalse(self.widget.auto_labeling_commit_blocked)
        self.assertIsNone(self.bridge.pending_commit)

    def test_delete_menu_finishes_only_after_missing_event_replay(self):
        self.sink.publish.side_effect = [
            RuntimeError("response lost"),
            "IDEMPOTENT_NO_OP",
        ]
        item = SimpleNamespace(setCheckState=mock.Mock())
        self.widget.annotation_commit_bridge = self.bridge
        self.widget._config = {"keep_prev": False}
        self.widget.tr = lambda value: value
        self.widget.get_label_file = lambda: str(self.label_path)
        self.widget.error_message = mock.Mock()
        self.widget.file_list_widget = SimpleNamespace(
            currentItem=mock.Mock(return_value=item)
        )
        self.widget.reset_state = mock.Mock()
        self.widget.load_file = mock.Mock(return_value=True)

        with mock.patch(
            "anylabeling.views.labeling.label_widget."
            "QtWidgets.QMessageBox.warning",
            return_value=QtWidgets.QMessageBox.Yes,
        ):
            self.assertFalse(LabelingWidget.delete_file(self.widget))
            self.assertFalse(self.label_path.exists())
            self.widget.reset_state.assert_not_called()

            self.assertTrue(LabelingWidget.delete_file(self.widget))

        self.assertEqual(
            self.sink.publish.call_args_list[0].args[0],
            self.sink.publish.call_args_list[1].args[0],
        )
        self.widget.reset_state.assert_called_once_with()
        self.widget.load_file.assert_called_once_with(str(self.image_path))

    def test_external_change_after_load_is_not_overwritten(self):
        loaded_digest = self.bridge.pre_document_digest(self.label_path)
        _write(self.label_path, _document("external"))
        label_file = LabelFile()
        with self.assertRaisesRegex(
            LabelFileError, "conflict_pre_document_digest"
        ):
            label_file.save(
                filename=self.label_path,
                shapes=[],
                image_path="image.png",
                image_height=6,
                image_width=8,
                image_data=None,
                other_data={"description": "ours"},
                flags={},
                pre_document_digest=loaded_digest,
            )
        self.assertEqual(
            json.loads(self.label_path.read_text())["description"],
            "external",
        )


class LabelingWidgetSaveRoutingTests(unittest.TestCase):
    def test_auto_save_marks_writer_kind_and_manual_save_returns_success(self):
        undo = SimpleNamespace(setEnabled=mock.Mock())
        widget = SimpleNamespace(
            actions=SimpleNamespace(undo=undo),
            canvas=SimpleNamespace(is_shape_restorable=False),
            _config={"auto_save": True},
            image_path="C:/images/image.png",
            output_dir=None,
            save_labels=mock.Mock(return_value=True),
            set_clean=mock.Mock(),
            filename="C:/images/image.png",
        )
        LabelingWidget.set_dirty(widget)
        widget.save_labels.assert_called_once_with(
            "C:/images/image.json",
            writer_kind="AUTO_SAVE",
        )
        widget.set_clean.assert_called_once_with()

        manual = SimpleNamespace(
            save_labels=mock.Mock(return_value=True),
            add_recent_file=mock.Mock(),
            set_clean=mock.Mock(),
        )
        self.assertTrue(LabelingWidget._save_file(manual, "image.json"))
        manual.save_labels.assert_called_once_with("image.json")

    def test_auto_save_failure_keeps_dirty_and_does_not_set_clean(self):
        save_action = SimpleNamespace(setEnabled=mock.Mock())
        widget = SimpleNamespace(
            actions=SimpleNamespace(
                undo=SimpleNamespace(setEnabled=mock.Mock()),
                save=save_action,
            ),
            canvas=SimpleNamespace(is_shape_restorable=False),
            _config={"auto_save": True},
            image_path="C:/images/image.png",
            output_dir=None,
            save_labels=mock.Mock(return_value=False),
            set_clean=mock.Mock(),
            filename="C:/images/image.png",
            dirty=False,
        )

        LabelingWidget.set_dirty(widget)

        self.assertTrue(widget.dirty)
        save_action.setEnabled.assert_called_once_with(True)
        widget.set_clean.assert_not_called()

    def test_menu_save_failure_returns_false_and_never_sets_clean(self):
        manual = SimpleNamespace(
            save_labels=mock.Mock(return_value=False),
            add_recent_file=mock.Mock(),
            set_clean=mock.Mock(),
        )

        self.assertFalse(LabelingWidget._save_file(manual, "image.json"))
        manual.add_recent_file.assert_not_called()
        manual.set_clean.assert_not_called()

    def test_bound_save_uses_authoritative_path_without_save_as_dialog(self):
        image = SimpleNamespace(isNull=mock.Mock(return_value=False))
        bridge = SimpleNamespace(
            authoritative_label_path=mock.Mock(
                return_value="C:/session/labels/image.json"
            )
        )
        widget = SimpleNamespace(
            image=image,
            filename="C:/session/images/image.png",
            annotation_commit_bridge=bridge,
            _save_file=mock.Mock(return_value=True),
            label_file=None,
            output_file=None,
        )

        self.assertTrue(LabelingWidget.save_file(widget))
        bridge.authoritative_label_path.assert_called_once_with(
            widget.filename
        )
        widget._save_file.assert_called_once_with(
            "C:/session/labels/image.json"
        )


if __name__ == "__main__":
    unittest.main()
