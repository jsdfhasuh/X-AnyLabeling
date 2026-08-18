import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui

from anylabeling.views.labeling.utils.auto_labeling_commit import (
    ANNOTATION_PRESENCE_INVALID,
    ANNOTATION_PRESENCE_VALID_EMPTY,
    AtomicWriteResultV1,
    ContractValidationError,
    ImageInputSnapshotError,
    LabelConflictError,
    LabelWriteError,
    PredictionOutcomeError,
    atomic_write_label_document,
    commit_label_for_image_v1,
    compose_final_label_document,
    decode_image_input_snapshot_v1,
    recover_image_commit_v1,
    remap_incoming_group_ids_v1,
    resolve_existing_label,
    validate_output_paths_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_run_store import (
    FaultInjectionCommitStoreV1,
    InMemoryAnnotationCommitSinkV1,
    InMemoryCommitStoreV1,
    InjectedCommitCrash,
    StoreConflictError,
    build_annotation_commit_event_v1,
    build_annotation_commit_event_v2,
    validate_annotation_commit_event_v2,
)


def _shape(label="box", group_id=None, shape_type="rectangle"):
    points = (
        [[1, 1], [5, 1], [5, 4], [1, 4]]
        if shape_type == "rectangle"
        else [[2, 2]]
    )
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "shape_type": shape_type,
        "flags": {},
        "description": "",
        "attributes": {},
        "kie_linking": [],
    }


def _document(shapes=None, description="human", **extra):
    document = {
        "version": "3.3.7",
        "flags": {"verified": True},
        "shapes": copy.deepcopy(shapes or []),
        "imagePath": "sample.png",
        "imageData": None,
        "imageHeight": 6,
        "imageWidth": 8,
        "description": description,
    }
    document.update(copy.deepcopy(extra))
    return document


def _prediction(shapes, replace, description=""):
    return {
        "status": "succeeded",
        "shapes": copy.deepcopy(shapes),
        "replace": replace,
        "description": description,
        "zero_target": len(shapes) == 0,
        "error_code": None,
    }


def _write_json(path, document, indent=None):
    Path(path).write_text(
        json.dumps(document, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


class LabelCompositionTruthTableTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "sample.png"
        self.label_path = self.root / "sample.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _existing(self, shapes=None, description="human", **extra):
        _write_json(
            self.label_path,
            _document(shapes, description=description, **extra),
        )
        return resolve_existing_label(self.label_path)

    def _missing(self):
        self.label_path.unlink(missing_ok=True)
        return resolve_existing_label(self.label_path)

    def _compose(self, existing, prediction, policy, **kwargs):
        return compose_final_label_document(
            existing,
            prediction,
            policy,
            label_path=self.label_path,
            image_path=self.image_path,
            image_height=6,
            image_width=8,
            **kwargs,
        )

    def test_missing_label_creates_nonempty_or_valid_empty_for_every_policy(
        self,
    ):
        cases = (
            ("SKIP_EXISTING", True),
            ("FORCE_REPLACE", False),
        )
        for policy, replace in cases:
            with self.subTest(policy=policy, replace=replace, targets=1):
                result = self._compose(
                    self._missing(),
                    _prediction([_shape("new")], replace),
                    policy,
                )
                self.assertEqual(result.action, "write")
                self.assertEqual(
                    [shape["label"] for shape in result.document["shapes"]],
                    ["new"],
                )
                self.assertFalse(result.zero_target)
            with self.subTest(policy=policy, replace=replace, targets=0):
                result = self._compose(
                    self._missing(), _prediction([], replace), policy
                )
                self.assertEqual(result.action, "write")
                self.assertEqual(result.document["shapes"], [])
                self.assertTrue(result.zero_target)

    def test_valid_empty_is_an_existing_negative_sample_for_skip(self):
        existing = self._existing([])
        self.assertEqual(existing.presence, ANNOTATION_PRESENCE_VALID_EMPTY)
        result = self._compose(
            existing,
            _prediction([_shape("new")], True),
            "SKIP_EXISTING",
        )
        self.assertEqual(result.action, "skipped")
        self.assertEqual(result.skip_reason, "existing_annotation")
        self.assertEqual(result.document["shapes"], [])

    def test_valid_nonempty_is_skipped_without_composition(self):
        existing = self._existing([_shape("old")])
        result = self._compose(
            existing, _prediction([], True), "SKIP_EXISTING"
        )
        self.assertEqual(result.action, "skipped")
        self.assertEqual(
            [shape["label"] for shape in result.document["shapes"]], ["old"]
        )

    def test_replace_truth_table_including_zero_target(self):
        for shapes in ([], [_shape("old")]):
            with self.subTest(existing_count=len(shapes), targets=1):
                result = self._compose(
                    self._existing(shapes),
                    _prediction([_shape("new")], False),
                    "FORCE_REPLACE",
                )
                self.assertEqual(
                    [shape["label"] for shape in result.document["shapes"]],
                    ["new"],
                )
            with self.subTest(existing_count=len(shapes), targets=0):
                result = self._compose(
                    self._existing(shapes),
                    _prediction([], False),
                    "FORCE_REPLACE",
                )
                self.assertEqual(result.document["shapes"], [])
                self.assertTrue(result.zero_target)

    def test_retired_merge_and_inherit_policies_are_rejected(self):
        existing = self._existing([_shape("old")])
        for policy in ("INHERIT_MODEL_POLICY", "FORCE_MERGE"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(
                    ValueError, "unsupported write policy"
                ):
                    self._compose(
                        existing,
                        _prediction([_shape("new")], True),
                        policy,
                    )

    def test_corrupt_label_is_never_overwritten_by_any_policy(self):
        self.label_path.write_text("{not-json", encoding="utf-8")
        existing = resolve_existing_label(self.label_path)
        self.assertEqual(existing.presence, ANNOTATION_PRESENCE_INVALID)
        for policy in (
            "SKIP_EXISTING",
            "FORCE_REPLACE",
        ):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(
                    LabelConflictError, "invalid_existing_label"
                ):
                    self._compose(
                        existing, _prediction([_shape()], True), policy
                    )
        self.assertEqual(
            self.label_path.read_text(encoding="utf-8"), "{not-json"
        )

    def test_none_and_failed_outcomes_are_not_zero_target_success(self):
        with self.assertRaisesRegex(
            PredictionOutcomeError, "invalid_prediction_outcome"
        ):
            self._compose(self._missing(), None, "FORCE_REPLACE")
        failed = {
            "status": "failed",
            "shapes": None,
            "replace": None,
            "description": "",
            "zero_target": False,
            "error_code": "model_failed",
        }
        with self.assertRaisesRegex(PredictionOutcomeError, "model_failed"):
            self._compose(self._missing(), failed, "FORCE_REPLACE")
        success = self._compose(
            self._missing(), _prediction([], True), "FORCE_REPLACE"
        )
        self.assertTrue(success.zero_target)
        self.assertEqual(success.document["shapes"], [])

    def test_description_replace_and_empty_rules_are_exact(self):
        existing = self._existing([_shape("old")], description="human")
        cases = (
            ("FORCE_REPLACE", "", "human"),
            ("FORCE_REPLACE", "model", "model"),
        )
        for policy, incoming, expected in cases:
            with self.subTest(policy=policy, incoming=incoming):
                result = self._compose(
                    existing,
                    _prediction(
                        [_shape("new")], policy == "FORCE_REPLACE", incoming
                    ),
                    policy,
                )
                self.assertEqual(result.document["description"], expected)

    def test_flags_unknown_fields_and_incoming_shape_order_are_preserved(
        self,
    ):
        existing = self._existing(
            [_shape("first"), _shape("second")],
            vendor={"camera": "a"},
        )
        result = self._compose(
            existing,
            _prediction([_shape("third")], False),
            "FORCE_REPLACE",
            flags={"ignored": True},
            other_data={"new_vendor_field": 7},
        )
        self.assertEqual(result.document["flags"], {"verified": True})
        self.assertEqual(result.document["vendor"], {"camera": "a"})
        self.assertEqual(result.document["new_vendor_field"], 7)
        self.assertEqual(
            [shape["label"] for shape in result.document["shapes"]],
            ["third"],
        )
        self.assertEqual(result.document["imagePath"], "sample.png")
        self.assertIsNone(result.document["imageData"])
        self.assertEqual(result.document["imageHeight"], 6)
        self.assertEqual(result.document["imageWidth"], 8)

    def test_illegal_nan_infinity_numpy_and_qpointf_inputs_are_rejected(self):
        values = (
            float("nan"),
            float("inf"),
            np.float32(1.5),
            QtCore.QPointF(1, 2),
        )
        for value in values:
            incoming = _shape("bad")
            incoming["points"] = [[value, 1]]
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ContractValidationError):
                    self._compose(
                        self._missing(),
                        _prediction([incoming], True),
                        "FORCE_REPLACE",
                    )

    def test_composition_does_not_mutate_existing_or_prediction_objects(self):
        existing_document = _document([_shape("old", 7)])
        _write_json(self.label_path, existing_document)
        existing = resolve_existing_label(self.label_path)
        prediction = _prediction([_shape("new", "7")], False, "model")
        existing_before = copy.deepcopy(existing.document)
        prediction_before = copy.deepcopy(prediction)
        self._compose(existing, prediction, "FORCE_REPLACE")
        self.assertEqual(existing.document, existing_before)
        self.assertEqual(prediction, prediction_before)


class PoseGroupRemappingTests(unittest.TestCase):
    pose_config = {"classes": {"person": ["nose", "eye"]}}

    @staticmethod
    def _pose(group_id, keypoints=("nose",)):
        return [_shape("person", group_id)] + [
            _shape(label, group_id, "point") for label in keypoints
        ]

    def test_existing_and_typed_incoming_ids_map_to_distinct_new_integers(
        self,
    ):
        existing = [_shape("old", 5)]
        incoming = self._pose(1) + self._pose("1")
        existing_before = copy.deepcopy(existing)
        incoming_before = copy.deepcopy(incoming)
        remapped = remap_incoming_group_ids_v1(
            existing, incoming, pose_config=self.pose_config
        )
        self.assertEqual([shape["group_id"] for shape in remapped[:2]], [6, 6])
        self.assertEqual([shape["group_id"] for shape in remapped[2:]], [7, 7])
        self.assertEqual(existing, existing_before)
        self.assertEqual(incoming, incoming_before)

    def test_multiple_pose_instances_keep_bbox_and_keypoints_together(self):
        incoming = self._pose(10, ("nose", "eye")) + self._pose(
            11, ("nose", "eye")
        )
        remapped = remap_incoming_group_ids_v1(
            [], incoming, pose_config=self.pose_config
        )
        self.assertEqual(
            [shape["group_id"] for shape in remapped], [0, 0, 0, 1, 1, 1]
        )

    def test_none_group_is_unchanged(self):
        incoming = [_shape("ungrouped", None)]
        remapped = remap_incoming_group_ids_v1([], incoming)
        self.assertIsNone(remapped[0]["group_id"])

    def test_unsupported_group_id_types_have_stable_conflict(self):
        for group_id in (True, 1.5, [], {}, ""):
            with self.subTest(group_id=group_id):
                with self.assertRaisesRegex(
                    LabelConflictError, "conflict_unsupported_group_id"
                ):
                    remap_incoming_group_ids_v1([], [_shape("bad", group_id)])

    def test_group_id_allocator_overflow_is_a_conflict(self):
        existing = [_shape("old", (2**31) - 1)]
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_group_id_exhausted"
        ):
            remap_incoming_group_ids_v1(existing, [_shape("new", 1)])

    def test_nonempty_kie_linking_requires_a_dedicated_adapter(self):
        incoming = self._pose(1)
        incoming[0]["kie_linking"] = [[1, 2]]
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_unsupported_kie_merge"
        ):
            remap_incoming_group_ids_v1(
                [], incoming, pose_config=self.pose_config
            )

    def test_pose_structure_requires_one_bbox_and_valid_unique_keypoints(self):
        invalid_groups = (
            [_shape("nose", 1, "point")],
            [_shape("person", 1), _shape("person", 1)],
            self._pose(1) + [_shape("unknown", 1, "point")],
            self._pose(1) + [_shape("nose", 1, "point")],
            [_shape("person", 1), _shape("line", 1, "line")],
        )
        for incoming in invalid_groups:
            with self.subTest(incoming=incoming):
                with self.assertRaisesRegex(
                    LabelConflictError, "invalid_pose_group_structure"
                ):
                    remap_incoming_group_ids_v1(
                        [], incoming, pose_config=self.pose_config
                    )


class AtomicLabelWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "sample.png"
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(self.image_path)
        self.label_path = self.root / "sample.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_atomic_replace_fsync_reread_and_input_immutability(self):
        document = _document([_shape("new")])
        before = copy.deepcopy(document)
        with (
            mock.patch(
                "anylabeling.views.labeling.utils.auto_labeling_commit.os.replace",
                wraps=os.replace,
            ) as replace,
            mock.patch(
                "anylabeling.views.labeling.utils.auto_labeling_commit.os.fsync",
                wraps=os.fsync,
            ) as fsync,
        ):
            result = atomic_write_label_document(
                self.label_path,
                document,
                pre_document_digest="MISSING",
                allowed_root=self.root,
                image_source_path=self.image_path,
            )
        self.assertIsInstance(result, AtomicWriteResultV1)
        self.assertEqual(document, before)
        self.assertEqual(
            resolve_existing_label(self.label_path).document_digest,
            result.document_digest,
        )
        replace.assert_called_once()
        fsync.assert_called_once()
        self.assertEqual(list(self.root.glob(".auto-labeling-*.tmp")), [])

    def test_pre_document_digest_mismatch_refuses_overwrite(self):
        _write_json(self.label_path, _document([_shape("old")]))
        stale = resolve_existing_label(self.label_path)
        external = _document([_shape("external")])
        _write_json(self.label_path, external)
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_pre_document_digest"
        ):
            atomic_write_label_document(
                self.label_path,
                _document([_shape("intended")]),
                pre_document_digest=stale.document_digest,
                allowed_root=self.root,
                image_source_path=self.image_path,
            )
        self.assertEqual(
            resolve_existing_label(self.label_path).document["shapes"],
            external["shapes"],
        )

    def test_corrupt_current_document_is_not_overwritten(self):
        _write_json(self.label_path, _document([_shape("old")]))
        stale = resolve_existing_label(self.label_path)
        self.label_path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(
            LabelConflictError, "invalid_existing_label"
        ):
            atomic_write_label_document(
                self.label_path,
                _document([_shape("intended")]),
                pre_document_digest=stale.document_digest,
                allowed_root=self.root,
                image_source_path=self.image_path,
            )
        self.assertEqual(
            self.label_path.read_text(encoding="utf-8"), "{broken"
        )

    def test_replace_failure_propagates_and_preserves_current_file(self):
        old = _document([_shape("old")])
        _write_json(self.label_path, old)
        current = resolve_existing_label(self.label_path)
        with mock.patch(
            "anylabeling.views.labeling.utils.auto_labeling_commit.os.replace",
            side_effect=OSError("disk failure"),
        ):
            with self.assertRaisesRegex(LabelWriteError, "label_write_failed"):
                atomic_write_label_document(
                    self.label_path,
                    _document([_shape("new")]),
                    pre_document_digest=current.document_digest,
                    allowed_root=self.root,
                    image_source_path=self.image_path,
                )
        self.assertEqual(
            [
                shape["label"]
                for shape in resolve_existing_label(self.label_path).document[
                    "shapes"
                ]
            ],
            ["old"],
        )

    def test_output_path_collision_preflight(self):
        first_dir = self.root / "a"
        second_dir = self.root / "b"
        first_dir.mkdir()
        second_dir.mkdir()
        entries = [
            {
                "image_path": self.image_path,
                "label_path": first_dir / "sample.json",
            },
            {
                "image_path": self.root / "other.png",
                "label_path": second_dir / "sample.json",
            },
        ]
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_output_basename_collision"
        ):
            validate_output_paths_v1(
                entries,
                allowed_root=self.root,
                require_unique_basename=True,
            )
        duplicate = [entries[0], dict(entries[0])]
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_output_path_collision"
        ):
            validate_output_paths_v1(duplicate, allowed_root=self.root)

    def test_samefile_and_image_alias_collisions_are_rejected(self):
        first = self.root / "first.json"
        alias = self.root / "alias.json"
        _write_json(first, _document([]))
        try:
            os.link(first, alias)
        except OSError as exc:
            self.skipTest(f"hard links are unavailable: {exc}")
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_output_samefile_collision"
        ):
            validate_output_paths_v1(
                [
                    {"image_path": self.image_path, "label_path": first},
                    {
                        "image_path": self.root / "other.png",
                        "label_path": alias,
                    },
                ],
                allowed_root=self.root,
            )
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_output_image_collision"
        ):
            validate_output_paths_v1(
                [
                    {
                        "image_path": self.image_path,
                        "label_path": self.image_path,
                    }
                ],
                allowed_root=self.root,
            )

    def test_symlink_output_and_symlink_escape_are_rejected_or_skipped_explicitly(
        self,
    ):
        target = self.root / "target.json"
        link = self.root / "link.json"
        _write_json(target, _document([]))
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symbolic links are unavailable: {exc}")
        with self.assertRaisesRegex(
            LabelConflictError, "conflict_output_symlink"
        ):
            validate_output_paths_v1(
                [{"image_path": self.image_path, "label_path": link}],
                allowed_root=self.root,
            )
        outside = Path(self.temp_dir.name).parent / f"outside-{os.getpid()}"
        outside.mkdir(exist_ok=True)
        symlink_dir = self.root / "redirect"
        try:
            os.symlink(outside, symlink_dir, target_is_directory=True)
            with self.assertRaisesRegex(
                LabelConflictError, "conflict_output_path_escape"
            ):
                validate_output_paths_v1(
                    [
                        {
                            "image_path": self.image_path,
                            "label_path": symlink_dir / "escaped.json",
                        }
                    ],
                    allowed_root=self.root,
                )
        finally:
            try:
                outside.rmdir()
            except OSError:
                pass


class ImageInputSnapshotDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "target.png"
        Image.new("RGB", (4, 3), color=(200, 10, 20)).save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_target_is_read_once_and_canvas_residue_is_not_used(self):
        residual_canvas_image = QtGui.QImage(4, 3, QtGui.QImage.Format_RGB888)
        residual_canvas_image.fill(QtGui.QColor(0, 0, 255))
        with mock.patch("builtins.open", wraps=open) as opened:
            snapshot = decode_image_input_snapshot_v1(
                image_id="image-a",
                requested_path=str(self.image_path),
                source="STANDALONE_FILE_LIST",
            )
        target_reads = [
            call
            for call in opened.call_args_list
            if call.args and os.fspath(call.args[0]) == str(self.image_path)
        ]
        self.assertEqual(len(target_reads), 1)
        self.assertEqual(snapshot.pixel_format, "RGB888")
        self.assertEqual(snapshot.to_rgb_array()[0, 0].tolist(), [200, 10, 20])
        self.assertNotEqual(
            snapshot.to_qimage().pixelColor(0, 0),
            residual_canvas_image.pixelColor(0, 0),
        )
        Image.new("RGB", (4, 3), color=(0, 255, 0)).save(self.image_path)
        self.assertEqual(snapshot.to_rgb_array()[0, 0].tolist(), [200, 10, 20])

    def test_expected_sha_mismatch_is_a_stable_conflict(self):
        with self.assertRaisesRegex(
            ImageInputSnapshotError, "conflict_image_changed"
        ):
            decode_image_input_snapshot_v1(
                image_id="image-a",
                requested_path=str(self.image_path),
                source="STANDALONE_FILE_LIST",
                expected_sha256="0" * 64,
            )

    def test_non_normalized_exif_is_rejected(self):
        path = self.root / "oriented.jpg"
        image = Image.new("RGB", (4, 3), color=(1, 2, 3))
        exif = Image.Exif()
        exif[0x0112] = 6
        image.save(path, exif=exif)
        with self.assertRaisesRegex(
            ImageInputSnapshotError, "exif_not_normalized"
        ):
            decode_image_input_snapshot_v1(
                image_id="image-a",
                requested_path=str(path),
                source="STANDALONE_FILE_LIST",
            )

    def test_embedded_path_must_be_regular_contained_and_not_a_symlink(self):
        session_root = self.root / "session-images"
        session_root.mkdir()
        inside = session_root / "inside.png"
        Image.new("RGB", (2, 2), color=(1, 2, 3)).save(inside)
        snapshot = decode_image_input_snapshot_v1(
            image_id="inside",
            requested_path=str(inside),
            source="SESSION_MANIFEST",
            session_images_root=str(session_root),
        )
        self.assertEqual(snapshot.decoded_width, 2)
        outside = self.root / "outside.png"
        Image.new("RGB", (2, 2), color=(3, 2, 1)).save(outside)
        with self.assertRaisesRegex(
            ImageInputSnapshotError, "session_image_path_escape"
        ):
            decode_image_input_snapshot_v1(
                image_id="outside",
                requested_path=str(outside),
                source="SESSION_MANIFEST",
                session_images_root=str(session_root),
            )
        link = session_root / "link.png"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symbolic links are unavailable: {exc}")
        with self.assertRaisesRegex(
            ImageInputSnapshotError,
            "session_image_path_escape|session_image_symlink",
        ):
            decode_image_input_snapshot_v1(
                image_id="link",
                requested_path=str(link),
                source="SESSION_MANIFEST",
                session_images_root=str(session_root),
            )

    def test_relative_path_and_non_file_are_rejected(self):
        with self.assertRaisesRegex(
            ImageInputSnapshotError, "image_path_not_absolute"
        ):
            decode_image_input_snapshot_v1(
                image_id="relative",
                requested_path="target.png",
                source="STANDALONE_FILE_LIST",
            )
        with self.assertRaisesRegex(ImageInputSnapshotError, "failed_input"):
            decode_image_input_snapshot_v1(
                image_id="directory",
                requested_path=str(self.root),
                source="STANDALONE_FILE_LIST",
            )


class CommitProtocolAndRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "sample.png"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(self.image_path)
        self.label_path = self.root / "sample.json"
        self.initial_document = _document([_shape("old")])
        self.outcome = _prediction([_shape("new")], False, "model")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _store(self, crash_point=None):
        store = FaultInjectionCommitStoreV1(crash_point)
        store.create_item("image-a", "attempt-a")
        return store

    def _commit(self, store, event_sink=None, **kwargs):
        return commit_label_for_image_v1(
            store=store,
            image_id="image-a",
            attempt_id="attempt-a",
            label_path=self.label_path,
            image_path=self.image_path,
            image_height=6,
            image_width=8,
            prediction_outcome=self.outcome,
            write_policy="FORCE_REPLACE",
            allowed_root=self.root,
            event_sink=event_sink,
            project_id="project-a",
            session_id="session-a",
            run_id="run-a",
            source_image_digest="1" * 64,
            **kwargs,
        )

    def _labels(self):
        resolution = resolve_existing_label(self.label_path)
        if resolution.document is None:
            return []
        return [shape["label"] for shape in resolution.document["shapes"]]

    def test_commit_intent_and_atomic_checkpoint_fields_are_exact(self):
        _write_json(self.label_path, self.initial_document)
        store = self._store("after_intent_before_label_write")
        with self.assertRaises(InjectedCommitCrash):
            self._commit(store)
        item = store.read_item("image-a")
        self.assertEqual(item["staged_commit_status"], "prepared")
        self.assertEqual(
            set(item["commit_intent"]),
            {
                "attempt_id",
                "pre_file_sha256",
                "pre_document_digest",
                "pre_semantic_digest",
                "intended_document_digest",
                "intended_semantic_digest",
                "phase",
            },
        )
        store.crash_point = None
        recovery = recover_image_commit_v1(
            store=store, image_id="image-a", label_path=self.label_path
        )
        self.assertEqual(recovery.action, "RETURNED_TO_QUEUE")
        completed = self._commit(store)
        item = completed.item
        self.assertEqual(item["staged_commit_status"], "committed")
        self.assertEqual(item["source_commit_status"], "pending")
        self.assertIsNone(item["commit_intent"])
        self.assertEqual(
            item["digests"]["staged_document_digest"],
            completed.write_result.document_digest,
        )
        self.assertEqual(
            item["digests"]["staged_annotation_digest"],
            completed.write_result.semantic_digest,
        )
        self.assertNotIn("staged_semantic_digest", item["digests"])
        self.assertNotIn("staged_file_sha256", item["digests"])

    def test_every_commit_crash_point_has_a_deterministic_recovery(self):
        cases = (
            ("before_intent", "RETURNED_TO_QUEUE", ["old"]),
            (
                "after_intent_before_label_write",
                "RETURNED_TO_QUEUE",
                ["old"],
            ),
            (
                "after_label_replace_before_checkpoint",
                "COMPLETED_CHECKPOINT",
                ["new"],
            ),
            (
                "after_checkpoint_before_summary",
                "CHECKPOINT_PRESENT",
                ["new"],
            ),
        )
        for point, expected_action, expected_labels in cases:
            with self.subTest(point=point):
                _write_json(self.label_path, self.initial_document)
                store = self._store(point)
                with self.assertRaises(InjectedCommitCrash):
                    self._commit(store)
                recovery = recover_image_commit_v1(
                    store=store,
                    image_id="image-a",
                    label_path=self.label_path,
                )
                self.assertEqual(recovery.action, expected_action)
                self.assertEqual(self._labels(), expected_labels)

    def test_replace_recovery_and_reentry_never_rewrites_twice(self):
        _write_json(self.label_path, self.initial_document)
        store = self._store("after_label_replace_before_checkpoint")
        with self.assertRaises(InjectedCommitCrash):
            self._commit(store)
        recovery = recover_image_commit_v1(
            store=store, image_id="image-a", label_path=self.label_path
        )
        self.assertEqual(recovery.action, "COMPLETED_CHECKPOINT")
        store.crash_point = None
        with mock.patch(
            "anylabeling.views.labeling.utils.auto_labeling_commit.os.replace",
            wraps=os.replace,
        ) as replace:
            repeated = self._commit(store)
        self.assertEqual(repeated.composition.action, "already_committed")
        replace.assert_not_called()
        self.assertEqual(self._labels(), ["new"])

    def test_v2_checkpoint_response_loss_recovers_from_session_receipt(self):
        _write_json(self.label_path, self.initial_document)
        existing = resolve_existing_label(self.label_path)
        authority_item = {
            "revision": 0,
            "staged_document_digest": existing.document_digest,
            "staged_semantic_digest": existing.semantic_digest,
        }
        authority_store = SimpleNamespace(
            read_item=mock.Mock(return_value=authority_item),
            recover_commit=mock.Mock(
                return_value=("CHECKPOINTED", {"revision": 2})
            ),
        )
        sink = SimpleNamespace(
            prepare=mock.Mock(return_value=("PREPARED", {"revision": 1})),
            checkpoint=mock.Mock(side_effect=RuntimeError("response lost")),
        )
        store = self._store()

        with self.assertRaisesRegex(RuntimeError, "response lost"):
            self._commit(
                store,
                event_sink=sink,
                annotation_item_store=authority_store,
                model_fingerprint={"model_sha256": "a" * 64},
            )

        repeated = self._commit(
            store,
            event_sink=sink,
            annotation_item_store=authority_store,
            model_fingerprint={"model_sha256": "a" * 64},
        )
        self.assertEqual(repeated.composition.action, "already_committed")
        authority_store.recover_commit.assert_called_once_with(
            "image-a",
            repeated.composition.document_digest,
            repeated.composition.semantic_digest,
        )
        self.assertEqual(sink.prepare.call_count, 1)
        self.assertEqual(sink.checkpoint.call_count, 1)
        self.assertEqual(self._labels(), ["new"])

    def test_recovery_uses_only_document_digest_not_raw_formatting(self):
        _write_json(self.label_path, self.initial_document)
        store = self._store("after_intent_before_label_write")
        with self.assertRaises(InjectedCommitCrash):
            self._commit(store)
        self.label_path.write_text(
            json.dumps(self.initial_document, separators=(",", ":")),
            encoding="utf-8",
        )
        recovery = recover_image_commit_v1(
            store=store, image_id="image-a", label_path=self.label_path
        )
        self.assertEqual(recovery.action, "RETURNED_TO_QUEUE")

        _write_json(self.label_path, self.initial_document)
        store = self._store("after_label_replace_before_checkpoint")
        with self.assertRaises(InjectedCommitCrash):
            self._commit(store)
        intended = resolve_existing_label(self.label_path).document
        self.label_path.write_text(
            json.dumps(intended, separators=(",", ":")), encoding="utf-8"
        )
        recovery = recover_image_commit_v1(
            store=store, image_id="image-a", label_path=self.label_path
        )
        self.assertEqual(recovery.action, "COMPLETED_CHECKPOINT")

    def test_recovery_conflicts_when_current_is_neither_pre_nor_intended(self):
        _write_json(self.label_path, self.initial_document)
        store = self._store("after_intent_before_label_write")
        with self.assertRaises(InjectedCommitCrash):
            self._commit(store)
        _write_json(self.label_path, _document([_shape("external")]))
        recovery = recover_image_commit_v1(
            store=store, image_id="image-a", label_path=self.label_path
        )
        self.assertEqual(recovery.action, "CONFLICT")
        self.assertEqual(recovery.item["execution_status"], "conflict")
        self.assertEqual(
            recovery.item["conflict"]["code"],
            "conflict_label_changed_during_commit",
        )

    def test_writer_failure_propagates_and_leaves_prepared_evidence(self):
        _write_json(self.label_path, self.initial_document)
        store = self._store()
        with mock.patch(
            "anylabeling.views.labeling.utils.auto_labeling_commit.os.replace",
            side_effect=OSError("disk failure"),
        ):
            with self.assertRaises(LabelWriteError):
                self._commit(store)
        item = store.read_item("image-a")
        self.assertEqual(item["staged_commit_status"], "prepared")
        self.assertIsNotNone(item["commit_intent"])
        self.assertEqual(self._labels(), ["old"])

    def test_fast_and_visible_callers_share_identical_pure_commit_results(
        self,
    ):
        results = []
        for mode in ("fast", "visible"):
            directory = self.root / mode
            directory.mkdir()
            image_path = directory / "sample.png"
            label_path = directory / "sample.json"
            Image.new("RGB", (8, 6), color=(20, 30, 40)).save(image_path)
            _write_json(label_path, self.initial_document)
            store = InMemoryCommitStoreV1()
            store.create_item("image-a", "attempt-a")
            result = commit_label_for_image_v1(
                store=store,
                image_id="image-a",
                attempt_id="attempt-a",
                label_path=label_path,
                image_path=image_path,
                image_height=6,
                image_width=8,
                prediction_outcome=self.outcome,
                write_policy="FORCE_REPLACE",
                allowed_root=directory,
            )
            results.append(result)
        self.assertEqual(
            results[0].write_result.document_digest,
            results[1].write_result.document_digest,
        )
        self.assertEqual(
            results[0].write_result.semantic_digest,
            results[1].write_result.semantic_digest,
        )
        self.assertEqual(
            results[0].write_result.document,
            results[1].write_result.document,
        )

    def test_continuous_event_is_notify_only_and_idempotent(self):
        _write_json(self.label_path, self.initial_document)
        store = self._store()
        sink = InMemoryAnnotationCommitSinkV1(store)
        result = self._commit(store, event_sink=sink)
        item = store.read_item("image-a")
        self.assertEqual(item["staged_commit_status"], "committed")
        self.assertEqual(item["source_commit_status"], "pending")
        first_revision = item["item_revision"]
        self.assertEqual(sink.publish(result.event), "IDEMPOTENT_NO_OP")
        self.assertEqual(
            store.read_item("image-a")["item_revision"], first_revision
        )
        self.assertEqual(self._labels(), ["new"])


class AnnotationCommitEventSinkTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.label_path = self.root / "sample.json"
        self.store = InMemoryCommitStoreV1()
        self.store.create_item("image-a", "attempt-a")
        self.sink = InMemoryAnnotationCommitSinkV1(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _manual_event(self, writer_kind, document_digest, semantic_digest):
        return build_annotation_commit_event_v1(
            project_id="project-a",
            session_id="session-a",
            run_id="run-a",
            image_id="image-a",
            attempt_id="attempt-a",
            writer_kind=writer_kind,
            commit_scope="STAGED",
            mutation_mode="APPLY_MANUAL_REVISION",
            document_digest=document_digest,
            semantic_digest=semantic_digest,
            source_image_digest="1" * 64,
            base_item_revision=0,
            created_at="2026-08-07T00:00:00Z",
        )

    def test_manual_save_auto_save_and_delete_use_manual_sink(self):
        _write_json(self.label_path, _document([_shape("manual")]))
        current = resolve_existing_label(self.label_path)
        manual = self._manual_event(
            "MANUAL_SAVE", current.document_digest, current.semantic_digest
        )
        self.assertEqual(
            self.sink.publish(manual, label_path=self.label_path),
            "APPLIED_MANUAL_REVISION",
        )
        self.assertEqual(
            self.sink.publish(manual, label_path=self.label_path),
            "IDEMPOTENT_NO_OP",
        )
        auto = self._manual_event(
            "AUTO_SAVE", current.document_digest, current.semantic_digest
        )
        self.assertEqual(
            self.sink.publish(auto, label_path=self.label_path),
            "IDEMPOTENT_NO_OP",
        )
        self.label_path.unlink()
        delete = self._manual_event("DELETE_LABEL", "MISSING", "MISSING")
        self.assertEqual(
            self.sink.publish(delete, label_path=self.label_path),
            "APPLIED_MANUAL_REVISION",
        )
        item = self.store.read_item("image-a")
        self.assertEqual(item["digests"]["staged_document_digest"], "MISSING")
        self.assertEqual(
            item["digests"]["staged_annotation_digest"], "MISSING"
        )

    def test_manual_sink_rereads_disk_and_rejects_stale_event(self):
        first = _document([_shape("first")])
        second = _document([_shape("second")])
        _write_json(self.label_path, first)
        stale = resolve_existing_label(self.label_path)
        _write_json(self.label_path, second)
        event = self._manual_event(
            "MANUAL_SAVE", stale.document_digest, stale.semantic_digest
        )
        with self.assertRaisesRegex(StoreConflictError, "stale_commit_event"):
            self.sink.publish(event, label_path=self.label_path)

    def test_new_prediction_attempt_does_not_reuse_manual_commit_checkpoint(
        self,
    ):
        image_path = self.root / "sample.png"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(image_path)
        _write_json(self.label_path, _document([_shape("manual")]))
        current = resolve_existing_label(self.label_path)
        manual = self._manual_event(
            "MANUAL_SAVE", current.document_digest, current.semantic_digest
        )
        self.assertEqual(
            self.sink.publish(manual, label_path=self.label_path),
            "APPLIED_MANUAL_REVISION",
        )

        started = self.store.begin_attempt("image-a", "attempt-b")
        self.assertEqual(started["execution_status"], "running")
        result = commit_label_for_image_v1(
            store=self.store,
            image_id="image-a",
            attempt_id="attempt-b",
            label_path=self.label_path,
            image_path=image_path,
            image_height=6,
            image_width=8,
            prediction_outcome=_prediction([_shape("predicted")], True),
            write_policy="FORCE_REPLACE",
            allowed_root=self.root,
        )

        self.assertEqual(result.composition.action, "write")
        self.assertEqual(
            [
                shape["label"]
                for shape in result.write_result.document["shapes"]
            ],
            ["predicted"],
        )
        self.assertEqual(
            self.store.read_item("image-a")["execution_status"],
            "succeeded",
        )

    def test_event_builder_enforces_writer_combination_matrix(self):
        with self.assertRaisesRegex(
            ContractValidationError, "commit_event_writer_combination"
        ):
            build_annotation_commit_event_v1(
                project_id="project-a",
                session_id="session-a",
                run_id="run-a",
                image_id="image-a",
                attempt_id="attempt-a",
                writer_kind="CONTINUOUS",
                commit_scope="STAGED",
                mutation_mode="APPLY_MANUAL_REVISION",
                document_digest="aldoc1:" + ("1" * 64),
                semantic_digest="alsem1:" + ("2" * 64),
                created_at="2026-08-07T00:00:00Z",
            )


class AnnotationCommitEventV2Tests(unittest.TestCase):
    def _event(self, **changes):
        values = {
            "project_id": "project-α",
            "session_id": "session-a",
            "image_id": "图像-01",
            "base_session_item_revision": 7,
            "writer": "AUTOMATIC",
            "pre_document_digest": "MISSING",
            "pre_semantic_digest": "MISSING",
            "intended_document_digest": "aldoc1:" + "1" * 64,
            "intended_semantic_digest": "alsem1:" + "2" * 64,
            "model_fingerprint": {
                "adapter": "yolo",
                "sha256": "3" * 64,
            },
            "created_at": "2026-08-11T00:00:00Z",
        }
        values.update(changes)
        return build_annotation_commit_event_v2(**values)

    def test_cross_repository_golden_event_id(self):
        event = self._event()
        self.assertEqual(
            event["event_id"],
            "6f9e428dc0d7a5ec3f3837dfd8f2acdba672f4f003fc821e6a12f0019a6ce932",
        )
        self.assertIs(validate_annotation_commit_event_v2(event), event)
        self.assertNotIn("run_id", event)
        self.assertNotIn("attempt_id", event)

    def test_writer_matrix_and_credential_free_fingerprint(self):
        for writer in ("MANUAL_SAVE", "AUTO_SAVE"):
            with self.subTest(writer=writer):
                event = self._event(writer=writer, model_fingerprint=None)
                self.assertEqual(event["writer"], writer)
        delete = self._event(
            writer="DELETE_LABEL",
            intended_document_digest="MISSING",
            intended_semantic_digest="MISSING",
            model_fingerprint=None,
        )
        self.assertEqual(delete["writer"], "DELETE_LABEL")
        for changes in (
            {"writer": "MANUAL_SAVE"},
            {
                "writer": "DELETE_LABEL",
                "model_fingerprint": None,
            },
            {"model_fingerprint": {"api_token": "secret"}},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(StoreConflictError):
                    self._event(**changes)


if __name__ == "__main__":
    unittest.main()
