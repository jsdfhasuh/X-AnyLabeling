import base64
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from anylabeling.views.labeling.utils.auto_labeling_contracts import (
    ANNOTATION_COMMIT_EVENT_V1_SCHEMA,
    ANNOTATION_PRESENCE_INVALID,
    ANNOTATION_PRESENCE_MISSING,
    ANNOTATION_PRESENCE_VALID_EMPTY,
    ANNOTATION_PRESENCE_VALID_NONEMPTY,
    IMAGE_INPUT_SNAPSHOT_V1_SCHEMA,
    WORKSET_SNAPSHOT_V1_SCHEMA,
    ContractValidationError,
    annotation_commit_event_id_v1,
    canonical_document_digest_v1,
    canonical_path_identity,
    canonicalize_annotation_document_v1,
    classify_annotation_document_v1,
    create_session_workset_snapshot_v1,
    raw_file_sha256_v1,
    select_workset_entries_v1,
    semantic_annotation_digest_v1,
    validate_annotation_commit_event_v1,
    validate_image_input_snapshot_v1,
    validate_workset_snapshot_v1,
    workset_digest_v1,
)


UTC_NOW = "2026-08-07T12:34:56Z"
SHA_A = "a" * 64
SHA_B = "b" * 64
WORKSET_GOLDEN = (
    "alworkset1:20d89441e263ed7d335568720ba35eac"
    "7d529ef75825c0b579aaa60fe8ff16d6"
)


def _label_document():
    return {
        "version": "3.3.7",
        "flags": {"verified": True},
        "shapes": [
            {
                "label": "robot\\arm",
                "points": [[1, -0.0], [2.0, 3.5]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        ],
        "imagePath": ".\\nested\\..\\image.jpg",
        "imageData": base64.b64encode(b"pixels").decode("ascii"),
        "imageHeight": 80,
        "imageWidth": 100,
        "description": "operator\\note",
        "other_data": {"source": "camera-a"},
    }


class WorksetSnapshotV1Tests(unittest.TestCase):
    def _records(self):
        return [
            {
                "image_id": "image-a",
                "image_copy_succeeded": True,
                "session_image_path": "images/a.jpg",
                "session_label_path": "labels/a.json",
                "source_image_sha256": SHA_A,
            },
            {
                "image_id": "image-failed",
                "image_copy_succeeded": False,
                "session_image_path": "images/failed.jpg",
                "session_label_path": "labels/failed.json",
                "source_image_sha256": "",
            },
            {
                "image_id": "image-b",
                "image_copy_succeeded": True,
                "session_image_path": "images/b.jpg",
                "session_label_path": "labels/b.json",
                "source_image_sha256": SHA_B,
            },
        ]

    def test_session_workset_uses_manifest_order_not_search_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records()
            visible_after_search = ["image-b"]
            first = create_session_workset_snapshot_v1(
                records,
                tmp,
                UTC_NOW,
                current_anchor_image_id="image-a",
            )
            visible_after_search.clear()
            second = create_session_workset_snapshot_v1(
                records,
                tmp,
                UTC_NOW,
                current_anchor_image_id="image-a",
            )

        self.assertEqual(first, second)
        self.assertFalse(first["search_filter_applied"])
        self.assertEqual(
            [entry["image_id"] for entry in first["entries"]],
            ["image-a", "image-b"],
        )
        self.assertEqual(
            [entry["manifest_sequence"] for entry in first["entries"]],
            [0, 2],
        )

    def test_current_to_end_requires_one_stable_image_id_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = create_session_workset_snapshot_v1(
                self._records(),
                tmp,
                UTC_NOW,
                current_anchor_image_id="image-b",
            )
            selected = select_workset_entries_v1(snapshot, "CURRENT_TO_END")
            self.assertEqual(
                [entry["image_id"] for entry in selected], ["image-b"]
            )

            missing = copy.deepcopy(snapshot)
            missing["current_anchor_image_id"] = "not-in-workset"
            with self.assertRaisesRegex(
                ContractValidationError, "current_anchor_not_unique"
            ):
                select_workset_entries_v1(missing, "CURRENT_TO_END")

            duplicate = copy.deepcopy(snapshot)
            duplicate["entries"][1]["image_id"] = "image-a"
            with self.assertRaisesRegex(
                ContractValidationError, "duplicate_image_id"
            ):
                select_workset_entries_v1(duplicate, "CURRENT_TO_END")

    def test_session_workset_rejects_search_flag_and_schema_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = create_session_workset_snapshot_v1(
                self._records(), tmp, UTC_NOW
            )
        snapshot["search_filter_applied"] = True
        with self.assertRaisesRegex(
            ContractValidationError, "session_workset_cannot_apply_search"
        ):
            validate_workset_snapshot_v1(snapshot)

        snapshot["search_filter_applied"] = False
        snapshot["future_field"] = True
        with self.assertRaisesRegex(
            ContractValidationError, "unexpected_fields"
        ):
            validate_workset_snapshot_v1(snapshot)

    def test_schema_declares_closed_versioned_objects(self):
        for schema in (
            WORKSET_SNAPSHOT_V1_SCHEMA,
            IMAGE_INPUT_SNAPSHOT_V1_SCHEMA,
            ANNOTATION_COMMIT_EVENT_V1_SCHEMA,
        ):
            with self.subTest(schema=schema["title"]):
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(
                    set(schema["properties"]), set(schema["required"])
                )
        entry_schema = WORKSET_SNAPSHOT_V1_SCHEMA["properties"]["entries"][
            "items"
        ]
        self.assertEqual(
            set(entry_schema["properties"]), set(entry_schema["required"])
        )
        self.assertEqual(
            WORKSET_SNAPSHOT_V1_SCHEMA["properties"]["workset_schema_version"],
            {"const": 1},
        )

    def test_workset_digest_has_frozen_path_independent_golden_value(self):
        entries = [
            {
                "image_id": "\u56fe\u50cf-\u03b2",
                "sequence": 2,
                "expected_image_sha256": None,
                "manifest_sequence": None,
                "canonical_image_path": "ignored-image-b",
                "canonical_label_path": "ignored-label-b",
            },
            {
                "image_id": "image-a",
                "sequence": 0,
                "expected_image_sha256": "0" * 64,
                "manifest_sequence": 4,
                "canonical_image_path": "ignored-image-a",
                "canonical_label_path": "ignored-label-a",
            },
        ]

        self.assertEqual(workset_digest_v1(entries), WORKSET_GOLDEN)
        paths_changed = copy.deepcopy(entries)
        paths_changed[0]["canonical_image_path"] = "another-session-image"
        paths_changed[0]["canonical_label_path"] = "another-session-label"
        self.assertEqual(workset_digest_v1(paths_changed), WORKSET_GOLDEN)

    def test_session_workset_rejects_absolute_and_parent_escape_paths(self):
        for field in ("session_image_path", "session_label_path"):
            for unsafe_path, error_code in (
                ("/outside/a.jpg", "session_path_absolute"),
                ("C:\\outside\\a.jpg", "session_path_absolute"),
                ("\\\\server\\share\\a.jpg", "session_path_absolute"),
                ("../outside/a.jpg", "session_path_escape"),
                ("images/../../outside/a.jpg", "session_path_escape"),
            ):
                records = self._records()
                records[0][field] = unsafe_path
                with (
                    self.subTest(field=field, unsafe_path=unsafe_path),
                    tempfile.TemporaryDirectory() as tmp,
                    self.assertRaisesRegex(
                        ContractValidationError, error_code
                    ),
                ):
                    create_session_workset_snapshot_v1(records, tmp, UTC_NOW)

    def test_session_workset_rejects_duplicate_canonical_paths(self):
        for field, error_code in (
            ("session_image_path", "duplicate_image_path"),
            ("session_label_path", "duplicate_label_path"),
        ):
            records = self._records()
            records[2][field] = records[0][field]
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as tmp,
                self.assertRaisesRegex(ContractValidationError, error_code),
            ):
                create_session_workset_snapshot_v1(records, tmp, UTC_NOW)

    def test_session_workset_rejects_symlink_root_escape(self):
        for field, link_name, safe_other_path in (
            ("session_image_path", "images", "safe-labels/a.json"),
            ("session_label_path", "labels", "safe-images/a.jpg"),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as tmp,
            ):
                session_root = os.path.join(tmp, "session")
                outside_root = os.path.join(tmp, "outside")
                os.makedirs(session_root)
                os.makedirs(outside_root)
                link_path = os.path.join(session_root, link_name)
                try:
                    os.symlink(
                        outside_root, link_path, target_is_directory=True
                    )
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"symbolic links are unavailable: {exc}")

                record = {
                    "image_id": "image-a",
                    "image_copy_succeeded": True,
                    "session_image_path": safe_other_path,
                    "session_label_path": safe_other_path,
                    "source_image_sha256": SHA_A,
                }
                record[field] = f"{link_name}/a.jpg"
                with self.assertRaisesRegex(
                    ContractValidationError, "session_path_outside_root"
                ):
                    create_session_workset_snapshot_v1(
                        [record], session_root, UTC_NOW
                    )


class ImageInputSnapshotV1Tests(unittest.TestCase):
    def _snapshot(self, path):
        absolute = os.path.abspath(path)
        return {
            "snapshot_schema_version": 1,
            "image_id": "image-a",
            "requested_path": absolute,
            "canonical_path": canonical_path_identity(absolute),
            "expected_sha256": SHA_A,
            "actual_sha256": SHA_A,
            "file_size": 123,
            "mtime_ns": 456,
            "decoded_width": 100,
            "decoded_height": 80,
            "pixel_format": "RGB888",
            "exif_orientation": 1,
            "source": "SESSION_MANIFEST",
        }

    def test_accepts_frozen_image_input_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self._snapshot(Path(tmp) / "image.jpg")
            self.assertIs(validate_image_input_snapshot_v1(snapshot), snapshot)

    def test_rejects_non_normalized_exif_and_wrong_content_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self._snapshot(Path(tmp) / "image.jpg")

        snapshot["exif_orientation"] = 6
        with self.assertRaisesRegex(
            ContractValidationError, "exif_not_normalized"
        ):
            validate_image_input_snapshot_v1(snapshot)

        snapshot["exif_orientation"] = 1
        snapshot["canonical_path"] = canonical_path_identity(
            os.path.join(
                os.path.dirname(snapshot["requested_path"]), "other.jpg"
            )
        )
        with self.assertRaisesRegex(
            ContractValidationError, "canonical_path_mismatch"
        ):
            validate_image_input_snapshot_v1(snapshot)

        snapshot = self._snapshot(snapshot["requested_path"])
        snapshot["actual_sha256"] = SHA_B
        with self.assertRaisesRegex(
            ContractValidationError, "conflict_image_changed"
        ):
            validate_image_input_snapshot_v1(snapshot)


class AnnotationCanonicalizationV1Tests(unittest.TestCase):
    def test_valid_empty_json_is_existing_negative_annotation(self):
        self.assertEqual(
            classify_annotation_document_v1(None),
            ANNOTATION_PRESENCE_MISSING,
        )
        self.assertEqual(
            classify_annotation_document_v1('{"shapes": []}'),
            ANNOTATION_PRESENCE_VALID_EMPTY,
        )
        self.assertEqual(
            classify_annotation_document_v1(_label_document()),
            ANNOTATION_PRESENCE_VALID_NONEMPTY,
        )
        self.assertEqual(
            classify_annotation_document_v1({"shapes": [{}]}),
            ANNOTATION_PRESENCE_INVALID,
        )
        self.assertEqual(
            classify_annotation_document_v1("{broken"),
            ANNOTATION_PRESENCE_INVALID,
        )

    def test_only_image_path_normalizes_backslashes(self):
        canonical = canonicalize_annotation_document_v1(_label_document())

        self.assertEqual(canonical["imagePath"], "image.jpg")
        self.assertEqual(canonical["description"], "operator\\note")
        self.assertEqual(canonical["shapes"][0]["label"], "robot\\arm")
        self.assertEqual(canonical["shapes"][0]["points"][0], [1, 0])
        self.assertEqual(canonical["shapes"][0]["points"][1], [2, 3.5])
        self.assertRegex(
            canonical["imageData"]["__image_data_sha256__"],
            r"^[0-9a-f]{64}$",
        )

    def test_document_semantic_and_raw_digests_are_not_interchangeable(self):
        first = _label_document()
        transport_changed = copy.deepcopy(first)
        transport_changed.update(
            version="future",
            imagePath="other/image.jpg",
            imageData=base64.b64encode(b"different-pixels").decode("ascii"),
            imageHeight=160,
            imageWidth=200,
        )

        self.assertNotEqual(
            canonical_document_digest_v1(first),
            canonical_document_digest_v1(transport_changed),
        )
        self.assertEqual(
            semantic_annotation_digest_v1(first),
            semantic_annotation_digest_v1(transport_changed),
        )
        self.assertRegex(
            canonical_document_digest_v1(first), r"^aldoc1:[0-9a-f]{64}$"
        )
        self.assertRegex(
            semantic_annotation_digest_v1(first),
            r"^alsem1:[0-9a-f]{64}$",
        )
        self.assertRegex(raw_file_sha256_v1(b"raw"), r"^[0-9a-f]{64}$")

        semantic_changed = copy.deepcopy(first)
        semantic_changed["other_data"]["source"] = "camera-b"
        self.assertNotEqual(
            semantic_annotation_digest_v1(first),
            semantic_annotation_digest_v1(semantic_changed),
        )

        raw_first = json.dumps(first, separators=(",", ":")).encode("utf-8")
        raw_pretty = json.dumps(first, indent=2).encode("utf-8")
        self.assertNotEqual(
            raw_file_sha256_v1(raw_first), raw_file_sha256_v1(raw_pretty)
        )
        self.assertEqual(
            canonical_document_digest_v1(first),
            canonical_document_digest_v1(json.loads(raw_pretty)),
        )

    def test_numeric_and_unicode_rules_are_deterministic(self):
        integer_document = _label_document()
        float_document = copy.deepcopy(integer_document)
        float_document["shapes"][0]["points"][0][0] = 1.0
        self.assertEqual(
            canonical_document_digest_v1(integer_document),
            canonical_document_digest_v1(float_document),
        )

        collision = {"shapes": [], "\u00e9": 1, "e\u0301": 2}
        with self.assertRaisesRegex(
            ContractValidationError, "normalized_key_collision"
        ):
            canonical_document_digest_v1(collision)

        invalid_number = _label_document()
        invalid_number["shapes"][0]["points"][0][0] = float("nan")
        with self.assertRaisesRegex(
            ContractValidationError, "non_finite_number"
        ):
            canonical_document_digest_v1(invalid_number)

    def test_image_path_escape_and_absolute_paths_are_rejected(self):
        for image_path in (
            "../image.jpg",
            "C:\\images\\image.jpg",
            "/image.jpg",
        ):
            with self.subTest(image_path=image_path):
                document = _label_document()
                document["imagePath"] = image_path
                with self.assertRaises(ContractValidationError):
                    canonical_document_digest_v1(document)

    def test_semantic_canonicalization_wraps_non_string_top_level_keys(self):
        document = {"shapes": [], 7: "not-a-json-object-key"}
        with self.assertRaisesRegex(ContractValidationError, "non_string_key"):
            semantic_annotation_digest_v1(document)


class AnnotationCommitEventV1Tests(unittest.TestCase):
    def _event(self):
        document = _label_document()
        event = {
            "event_schema_version": 1,
            "event_id": "0" * 64,
            "project_id": "project-a",
            "session_id": "session-a",
            "run_id": "run-a",
            "image_id": "image-a",
            "attempt_id": "attempt-a",
            "writer_kind": "CONTINUOUS",
            "commit_scope": "STAGED",
            "mutation_mode": "NOTIFY_ONLY",
            "document_digest": canonical_document_digest_v1(document),
            "semantic_digest": semantic_annotation_digest_v1(document),
            "source_image_digest": SHA_A,
            "base_item_revision": 4,
            "created_at": UTC_NOW,
        }
        event["event_id"] = annotation_commit_event_id_v1(event)
        return event

    def test_event_id_binds_the_frozen_idempotency_fields(self):
        event = self._event()
        self.assertIs(validate_annotation_commit_event_v1(event), event)

        changed = copy.deepcopy(event)
        changed["attempt_id"] = "attempt-b"
        with self.assertRaisesRegex(
            ContractValidationError, "event_id_mismatch"
        ):
            validate_annotation_commit_event_v1(changed)

    def test_event_rejects_missing_or_mixed_digest_types(self):
        event = self._event()
        event["semantic_digest"] = event["document_digest"]
        event["event_id"] = annotation_commit_event_id_v1(event)
        with self.assertRaisesRegex(
            ContractValidationError, "invalid_semantic_digest"
        ):
            validate_annotation_commit_event_v1(event)

        event = self._event()
        del event["mutation_mode"]
        with self.assertRaisesRegex(ContractValidationError, "missing_fields"):
            validate_annotation_commit_event_v1(event)

    def test_writer_scope_mutation_and_digest_matrix_is_closed(self):
        valid = {
            "CONTINUOUS": {("STAGED", "NOTIFY_ONLY", "present")},
            "MANUAL_SAVE": {("STAGED", "APPLY_MANUAL_REVISION", "present")},
            "AUTO_SAVE": {("STAGED", "APPLY_MANUAL_REVISION", "present")},
            "DELETE_LABEL": {("STAGED", "APPLY_MANUAL_REVISION", "missing")},
            "SESSION_SOURCE_SYNC": {
                ("SOURCE", "NOTIFY_ONLY", "present"),
                ("SOURCE", "NOTIFY_ONLY", "missing"),
            },
        }
        for writer_kind, allowed in valid.items():
            for commit_scope in ("STAGED", "SOURCE"):
                for mutation_mode in (
                    "NOTIFY_ONLY",
                    "APPLY_MANUAL_REVISION",
                ):
                    for digest_state in ("present", "missing", "mixed"):
                        event = self._event()
                        event.update(
                            writer_kind=writer_kind,
                            commit_scope=commit_scope,
                            mutation_mode=mutation_mode,
                        )
                        if digest_state == "missing":
                            event.update(
                                document_digest="MISSING",
                                semantic_digest="MISSING",
                            )
                        elif digest_state == "mixed":
                            event["semantic_digest"] = "MISSING"
                        event["event_id"] = annotation_commit_event_id_v1(
                            event
                        )
                        combination = (
                            commit_scope,
                            mutation_mode,
                            digest_state,
                        )
                        with self.subTest(
                            writer_kind=writer_kind,
                            combination=combination,
                        ):
                            if combination in allowed:
                                validate_annotation_commit_event_v1(event)
                            else:
                                with self.assertRaises(
                                    ContractValidationError
                                ):
                                    validate_annotation_commit_event_v1(event)


if __name__ == "__main__":
    unittest.main()
