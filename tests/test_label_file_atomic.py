import base64
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from anylabeling.app_info import __version__
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    canonical_document_digest_v1,
    resolve_existing_label,
)


class LabelFileAtomicSaveRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.label_path = self.root / "sample.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _rectangle():
        return {
            "label": "box",
            "score": None,
            "points": [(5, 4), (1, 1)],
            "group_id": None,
            "description": "shape note",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {"source": "human"},
            "kie_linking": [],
            "vendor_shape_field": "preserved",
        }

    def test_existing_labelme_save_shape_and_unknown_fields_remain_compatible(
        self,
    ):
        shapes = [self._rectangle()]
        shapes_before = copy.deepcopy(shapes)
        other_data = {
            "description": "document note",
            "vendor_document_field": {"camera": "a"},
        }
        flags = {"verified": True}
        label_file = LabelFile()
        label_file.save(
            filename=self.label_path,
            shapes=shapes,
            image_path="sample.png",
            image_height=6,
            image_width=8,
            image_data=None,
            other_data=other_data,
            flags=flags,
        )
        self.assertEqual(shapes, shapes_before)
        saved = json.loads(self.label_path.read_text(encoding="utf-8"))
        expected_shape = copy.deepcopy(shapes_before[0])
        expected_shape["points"] = [[1, 1], [5, 1], [5, 4], [1, 4]]
        self.assertEqual(
            saved,
            {
                "version": __version__,
                "flags": flags,
                "shapes": [expected_shape],
                "imagePath": "sample.png",
                "imageData": None,
                "imageHeight": 6,
                "imageWidth": 8,
                **other_data,
            },
        )
        self.assertEqual(label_file.filename, self.label_path)

    def test_image_data_is_encoded_without_mutating_callers(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 6), color=(1, 2, 3)).save(buffer, format="PNG")
        raw_image = buffer.getvalue()
        shapes = [self._rectangle()]
        label_file = LabelFile()
        label_file.save(
            filename=self.label_path,
            shapes=shapes,
            image_path="sample.png",
            image_height=6,
            image_width=8,
            image_data=raw_image,
            other_data={},
            flags={},
        )
        saved = json.loads(self.label_path.read_text(encoding="utf-8"))
        self.assertEqual(
            saved["imageData"], base64.b64encode(raw_image).decode("utf-8")
        )

    def test_corrupt_existing_label_is_not_overwritten(self):
        self.label_path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(LabelFileError):
            LabelFile().save(
                filename=self.label_path,
                shapes=[self._rectangle()],
                image_path="sample.png",
                image_height=6,
                image_width=8,
                image_data=None,
                other_data={},
                flags={},
            )
        self.assertEqual(
            self.label_path.read_text(encoding="utf-8"), "{broken"
        )

    def test_sibling_image_directory_normalizes_before_callback_and_write(
        self,
    ):
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        image_path = images_dir / "image-a.jpg"
        label_path = labels_dir / "image-a.json"
        Image.new("RGB", (8, 6), color=(20, 30, 40)).save(image_path)
        observed = []

        def before_write(document, current):
            observed.append((document, current.document_digest))

        LabelFile().save(
            filename=label_path,
            shapes=[],
            image_path="../images/image-a.jpg",
            image_source_path=image_path,
            image_height=6,
            image_width=8,
            image_data=None,
            other_data={"description": "sibling"},
            flags={},
            before_write=before_write,
        )

        self.assertEqual(len(observed), 1)
        callback_document, callback_pre_digest = observed[0]
        self.assertEqual(callback_document["imagePath"], "image-a.jpg")
        self.assertNotIn("..", callback_document["imagePath"])
        self.assertEqual(callback_pre_digest, "MISSING")
        saved = json.loads(label_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["imagePath"], "image-a.jpg")
        self.assertFalse(Path(saved["imagePath"]).is_absolute())
        self.assertEqual(
            canonical_document_digest_v1(callback_document),
            resolve_existing_label(label_path).document_digest,
        )

    def test_callback_failure_happens_before_file_creation(self):
        callback = mock.Mock(side_effect=RuntimeError("prepare failed"))
        with self.assertRaisesRegex(LabelFileError, "prepare failed"):
            LabelFile().save(
                filename=self.label_path,
                shapes=[],
                image_path="sample.png",
                image_height=6,
                image_width=8,
                image_data=None,
                other_data={},
                flags={},
                before_write=callback,
            )
        callback.assert_called_once()
        self.assertFalse(self.label_path.exists())

    def test_other_data_cannot_replace_writer_owned_fields(self):
        with self.assertRaisesRegex(LabelFileError, "field collision"):
            LabelFile().save(
                filename=self.label_path,
                shapes=[self._rectangle()],
                image_path="sample.png",
                image_height=6,
                image_width=8,
                image_data=None,
                other_data={"shapes": []},
                flags={},
            )


if __name__ == "__main__":
    unittest.main()
