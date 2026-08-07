import base64
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from anylabeling.app_info import __version__
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError


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
