import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from PyQt5 import QtWidgets

from anylabeling.views.labeling.utils import batch


class _SerializedShape:
    def __init__(self, label):
        self.label = label

    def to_dict(self):
        return {
            "label": self.label,
            "points": [[1, 2], [3, 4]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }


class LegacyAutoRunBaselineTests(unittest.TestCase):
    def test_existing_replace_and_merge_save_semantics_remain_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images" / "sample.jpg"
            output_dir = root / "labels"
            image_path.parent.mkdir()
            output_dir.mkdir()
            Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
            label_path = output_dir / "sample.json"
            label_path.write_text(
                json.dumps(
                    {
                        "version": "3.3.7",
                        "flags": {},
                        "shapes": [_SerializedShape("old").to_dict()],
                        "imagePath": "sample.jpg",
                        "imageData": None,
                        "imageHeight": 10,
                        "imageWidth": 20,
                        "description": "human",
                    }
                ),
                encoding="utf-8",
            )
            widget = SimpleNamespace(
                output_dir=str(output_dir), _config={"store_data": False}
            )

            merge = SimpleNamespace(
                shapes=[_SerializedShape("merged")],
                description="model",
                replace=False,
            )
            batch.save_auto_labeling_result(widget, str(image_path), merge)
            merged = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [shape["label"] for shape in merged["shapes"]],
                ["old", "merged"],
            )
            self.assertEqual(merged["description"], "humanmodel")

            replace = SimpleNamespace(
                shapes=[_SerializedShape("replacement")],
                description="replacement-description",
                replace=True,
            )
            batch.save_auto_labeling_result(widget, str(image_path), replace)
            replaced = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [shape["label"] for shape in replaced["shapes"]],
                ["replacement"],
            )
            self.assertEqual(
                replaced["description"], "replacement-description"
            )

    def test_auto_run_starts_at_the_current_image_for_supported_model(self):
        files = [os.path.abspath(name) for name in ("a.jpg", "b.jpg", "c.jpg")]
        widget = SimpleNamespace(
            image_list=files,
            filename=files[1],
            fn_to_index={name: index for index, name in enumerate(files)},
            tr=lambda text: text,
            auto_labeling_widget=SimpleNamespace(
                model_manager=SimpleNamespace(
                    loaded_model_config={
                        "type": "yolov8",
                        "model": object(),
                    },
                    new_model_status=mock.Mock(),
                )
            ),
        )
        original_message_box = QtWidgets.QMessageBox

        class _MessageBox:
            Warning = original_message_box.Warning
            Cancel = original_message_box.Cancel
            Ok = original_message_box.Ok

            def setIcon(self, _icon):
                pass

            def setWindowTitle(self, _title):
                pass

            def setText(self, _text):
                pass

            def setStandardButtons(self, _buttons):
                pass

            def setStyleSheet(self, _style):
                pass

            def exec_(self):
                return self.Ok

        with (
            mock.patch.object(batch.QtWidgets, "QMessageBox", _MessageBox),
            mock.patch.object(
                batch, "show_progress_dialog_and_process"
            ) as show,
        ):
            batch.run_all_images(widget)

        self.assertEqual(widget.current_index, 1)
        self.assertEqual(widget.image_index, 1)
        self.assertEqual(widget.text_prompt, "")
        self.assertFalse(widget.run_tracker)
        show.assert_called_once_with(widget)


if __name__ == "__main__":
    unittest.main()
