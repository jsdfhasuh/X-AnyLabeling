import os
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore

from anylabeling.views.labeling.pose_config import (
    PoseConfigError,
    create_pose_shape_dicts,
    load_pose_config,
    map_point_between_rects,
    normalize_pose_config,
    skeleton_segments,
)
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas


class TestPoseConfig(unittest.TestCase):

    def test_load_pose_config_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            pose_config_path = Path(tmp) / "pose_config.yaml"
            pose_config_path.write_text(
                yaml.safe_dump(
                    {
                        "has_visible": True,
                        "classes": {"robot": ["base", "tip"]},
                        "skeletons": {"robot": [["base", "tip"]]},
                        "template_points": {
                            "robot": {
                                "base": [0.1, 0.9],
                                "tip": [0.8, 0.2],
                            }
                        },
                        "flip_idx": [0, 1],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            pose_config = load_pose_config(str(pose_config_path))

        self.assertEqual(pose_config["classes"]["robot"], ["base", "tip"])
        self.assertEqual(pose_config["skeletons"]["robot"], [["base", "tip"]])
        self.assertEqual(
            pose_config["template_points"]["robot"]["tip"], [0.8, 0.2]
        )

    def test_create_pose_shape_dicts_uses_template_and_group_id(self):
        pose_config = {
            "classes": {"robot": ["base", "tip"]},
            "skeletons": {"robot": [["base", "tip"]]},
            "template_points": {
                "robot": {"base": [0.1, 0.9], "tip": [0.8, 0.2]}
            },
        }

        shapes = create_pose_shape_dicts(
            pose_config,
            "robot",
            [[10, 20], [110, 20], [110, 220], [10, 220]],
            group_id=7,
        )

        self.assertEqual([shape["group_id"] for shape in shapes], [7, 7, 7])
        self.assertEqual(shapes[0]["shape_type"], "rectangle")
        self.assertEqual(shapes[0]["label"], "robot")
        self.assertEqual(shapes[1]["label"], "base")
        self.assertEqual(shapes[1]["points"][0], [20.0, 200.0])
        self.assertEqual(shapes[2]["label"], "tip")
        self.assertEqual(shapes[2]["points"][0], [90.0, 60.0])

    def test_map_point_between_rects_scales_keypoint(self):
        self.assertEqual(
            map_point_between_rects(
                [20, 40],
                old_rect=(10, 20, 110, 220),
                new_rect=(20, 40, 220, 440),
            ),
            [40.0, 80.0],
        )

    def test_skeleton_segments_from_labelme_shapes(self):
        pose_config = {
            "classes": {"robot": ["base", "tip"]},
            "skeletons": {"robot": [["base", "tip"]]},
            "template_points": {},
        }
        rect = Shape(label="robot", shape_type="rectangle", group_id=3)
        rect.points = [
            QtCore.QPointF(0, 0),
            QtCore.QPointF(10, 0),
            QtCore.QPointF(10, 10),
            QtCore.QPointF(0, 10),
        ]
        base = Shape(label="base", shape_type="point", group_id=3)
        base.points = [QtCore.QPointF(2, 8)]
        tip = Shape(label="tip", shape_type="point", group_id=3)
        tip.points = [QtCore.QPointF(8, 2)]

        segments = skeleton_segments([rect, base, tip], pose_config)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["group_id"], 3)
        self.assertEqual(segments[0]["start_label"], "base")
        self.assertEqual(segments[0]["end_label"], "tip")

    def test_top_level_skeleton_filters_per_class(self):
        pose_config = normalize_pose_config(
            {
                "classes": {
                    "person": ["nose", "eye"],
                    "robot": ["base", "tip"],
                },
                "skeleton": [["nose", "eye"], ["base", "tip"]],
            }
        )

        self.assertEqual(
            pose_config["skeletons"],
            {
                "person": [["nose", "eye"]],
                "robot": [["base", "tip"]],
            },
        )

    def test_top_level_skeleton_rejects_unknown_keypoint(self):
        with self.assertRaises(PoseConfigError):
            normalize_pose_config(
                {
                    "classes": {"person": ["nose"]},
                    "skeleton": [["nose", "eye"]],
                }
            )

    def test_canvas_pose_rectangle_transform_scales_keypoints(self):
        canvas = SimpleNamespace(
            pose_config={
                "classes": {"robot": ["base", "tip"]},
                "skeletons": {"robot": [["base", "tip"]]},
                "template_points": {},
            },
            shapes=[],
        )
        canvas._is_pose_rectangle = MethodType(
            Canvas._is_pose_rectangle, canvas
        )
        canvas._shape_rect_tuple = Canvas._shape_rect_tuple
        canvas._apply_pose_rectangle_transform = MethodType(
            Canvas._apply_pose_rectangle_transform, canvas
        )
        rect = Shape(label="robot", shape_type="rectangle", group_id=9)
        rect.points = [
            QtCore.QPointF(10, 20),
            QtCore.QPointF(110, 20),
            QtCore.QPointF(110, 220),
            QtCore.QPointF(10, 220),
        ]
        base = Shape(label="base", shape_type="point", group_id=9)
        base.points = [QtCore.QPointF(20, 40)]
        ignored = Shape(label="other", shape_type="point", group_id=9)
        ignored.points = [QtCore.QPointF(20, 40)]
        canvas.shapes = [rect, base, ignored]

        old_rect = Canvas._shape_rect_tuple(rect)
        rect.points = [
            QtCore.QPointF(20, 40),
            QtCore.QPointF(220, 40),
            QtCore.QPointF(220, 440),
            QtCore.QPointF(20, 440),
        ]
        canvas._apply_pose_rectangle_transform(rect, old_rect)

        self.assertEqual(base.points[0], QtCore.QPointF(40, 80))
        self.assertEqual(ignored.points[0], QtCore.QPointF(20, 40))
