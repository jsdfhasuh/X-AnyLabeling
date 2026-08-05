import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from anylabeling.services.auto_labeling.__base__.yolo import YOLO


class TestYoloScalarValues(unittest.TestCase):

    def _make_model(self, tracker=None):
        model = YOLO.__new__(YOLO)
        model.classes = ["robot"]
        model.task = "pose"
        model.tracker = tracker
        return model

    @staticmethod
    def _point_coordinates(shape):
        return [(point.x(), point.y()) for point in shape.points]

    def _predict_pose_keypoint(self, keypoint):
        model = self._make_model()
        model.model_type = "mock"
        model.show_boxes = False
        model.keypoint_name = {"robot": ["head"]}
        model.kpt_thres = 0.5
        model.replace = True
        model.preprocess = mock.Mock(return_value=object())
        model.inference = mock.Mock(return_value=object())
        model.postprocess = mock.Mock(
            return_value=(
                np.array([[1, 2, 21, 22]], dtype=np.float32),
                np.array([[0]], dtype=np.int64),
                np.array([[0.93]], dtype=np.float32),
                None,
                [np.expand_dims(keypoint, axis=0)],
            )
        )
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        with mock.patch(
            "anylabeling.services.auto_labeling.__base__.yolo."
            "qt_img_to_rgb_cv_img",
            return_value=image,
        ):
            return model.predict_shapes(image)

    def test_rectangle_shape_accepts_single_element_arrays(self):
        model = self._make_model()

        shape = model.create_rectangle_shape(
            np.array([1, 2, 11, 22], dtype=np.float32),
            np.array([[0.93]], dtype=np.float32),
            np.array([5]),
            np.array([[0.0]], dtype=np.float32),
            [],
        )

        self.assertEqual(shape.label, "robot")
        self.assertEqual(shape.group_id, 5)
        self.assertAlmostEqual(shape.score, 0.93, places=5)

    def test_rectangle_shape_accepts_nested_box(self):
        model = self._make_model()

        shape = model.create_rectangle_shape(
            np.array([[1, 2, 11, 22]], dtype=np.float32),
            np.array([[0.93]], dtype=np.float32),
            np.array([[5]]),
            np.array([[0.0]], dtype=np.float32),
            [],
        )

        self.assertEqual(shape.label, "robot")
        self.assertEqual(shape.group_id, 5)
        self.assertAlmostEqual(shape.score, 0.93, places=5)
        self.assertEqual(
            self._point_coordinates(shape),
            [(1.0, 2.0), (11.0, 2.0), (11.0, 22.0), (1.0, 22.0)],
        )

    def test_rectangle_shape_accepts_column_box(self):
        model = self._make_model()

        shape = model.create_rectangle_shape(
            np.array([[1], [2], [11], [22]], dtype=np.float32),
            np.array([[0.93]], dtype=np.float32),
            np.array([[5]]),
            np.array([[0.0]], dtype=np.float32),
            [],
        )

        self.assertEqual(
            self._point_coordinates(shape),
            [(1.0, 2.0), (11.0, 2.0), (11.0, 22.0), (1.0, 22.0)],
        )

    def test_keypoint_shape_accepts_single_element_arrays(self):
        model = self._make_model(tracker=object())

        shape = model.create_keypoint_shape(
            (np.array([12.7]), np.array([[34.2]])),
            ["head", "arm"],
            np.array([0.81], dtype=np.float32),
            np.array([[1]]),
            np.array([6]),
            np.array([[9]]),
        )

        self.assertEqual(shape.label, "arm")
        self.assertEqual(shape.group_id, 9)
        self.assertAlmostEqual(shape.score, 0.81, places=5)

    def test_pose_keypoint_accepts_nested_xy(self):
        result = self._predict_pose_keypoint(
            np.array([[12.0, 14.0]], dtype=np.float32)
        )

        self.assertEqual(len(result.shapes), 1)
        shape = result.shapes[0]
        self.assertEqual(shape.label, "head")
        self.assertEqual(shape.score, 1.0)
        self.assertEqual(self._point_coordinates(shape), [(12.0, 14.0)])

    def test_pose_keypoint_accepts_nested_xys(self):
        result = self._predict_pose_keypoint(
            np.array([[12.0, 14.0, 0.75]], dtype=np.float32)
        )

        self.assertEqual(len(result.shapes), 1)
        shape = result.shapes[0]
        self.assertEqual(shape.label, "head")
        self.assertAlmostEqual(shape.score, 0.75, places=5)
        self.assertEqual(self._point_coordinates(shape), [(12.0, 14.0)])

    def test_polygon_shape_accepts_nested_points(self):
        model = self._make_model()

        shape = model.create_polygon_shape(
            np.array(
                [[[1, 2]], [[11, 2]], [[11, 22]], [[1, 22]]],
                dtype=np.float32,
            ),
            np.array([[0.93]], dtype=np.float32),
            np.array([[0]]),
            [],
        )

        self.assertEqual(shape.label, "robot")
        self.assertAlmostEqual(shape.score, 0.93, places=5)
        self.assertEqual(
            self._point_coordinates(shape),
            [(1.0, 2.0), (11.0, 2.0), (11.0, 22.0), (1.0, 22.0)],
        )

    def test_obb_shape_accepts_nested_box(self):
        model = self._make_model()

        shape = model.create_obb_shape(
            np.array([[10, 20, 4, 6, 0]], dtype=np.float32),
            np.array([[0.93]], dtype=np.float32),
            np.array([[0]]),
            [],
        )

        self.assertEqual(shape.label, "robot")
        self.assertAlmostEqual(shape.score, 0.93, places=5)
        self.assertEqual(
            self._point_coordinates(shape),
            [(12.0, 23.0), (12.0, 17.0), (8.0, 17.0), (8.0, 23.0)],
        )


if __name__ == "__main__":
    unittest.main()
