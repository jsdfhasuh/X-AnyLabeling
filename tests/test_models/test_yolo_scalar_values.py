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

    def _predict_pose_shapes(self, boxes, keypoints, show_boxes=False):
        model = self._make_model()
        model.model_type = "mock"
        model.show_boxes = show_boxes
        model.keypoint_name = {"robot": ["head"]}
        model.kpt_thres = 0.5
        model.replace = True
        model.preprocess = mock.Mock(return_value=object())
        model.inference = mock.Mock(return_value=object())
        model.postprocess = mock.Mock(
            return_value=(
                boxes,
                np.array([[0]], dtype=np.int64),
                np.array([[0.93]], dtype=np.float32),
                None,
                keypoints,
            )
        )
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        with mock.patch(
            "anylabeling.services.auto_labeling.__base__.yolo."
            "qt_img_to_rgb_cv_img",
            return_value=image,
        ):
            return model.predict_shapes(image)

    def _predict_pose_keypoint(self, keypoint):
        return self._predict_pose_shapes(
            np.array([[1, 2, 21, 22]], dtype=np.float32),
            [np.expand_dims(keypoint, axis=0)],
        )

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

    def test_predict_shapes_accepts_nested_pose_box_and_xys(self):
        result = self._predict_pose_shapes(
            np.array([[[1, 2, 21, 22]]], dtype=np.float32),
            np.array([[[[12.0, 14.0, 0.75]]]], dtype=np.float32),
            show_boxes=True,
        )

        self.assertEqual(len(result.shapes), 2)
        rectangle, keypoint = result.shapes
        self.assertEqual(rectangle.shape_type, "rectangle")
        self.assertEqual(rectangle.label, "robot")
        self.assertAlmostEqual(rectangle.score, 0.93, places=5)
        self.assertEqual(
            self._point_coordinates(rectangle),
            [(1.0, 2.0), (21.0, 2.0), (21.0, 22.0), (1.0, 22.0)],
        )
        self.assertEqual(keypoint.shape_type, "point")
        self.assertEqual(keypoint.label, "head")
        self.assertAlmostEqual(keypoint.score, 0.75, places=5)
        self.assertEqual(self._point_coordinates(keypoint), [(12.0, 14.0)])
        self.assertEqual(rectangle.group_id, keypoint.group_id)

    def test_predict_shapes_accepts_nested_pose_box_and_xy(self):
        result = self._predict_pose_shapes(
            np.array([[[1, 2, 21, 22]]], dtype=np.float32),
            np.array([[[[12.0, 14.0]]]], dtype=np.float32),
            show_boxes=True,
        )

        self.assertEqual(len(result.shapes), 2)
        rectangle, keypoint = result.shapes
        self.assertEqual(rectangle.label, "robot")
        self.assertAlmostEqual(rectangle.score, 0.93, places=5)
        self.assertEqual(keypoint.score, 1.0)
        self.assertEqual(self._point_coordinates(keypoint), [(12.0, 14.0)])
        self.assertEqual(rectangle.group_id, keypoint.group_id)

    def test_rectangle_shape_rejects_invalid_box_lengths(self):
        model = self._make_model()

        for box in ([1, 2, 11], [1, 2, 11, 22, 23]):
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError, "Expected 4 values"):
                    model.create_rectangle_shape(
                        np.array([box], dtype=np.float32),
                        0.93,
                        5,
                        0,
                        [],
                    )

    def test_pose_keypoint_rejects_invalid_lengths(self):
        boxes = np.array([[[1, 2, 21, 22]]], dtype=np.float32)

        for keypoint in ([12.0], [12.0, 14.0, 0.75, 0.5]):
            with self.subTest(keypoint=keypoint):
                keypoints = np.array([[[keypoint]]], dtype=np.float32)
                with self.assertRaisesRegex(
                    ValueError, "Expected 2 or 3 values"
                ):
                    self._predict_pose_shapes(boxes, keypoints)

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

    def test_polygon_shape_rejects_invalid_point_lengths(self):
        model = self._make_model()

        for point in ([1], [1, 2, 3]):
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "Expected 2 values"):
                    model.create_polygon_shape(
                        np.array([[point]], dtype=np.float32),
                        0.93,
                        0,
                        [],
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

    def test_obb_shape_rejects_invalid_box_lengths(self):
        model = self._make_model()

        for box in ([10, 20, 4, 6], [10, 20, 4, 6, 0, 1]):
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError, "Expected 5 values"):
                    model.create_obb_shape(
                        np.array([box], dtype=np.float32),
                        0.93,
                        0,
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
