import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
