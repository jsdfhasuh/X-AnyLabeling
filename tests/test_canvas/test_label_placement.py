import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import (
    LABEL_EDGE_MARGIN_PX,
    LABEL_FONT_POINT_SIZE,
    LABEL_GAP_PX,
    LABEL_PADDING_X_PX,
    LABEL_PADDING_Y_PX,
    Canvas,
)


APPLICATION = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class LabelPainterSpy:

    def __init__(self):
        self._transform = QtGui.QTransform()
        self.draw_text_calls = []

    def transform(self):
        return self._transform

    def save(self):
        pass

    def resetTransform(self):
        self._transform = QtGui.QTransform()

    def setFont(self, _font):
        pass

    def fillRect(self, _rect, _color):
        pass

    def setPen(self, _pen):
        pass

    def drawText(self, x, y, text):
        self.draw_text_calls.append((x, y, text))

    def restore(self):
        pass


class TestLabelPlacement(unittest.TestCase):

    def setUp(self):
        self.original_point_size = Shape.point_size
        self.canvas = Canvas(parent=None)
        self.canvas.show_texts = True
        self.canvas.show_attributes = True

    def tearDown(self):
        Shape.point_size = self.original_point_size
        self.canvas.close()
        self.canvas.deleteLater()
        APPLICATION.processEvents()

    @staticmethod
    def _make_shape(
        shape_type="rectangle",
        bounds=(50, 50, 90, 90),
        label="target",
        **kwargs,
    ):
        x1, y1, x2, y2 = bounds
        shape = Shape(
            label=label,
            shape_type=shape_type,
            attributes=kwargs.pop("attributes", {}),
            **kwargs,
        )
        if shape_type == "polygon":
            shape.points = [
                QtCore.QPointF(x1, y1 + 3),
                QtCore.QPointF(x2 - 2, y1),
                QtCore.QPointF(x2, y2 - 3),
                QtCore.QPointF(x1 + 2, y2),
            ]
        else:
            shape.points = [
                QtCore.QPointF(x1, y1),
                QtCore.QPointF(x2, y1),
                QtCore.QPointF(x2, y2),
                QtCore.QPointF(x1, y2),
            ]
        shape.close()
        return shape

    @staticmethod
    def _font_metrics():
        font = QtGui.QFont("Arial", int(round(LABEL_FONT_POINT_SIZE)))
        return QtGui.QFontMetrics(font)

    def _label_size(self, display_text):
        metrics = self._font_metrics()
        return QtCore.QSizeF(
            metrics.horizontalAdvance(display_text) + 2 * LABEL_PADDING_X_PX,
            metrics.height() + 2 * LABEL_PADDING_Y_PX,
        )

    @staticmethod
    def _image_bounds(width=200, height=160):
        return QtCore.QRectF(0, 0, width, height)

    def _place(
        self,
        shape,
        shape_rect,
        label_size=QtCore.QSizeF(42, 18),
        image_rect=None,
        transform=None,
    ):
        if image_rect is None:
            image_rect = self._image_bounds()
        if transform is None:
            transform = QtGui.QTransform()
        occupied_rects = self.canvas._shape_label_occupied_rects(
            shape, transform
        )
        return self.canvas._place_shape_label(
            shape,
            shape_rect,
            label_size,
            image_rect,
            occupied_rects,
        )

    def _prepare_render_canvas(self, width=200, height=160):
        self.canvas.setFixedSize(width, height)
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtGui.QColor("#FFFFFF"))
        self.canvas.load_pixmap(pixmap)
        self.canvas.cross_line_show = False
        self.canvas.show_groups = False
        self.canvas.show_masks = False
        self.canvas.show_texts = False
        self.canvas.show_attributes = False
        self.canvas.show_linking = False
        self.canvas.show_degrees = False

    def _rendered_label_rect(self, shape):
        metrics = self._font_metrics()
        image_rect = self._image_bounds(
            self.canvas.pixmap.width(), self.canvas.pixmap.height()
        )
        label_text = self.canvas._build_shape_label_text(shape)
        display_text = self.canvas._elide_label_text(
            label_text, metrics, image_rect
        )
        label_size = self._label_size(display_text)
        shape_rect = self.canvas._shape_label_screen_rect(
            shape, QtGui.QTransform()
        )
        return self._place(shape, shape_rect, label_size, image_rect)

    def test_normal_rectangle_uses_top_external_position(self):
        shape = self._make_shape()
        shape_rect = QtCore.QRectF(50, 50, 40, 40)

        label_rect = self._place(shape, shape_rect)

        self.assertLessEqual(
            label_rect.bottom(), shape_rect.top() - LABEL_GAP_PX
        )
        self.assertFalse(label_rect.intersects(shape_rect))

    def test_top_boundary_falls_back_below_shape(self):
        shape = self._make_shape(bounds=(50, 1, 90, 31))
        shape_rect = QtCore.QRectF(50, 1, 40, 30)

        label_rect = self._place(shape, shape_rect)

        self.assertGreaterEqual(
            label_rect.top(), shape_rect.bottom() + LABEL_GAP_PX
        )
        self.assertFalse(label_rect.intersects(shape_rect))

    def test_bottom_boundary_with_description_uses_side(self):
        shape = self._make_shape(
            bounds=(80, 170, 110, 190), description="inspection note"
        )
        shape_rect = QtCore.QRectF(80, 170, 30, 20)
        image_rect = self._image_bounds(220, 200)

        label_rect = self._place(shape, shape_rect, image_rect=image_rect)

        self.assertGreaterEqual(
            label_rect.left(), shape_rect.right() + LABEL_GAP_PX
        )
        self.assertFalse(label_rect.intersects(shape_rect))

    def test_all_image_corners_keep_label_inside_image(self):
        image_rect = self._image_bounds()
        safe_image = image_rect.adjusted(
            LABEL_EDGE_MARGIN_PX,
            LABEL_EDGE_MARGIN_PX,
            -LABEL_EDGE_MARGIN_PX,
            -LABEL_EDGE_MARGIN_PX,
        )
        corners = (
            QtCore.QRectF(1, 1, 10, 10),
            QtCore.QRectF(189, 1, 10, 10),
            QtCore.QRectF(1, 149, 10, 10),
            QtCore.QRectF(189, 149, 10, 10),
        )

        for shape_rect in corners:
            with self.subTest(shape_rect=shape_rect):
                shape = self._make_shape(
                    bounds=(
                        shape_rect.left(),
                        shape_rect.top(),
                        shape_rect.right(),
                        shape_rect.bottom(),
                    )
                )
                label_rect = self._place(
                    shape, shape_rect, image_rect=image_rect
                )
                self.assertTrue(safe_image.contains(label_rect))

    def test_small_box_and_vertex_area_remain_uncovered(self):
        shape = self._make_shape(bounds=(80, 70, 82, 72))
        shape.selected = True
        shape_rect = QtCore.QRectF(80, 70, 2, 2)
        vertex_protection = shape_rect.adjusted(
            -Shape.point_size / 2,
            -Shape.point_size / 2,
            Shape.point_size / 2,
            Shape.point_size / 2,
        )

        label_rect = self._place(shape, shape_rect)

        self.assertFalse(label_rect.intersects(shape_rect))
        self.assertFalse(label_rect.intersects(vertex_protection))

    def test_default_point_size_keeps_box_labels_external(self):
        Shape.point_size = 10
        protection_distance = self.canvas._label_protection_distance()
        self.assertEqual(protection_distance, 6)

        for shape_type in ("rectangle", "polygon", "rotation"):
            with self.subTest(shape_type=shape_type):
                shape = self._make_shape(
                    shape_type=shape_type,
                    bounds=(90, 80, 130, 120),
                )
                shape_rect = self.canvas._shape_label_screen_rect(
                    shape, QtGui.QTransform()
                )
                label_rect = self._place(shape, shape_rect)
                vertex_protection = shape_rect.adjusted(-5, -5, 5, 5)

                self.assertFalse(label_rect.intersects(shape_rect))
                self.assertFalse(label_rect.intersects(vertex_protection))
                self.assertTrue(
                    label_rect.bottom()
                    <= shape_rect.top() - protection_distance
                    or label_rect.top()
                    >= shape_rect.bottom() + protection_distance
                    or label_rect.right()
                    <= shape_rect.left() - protection_distance
                    or label_rect.left()
                    >= shape_rect.right() + protection_distance
                )

    def test_polygon_and_rotation_use_mapped_bounding_rectangles(self):
        transform = QtGui.QTransform()
        transform.translate(7, 11)
        transform.scale(2, 3)

        for shape_type in ("polygon", "rotation"):
            with self.subTest(shape_type=shape_type):
                shape = self._make_shape(
                    shape_type=shape_type, bounds=(20, 30, 60, 70)
                )
                screen_rect = self.canvas._shape_label_screen_rect(
                    shape, transform
                )
                expected = transform.mapRect(shape.bounding_rect())
                self.assertEqual(screen_rect, expected)

                image_rect = transform.mapRect(self._image_bounds(120, 100))
                label_rect = self._place(
                    shape,
                    screen_rect,
                    image_rect=image_rect,
                    transform=transform,
                )
                self.assertFalse(label_rect.intersects(screen_rect))

    def test_description_and_attributes_reserve_top_and_bottom(self):
        shape = self._make_shape(
            bounds=(70, 60, 110, 100),
            description="note",
            attributes={"state": "ok"},
        )
        shape_rect = QtCore.QRectF(70, 60, 40, 40)

        label_rect = self._place(shape, shape_rect)

        self.assertGreaterEqual(
            label_rect.left(), shape_rect.right() + LABEL_GAP_PX
        )

    def test_attribute_reserves_bottom_when_top_is_unavailable(self):
        shape = self._make_shape(
            bounds=(70, 1, 110, 41), attributes={"state": "ok"}
        )
        shape_rect = QtCore.QRectF(70, 1, 40, 40)

        label_rect = self._place(shape, shape_rect)

        self.assertGreaterEqual(
            label_rect.left(), shape_rect.right() + LABEL_GAP_PX
        )

    def test_description_degraded_top_candidate_remains_available(self):
        Shape.point_size = 10
        shape = self._make_shape(
            bounds=(1, 80, 119, 110),
            description="inspection note",
        )
        shape_rect = QtCore.QRectF(1, 80, 118, 30)
        image_rect = self._image_bounds(120, 120)
        occupied_rects = self.canvas._shape_label_occupied_rects(
            shape, QtGui.QTransform()
        )

        label_rect = self._place(shape, shape_rect, image_rect=image_rect)

        self.assertLessEqual(
            label_rect.bottom(),
            occupied_rects["top"].top()
            - self.canvas._label_protection_distance(),
        )
        self.assertFalse(label_rect.intersects(occupied_rects["top"]))

    def test_attribute_degraded_bottom_candidate_remains_available(self):
        Shape.point_size = 10
        shape = self._make_shape(
            bounds=(1, 1, 119, 31),
            attributes={"state": "ok"},
        )
        shape_rect = QtCore.QRectF(1, 1, 118, 30)
        image_rect = self._image_bounds(120, 120)
        occupied_rects = self.canvas._shape_label_occupied_rects(
            shape, QtGui.QTransform()
        )

        label_rect = self._place(shape, shape_rect, image_rect=image_rect)

        self.assertGreaterEqual(
            label_rect.top(),
            occupied_rects["bottom"].bottom()
            + self.canvas._label_protection_distance(),
        )
        self.assertFalse(label_rect.intersects(occupied_rects["bottom"]))

    def test_inside_fallback_is_used_only_when_external_positions_fail(self):
        shape = self._make_shape(bounds=(2, 2, 198, 158))
        shape_rect = QtCore.QRectF(2, 2, 196, 156)

        label_rect = self._place(shape, shape_rect)

        self.assertTrue(label_rect.intersects(shape_rect))
        self.assertTrue(
            self._image_bounds().adjusted(1, 1, -1, -1).contains(label_rect)
        )

    def test_inside_fallback_does_not_overlap_attributes(self):
        shape = self._make_shape(
            bounds=(1, 5, 49, 7),
            attributes={"state": "ok"},
        )
        shape_rect = QtCore.QRectF(1, 5, 48, 2)

        label_rect = self._place(
            shape,
            shape_rect,
            image_rect=self._image_bounds(50, 30),
        )

        self.assertIsNone(label_rect)

    def test_long_label_is_elided_without_mutating_shape(self):
        original_label = "calibration_target_with_a_very_long_identifier"
        shape = self._make_shape(label=original_label)
        image_rect = self._image_bounds(90, 160)
        metrics = self._font_metrics()
        full_text = self.canvas._build_shape_label_text(shape)

        display_text = self.canvas._elide_label_text(
            full_text, metrics, image_rect
        )

        max_width = (
            image_rect.width()
            - 2 * LABEL_EDGE_MARGIN_PX
            - 2 * LABEL_PADDING_X_PX
        )
        self.assertNotEqual(display_text, full_text)
        self.assertIn("\u2026", display_text)
        self.assertLessEqual(
            metrics.horizontalAdvance(display_text), max_width
        )
        self.assertEqual(shape.label, original_label)

    def test_score_toggle_changes_text_without_breaking_constraints(self):
        shape = self._make_shape(score=0.876)
        shape_rect = QtCore.QRectF(50, 50, 40, 40)
        metrics = self._font_metrics()

        self.canvas.show_scores = False
        plain_text = self.canvas._build_shape_label_text(shape)
        plain_rect = self._place(
            shape, shape_rect, self._label_size(plain_text)
        )
        self.canvas.show_scores = True
        scored_text = self.canvas._build_shape_label_text(shape)
        scored_rect = self._place(
            shape, shape_rect, self._label_size(scored_text)
        )

        self.assertEqual(plain_text, "target")
        self.assertEqual(scored_text, "target 0.88")
        self.assertEqual(plain_rect.topLeft(), scored_rect.topLeft())
        for text, rect in (
            (plain_text, plain_rect),
            (scored_text, scored_rect),
        ):
            self.assertLessEqual(
                metrics.horizontalAdvance(text) + 2 * LABEL_PADDING_X_PX,
                rect.width(),
            )
            self.assertFalse(rect.intersects(shape_rect))

    def test_zoom_keeps_label_size_and_gap_stable(self):
        shape = self._make_shape(bounds=(100, 100, 180, 160))
        metrics = self._font_metrics()
        display_text = self.canvas._build_shape_label_text(shape)
        label_size = self._label_size(display_text)
        layouts = []

        for scale in (0.5, 1.0, 3.0):
            transform = QtGui.QTransform()
            transform.scale(scale, scale)
            shape_rect = self.canvas._shape_label_screen_rect(shape, transform)
            image_rect = transform.mapRect(self._image_bounds(400, 300))
            text = self.canvas._elide_label_text(
                display_text, metrics, image_rect
            )
            self.assertEqual(text, display_text)
            label_rect = self._place(shape, shape_rect, label_size, image_rect)
            layouts.append((shape_rect, label_rect))

        widths = {label_rect.width() for _, label_rect in layouts}
        heights = {label_rect.height() for _, label_rect in layouts}
        gaps = {
            round(shape_rect.top() - label_rect.bottom(), 3)
            for shape_rect, label_rect in layouts
        }
        self.assertEqual(len(widths), 1)
        self.assertEqual(len(heights), 1)
        self.assertEqual(len(gaps), 1)

    def test_other_shape_types_keep_their_existing_anchors(self):
        label_size = QtCore.QSizeF(40, 18)
        transform = QtGui.QTransform()
        transform.scale(2, 2)

        circle = Shape(label="circle", shape_type="circle", attributes={})
        circle.points = [QtCore.QPointF(30, 40), QtCore.QPointF(35, 40)]
        circle_rect = self.canvas._legacy_shape_label_rect(
            circle, label_size, transform
        )
        self.assertEqual(circle_rect.center(), QtCore.QPointF(60, 80))

        for shape_type in ("point", "line", "linestrip"):
            with self.subTest(shape_type=shape_type):
                shape = Shape(
                    label=shape_type,
                    shape_type=shape_type,
                    attributes={},
                )
                shape.points = [QtCore.QPointF(30, 40)]
                rect = self.canvas._legacy_shape_label_rect(
                    shape, label_size, transform
                )
                self.assertEqual(rect.left(), 60 + Shape.point_size)
                self.assertEqual(rect.top(), 80 - 15)

    def test_mask_toggle_does_not_change_label_position(self):
        shape = self._make_shape()
        shape_rect = QtCore.QRectF(50, 50, 40, 40)

        self.canvas.show_masks = False
        without_masks = self._place(shape, shape_rect)
        self.canvas.show_masks = True
        with_masks = self._place(shape, shape_rect)

        self.assertEqual(without_masks, with_masks)

    def test_layout_does_not_change_serialized_shape_data(self):
        shape = self._make_shape(
            group_id=7,
            score=0.9,
            description="note",
            attributes={"state": "ok"},
        )
        before = copy.deepcopy(shape.to_dict())
        transform = QtGui.QTransform()
        shape_rect = self.canvas._shape_label_screen_rect(shape, transform)

        label_text = self.canvas._build_shape_label_text(shape)
        self._place(shape, shape_rect, self._label_size(label_text))

        self.assertEqual(shape.to_dict(), before)
        self.assertNotIn("label_position", shape.to_dict())

    def test_empty_shape_is_skipped_safely(self):
        shape = Shape(label="empty", shape_type="rectangle", attributes={})

        screen_rect = self.canvas._shape_label_screen_rect(
            shape, QtGui.QTransform()
        )

        self.assertIsNone(screen_rect)

    def test_label_drawing_restores_painter_transform(self):
        pixmap = QtGui.QPixmap(200, 160)
        pixmap.fill(QtGui.QColor("#FFFFFF"))
        self.canvas.load_pixmap(pixmap)
        self.canvas.load_shapes([self._make_shape()])
        image = QtGui.QImage(
            400, 320, QtGui.QImage.Format_ARGB32_Premultiplied
        )
        painter = QtGui.QPainter(image)
        painter.scale(2, 2)
        painter.translate(3, 4)
        original_transform = painter.transform()

        self.canvas._draw_shape_labels(painter)

        self.assertEqual(painter.transform(), original_transform)
        painter.end()

    def test_label_drawing_emits_text_command(self):
        pixmap = QtGui.QPixmap(200, 160)
        pixmap.fill(QtGui.QColor("#FFFFFF"))
        self.canvas.load_pixmap(pixmap)
        self.canvas.load_shapes([self._make_shape()])
        painter = LabelPainterSpy()

        self.canvas._draw_shape_labels(painter)

        self.assertEqual(len(painter.draw_text_calls), 1)
        self.assertEqual(painter.draw_text_calls[0][2], "target")

    def test_canvas_grab_renders_label_outside_box(self):
        self._prepare_render_canvas()
        color = QtGui.QColor(12, 34, 56, 255)
        shape = self._make_shape(line_color=color)
        self.canvas.load_shapes([shape])
        label_rect = self._rendered_label_rect(shape)

        self.canvas.show()
        APPLICATION.processEvents()
        image = self.canvas.grab().toImage()
        sample = QtCore.QPoint(
            int(label_rect.left()) + 1,
            int(label_rect.top()) + 1,
        )

        self.assertEqual(image.pixelColor(sample), color)
        self.assertFalse(label_rect.intersects(shape.bounding_rect()))

    def test_canvas_grab_renders_named_small_box_labels_externally(self):
        Shape.point_size = 10
        self._prepare_render_canvas(width=360, height=240)
        shape_specs = (
            ("cal_hole1", (90, 80, 100, 90), QtGui.QColor(12, 90, 40)),
            ("cal_hole2", (180, 3, 190, 13), QtGui.QColor(170, 90, 5)),
            (
                "target_hole",
                (280, 185, 292, 197),
                QtGui.QColor(20, 80, 170),
            ),
        )
        shapes = []
        label_rects = []
        for label, bounds, color in shape_specs:
            shape = self._make_shape(
                label=label,
                bounds=bounds,
                line_color=color,
            )
            shape.selected = True
            shapes.append(shape)
            label_rects.append(self._rendered_label_rect(shape))
        self.canvas.load_shapes(shapes)

        self.canvas.show()
        APPLICATION.processEvents()
        image = self.canvas.grab().toImage()

        for shape, label_rect in zip(shapes, label_rects):
            with self.subTest(label=shape.label):
                shape_rect = shape.bounding_rect()
                vertex_protection = shape_rect.adjusted(-5, -5, 5, 5)
                sample = QtCore.QPoint(
                    int(label_rect.left()) + 1,
                    int(label_rect.top()) + 1,
                )
                self.assertEqual(image.pixelColor(sample), shape.line_color)
                self.assertFalse(label_rect.intersects(shape_rect))
                self.assertFalse(label_rect.intersects(vertex_protection))

    def test_canvas_grab_honors_both_visibility_controls(self):
        self._prepare_render_canvas()
        color = QtGui.QColor(90, 40, 10, 255)
        shape = self._make_shape(line_color=color)
        self.canvas.load_shapes([shape])
        label_rect = self._rendered_label_rect(shape)
        sample = QtCore.QPoint(
            int(label_rect.left()) + 1,
            int(label_rect.top()) + 1,
        )
        self.canvas.show()

        shape.visible = False
        self.canvas.update()
        APPLICATION.processEvents()
        hidden_image = self.canvas.grab().toImage()
        self.assertEqual(
            hidden_image.pixelColor(sample), QtGui.QColor("#FFFFFF")
        )

        shape.visible = True
        self.canvas.set_shape_visible(shape, False)
        APPLICATION.processEvents()
        filtered_image = self.canvas.grab().toImage()
        self.assertEqual(
            filtered_image.pixelColor(sample), QtGui.QColor("#FFFFFF")
        )

    def test_canvas_grab_skips_internal_auto_labels(self):
        self._prepare_render_canvas()
        color = QtGui.QColor(30, 80, 120, 255)
        shape = self._make_shape(label="AUTOLABEL_OBJECT", line_color=color)
        self.canvas.load_shapes([shape])
        regular_shape = self._make_shape(label="target", line_color=color)
        label_rect = self._rendered_label_rect(regular_shape)
        sample = QtCore.QPoint(
            int(label_rect.left()) + 1,
            int(label_rect.top()) + 1,
        )

        self.canvas.show()
        APPLICATION.processEvents()
        image = self.canvas.grab().toImage()

        self.assertEqual(image.pixelColor(sample), QtGui.QColor("#FFFFFF"))


if __name__ == "__main__":
    unittest.main()
