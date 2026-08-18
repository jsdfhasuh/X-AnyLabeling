"""This module defines Canvas widget - the core component for drawing image labels"""

import math
from copy import deepcopy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QWheelEvent

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.views.labeling.utils.colormap import label_colormap

from .. import utils
from .. import pose_config as pose_config_utils
from ..shape import Shape

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

AUTO_DECODE_DELAY_MS = 100
MAX_AUTO_DECODE_MARKS = 42
AUTO_DECODE_MOVE_THRESHOLD = 5.0
MOVE_SPEED = 5.0
LARGE_ROTATION_INCREMENT = math.radians(1.0)
SMALL_ROTATION_INCREMENT = math.radians(0.1)

LABEL_COLORMAP = label_colormap()

LABEL_FONT_POINT_SIZE = 8.0
LABEL_PADDING_X_PX = 4
LABEL_PADDING_Y_PX = 2
LABEL_GAP_PX = 3
LABEL_EDGE_MARGIN_PX = 1


class Canvas(
    QtWidgets.QWidget
):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Canvas widget to handle label drawing"""

    zoom_request = QtCore.pyqtSignal(int, QtCore.QPoint)
    scroll_request = QtCore.pyqtSignal(float, int, int)
    # [Feature] support for automatically switching to editing mode
    # when the cursor moves over an object
    mode_changed = QtCore.pyqtSignal()
    new_shape = QtCore.pyqtSignal()
    show_shape = QtCore.pyqtSignal(int, int, QtCore.QPointF)
    selection_changed = QtCore.pyqtSignal(list)
    shape_moved = QtCore.pyqtSignal()
    shape_rotated = QtCore.pyqtSignal()
    drawing_polygon = QtCore.pyqtSignal(bool)
    vertex_selected = QtCore.pyqtSignal(bool)
    auto_labeling_marks_updated = QtCore.pyqtSignal(list)
    auto_decode_requested = QtCore.pyqtSignal(list)
    auto_decode_finish_requested = QtCore.pyqtSignal()
    shape_hover_changed = QtCore.pyqtSignal()
    split_position_changed = QtCore.pyqtSignal(float)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, rotation, line, or point
    _create_mode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                f"Unexpected value for double_click event: {self.double_click}"
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self.wheel_rectangle_editing = kwargs.pop(
            "wheel_rectangle_editing", {}
        )
        self.enable_wheel_rectangle_editing = self.wheel_rectangle_editing.get(
            "enable", False
        )
        self.rect_adjust_step = self.wheel_rectangle_editing.get(
            "adjust_step", 2.0
        )
        self.rect_scale_step = self.wheel_rectangle_editing.get(
            "scale_step", 0.05
        )
        self.attributes_config = kwargs.pop("attributes", {})
        self.rotation_config = kwargs.pop("rotation", {})
        self.mask_config = kwargs.pop("mask", {})
        self.parent = kwargs.pop("parent")
        super().__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.is_auto_labeling = False
        self.is_move_editing = False
        self.auto_labeling_mode: AutoLabelingMode = None
        self.shapes = []
        self.shapes_backups = []
        self.current = None
        self.selected_shapes = []  # save the selected shapes here
        self.selected_shapes_copy = []
        # self.line represents:
        #   - create_mode == 'polygon': edge from last point to current
        #   - create_mode == 'rectangle': diagonal line of the rectangle
        #   - create_mode == 'line': the line
        #   - create_mode == 'point': the point
        self.line = Shape()
        self.prev_point = QtCore.QPoint()
        self.prev_pan_point = QtCore.QPoint()
        self.prev_move_point = QtCore.QPoint()
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hide_backround = False
        self.hide_backround = False
        self.h_hape = None
        self.prev_h_shape = None
        self.h_vertex = None
        self.prev_h_vertex = None
        self.h_edge = None
        self.prev_h_edge = None
        self.moving_shape = False
        self.rotating_shape = False
        self.snapping = True
        self.h_shape_is_selected = False
        self.h_shape_is_hovered = None
        self.allowed_oop_shape_types = ["rotation"]
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.show_groups = False
        self.show_masks = True
        self.show_texts = True
        self.show_labels = True
        self.show_scores = True
        self.show_degrees = False
        self.show_attributes = True
        self.show_linking = True

        # Set cross line options.
        self.cross_line_show = True
        self.cross_line_width = 2.0
        self.cross_line_color = "#00FF00"
        self.cross_line_opacity = 0.5

        # Set attributes color options.
        self.attr_background_color = self.attributes_config.get(
            "background_color", [33, 33, 33, 255]
        )
        self.attr_border_color = self.attributes_config.get(
            "border_color", [66, 66, 66, 255]
        )
        self.attr_text_color = self.attributes_config.get(
            "text_color", [33, 150, 243, 255]
        )

        # Set rotation increment options.
        self.large_rotation_increment = math.radians(
            self.rotation_config.get("large_increment", 1.0)
        )
        self.small_rotation_increment = math.radians(
            self.rotation_config.get("small_increment", 0.1)
        )

        # Set mask opacity options.
        self.mask_opacity = self.mask_config.get("opacity", 80)

        self.is_loading = False
        self.sequence_edit_locked = False
        self.loading_text = self.tr("Loading...")
        self.loading_angle = 0

        # Auto mask decode mode
        self.auto_decode_mode = False
        self.auto_decode_timer = QTimer()
        self.auto_decode_timer.timeout.connect(self.on_auto_decode_timeout)
        self.auto_decode_timer.setSingleShot(True)
        self.auto_decode_tracklet = []
        self.last_mouse_pos = None
        self.pose_config = None
        self.pose_creation_class = None

        # Compare view support
        self.compare_pixmap = None
        self.split_position = 0.5

    def set_loading(self, is_loading: bool, loading_text: str = None):
        """Set loading state"""
        self.is_loading = is_loading
        if loading_text:
            self.loading_text = loading_text
        self.update()

    def set_auto_labeling_mode(self, mode: AutoLabelingMode):
        """Set auto labeling mode"""
        if mode == AutoLabelingMode.NONE:
            self.is_auto_labeling = False
            self.auto_labeling_mode = mode
        else:
            self.is_auto_labeling = True
            self.auto_labeling_mode = mode
            self.create_mode = mode.shape_type
            self.parent.toggle_draw_mode(
                False, mode.shape_type, disable_auto_labeling=False
            )

    def set_sequence_edit_locked(self, locked):
        """Block shape mutations while preserving wheel zoom and canvas pan."""

        self.sequence_edit_locked = bool(locked)
        if self.sequence_edit_locked:
            self.is_move_editing = False
            self.moving_shape = False
            self.rotating_shape = False
            self.selected_shapes_copy = []
            self.auto_decode_timer.stop()
            self.restore_cursor()

    def set_pose_config(self, pose_config):
        """Set optional pose skeleton drawing config."""
        self.pose_config = pose_config
        self.update()

    def set_pose_creation_class(self, class_name):
        """Enable drag-to-create pose rectangle mode for a class."""
        self.pose_creation_class = class_name

    def clear_pose_creation_class(self):
        """Disable drag-to-create pose rectangle mode."""
        self.pose_creation_class = None

    def _is_pose_rectangle(self, shape):
        if not self.pose_config:
            return False
        if shape is None or shape.group_id is None:
            return False
        if shape.shape_type != "rectangle":
            return False
        return shape.label in (self.pose_config.get("classes") or {})

    @staticmethod
    def _shape_rect_tuple(shape):
        rect = shape.bounding_rect()
        return (rect.left(), rect.top(), rect.right(), rect.bottom())

    def _apply_pose_rectangle_transform(
        self, rectangle_shape, old_rect, excluded_shapes=None
    ):
        if old_rect is None or not self._is_pose_rectangle(rectangle_shape):
            return
        excluded_shapes = set(excluded_shapes or [])
        new_rect = self._shape_rect_tuple(rectangle_shape)
        keypoint_names = set(
            (self.pose_config.get("classes") or {}).get(
                rectangle_shape.label, []
            )
        )
        for shape in self.shapes:
            if shape in excluded_shapes:
                continue
            if shape.group_id != rectangle_shape.group_id:
                continue
            if shape.shape_type != "point" or not shape.points:
                continue
            if shape.label not in keypoint_names:
                continue
            old_point = [shape.points[0].x(), shape.points[0].y()]
            new_point = pose_config_utils.map_point_between_rects(
                old_point, old_rect, new_rect
            )
            shape.points[0] = QtCore.QPointF(new_point[0], new_point[1])

    def _finalise_pose_rectangle(self, pos):
        if (
            not self.current
            or self.create_mode != "rectangle"
            or not self.pose_creation_class
        ):
            return False

        init_pos = self.current[0]
        target_pos = pos
        if len(self.line.points) > 1 and self.line[1] != self.line[0]:
            target_pos = self.line[1]
        if abs(init_pos.x() - target_pos.x()) < 1:
            self.current = None
            self.set_hiding(False)
            self.drawing_polygon.emit(False)
            self.update()
            return False
        if abs(init_pos.y() - target_pos.y()) < 1:
            self.current = None
            self.set_hiding(False)
            self.drawing_polygon.emit(False)
            self.update()
            return False

        if self.current.reach_max_points() is False:
            min_x = init_pos.x()
            min_y = init_pos.y()
            max_x = target_pos.x()
            max_y = target_pos.y()
            self.current.add_point(QtCore.QPointF(max_x, min_y))
            self.current.add_point(QtCore.QPointF(max_x, max_y))
            self.current.add_point(QtCore.QPointF(min_x, max_y))
        before_count = len(self.shapes)
        self.finalise()
        return len(self.shapes) > before_count

    def set_auto_decode_mode(self, enabled: bool):
        """Set auto decode mode"""
        if self.auto_decode_mode and not enabled:
            self.reset_auto_decode_state()
        self.auto_decode_mode = enabled

    def reset_auto_decode_state(self):
        """Reset auto decode state"""
        if self.auto_decode_timer.isActive():
            self.auto_decode_timer.stop()
        self.auto_decode_tracklet.clear()
        self.last_mouse_pos = None

    def fill_drawing(self):
        """Get option to fill shapes by color"""
        return self._fill_drawing

    def set_fill_drawing(self, value):
        """Set shape filling option"""
        self._fill_drawing = value

    @property
    def create_mode(self):
        """Create mode for canvas - Modes: polygon, rectangle, rotation, circle,..."""
        return self._create_mode

    @create_mode.setter
    def create_mode(self, value):
        """Set create mode for canvas"""
        if value not in [
            "polygon",
            "rectangle",
            "rotation",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            raise ValueError(f"Unsupported create_mode: {value}")
        self._create_mode = value

    def store_shapes(self):
        """Store shapes for restoring later (Undo feature)"""
        shapes_backup = []
        for shape in self.shapes:
            shapes_backup.append(shape.copy())
        if len(self.shapes_backups) > self.num_backups:
            self.shapes_backups = self.shapes_backups[-self.num_backups - 1 :]
        self.shapes_backups.append(shapes_backup)

    def store_moving_shape(self):
        """Store a moving shape"""
        if self.moving_shape:
            moving_shapes = (
                [self.h_hape] + self.selected_shapes
                if self.h_hape and self.h_hape not in self.selected_shapes
                else self.selected_shapes.copy()
            )
            for shape in moving_shapes:
                if shape in self.shapes:
                    index = self.shapes.index(shape)
                    if (
                        len(self.shapes_backups) > 0
                        and index < len(self.shapes_backups[-1])
                        and self.shapes_backups[-1][index].points
                        != self.shapes[index].points
                    ):
                        self.store_shapes()
                        self.shape_moved.emit()
                        break

            self.moving_shape = False

    def clip_rectangle_to_pixmap(self, shape):
        """Clip rectangle shape to pixmap boundaries"""
        if self.pixmap is None or shape.shape_type != "rectangle":
            return True

        w, h = self.pixmap.width(), self.pixmap.height()
        points = shape.points

        if len(points) != 4:
            return True

        x_coords = [p.x() for p in points]
        y_coords = [p.y() for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        clipped_min_x = max(0, min_x)
        clipped_min_y = max(0, min_y)
        clipped_max_x = min(w - 1, max_x)
        clipped_max_y = min(h - 1, max_y)

        if clipped_max_x <= clipped_min_x or clipped_max_y <= clipped_min_y:
            return False

        shape.points = [
            QtCore.QPointF(clipped_min_x, clipped_min_y),
            QtCore.QPointF(clipped_max_x, clipped_min_y),
            QtCore.QPointF(clipped_max_x, clipped_max_y),
            QtCore.QPointF(clipped_min_x, clipped_max_y),
        ]
        return True

    @property
    def is_shape_restorable(self):
        """Check if shape can be restored from backup"""
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapes_backups) < 2:
            return False
        return True

    def restore_shape(self):
        """Restore/Undo a shape"""
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::load_shapes and our own Canvas::load_shapes function.
        if not self.is_shape_restorable:
            return
        self.shapes_backups.pop()  # latest

        # The application will eventually call Canvas.load_shapes which will
        # push this right back onto the stack.
        shapes_backup = self.shapes_backups.pop()
        self.shapes = shapes_backup
        self.selected_shapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, _):
        """Mouse enter event"""
        self.override_cursor(self._cursor)

    def leaveEvent(self, _):
        """Mouse leave event"""
        self.store_moving_shape()
        self.un_highlight()
        self.restore_cursor()
        self.shape_hover_changed.emit()

    def focusOutEvent(self, _):
        """Window out of focus event"""
        self.restore_cursor()

    def is_visible(self, shape):
        """Check if a shape is visible"""
        return self.visible.get(shape, True)

    def drawing(self):
        """Check if user is drawing (mode==CREATE)"""
        return self.mode == self.CREATE

    def editing(self):
        """Check if user is editing (mode==EDIT)"""
        return self.mode == self.EDIT

    def set_auto_labeling(self, value=True):
        """Set auto labeling mode"""
        self.is_auto_labeling = value
        if self.auto_labeling_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
            self.parent.toggle_draw_mode(
                True, "rectangle", disable_auto_labeling=True
            )

    def get_mode(self):
        """Get current mode"""
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            return self.tr("Auto Labeling")
        if self.mode == self.CREATE:
            return self.tr("Drawing")
        elif self.mode == self.EDIT:
            return self.tr("Editing")
        else:
            return self.tr("Unknown")

    def set_editing(self, value=True):
        """Set editing mode. Editing is set to False, user is drawing"""
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.un_highlight()
            self.deselect_shape()
            self.is_move_editing = False
            self.shape_hover_changed.emit()

    def un_highlight(self):
        """Unhighlight shape/vertex/edge"""
        if self.h_hape:
            self.h_hape.highlight_clear()
            self.update()
        self.prev_h_shape = self.h_hape
        self.prev_h_vertex = self.h_vertex
        self.prev_h_edge = self.h_edge
        self.h_hape = self.h_vertex = self.h_edge = None

    def selected_vertex(self):
        """Check if selected a vertex"""
        return self.h_vertex is not None

    def selected_edge(self):
        """Check if selected an edge"""
        return self.h_edge is not None

    def _should_trigger_auto_decode(self, pos):
        """Check if mouse movement exceeds threshold to trigger auto decode"""
        if not self.auto_decode_tracklet:
            return True

        last_point = self.auto_decode_tracklet[-1]["data"]
        distance = (
            (pos.x() - last_point[0]) ** 2 + (pos.y() - last_point[1]) ** 2
        ) ** 0.5
        return distance >= AUTO_DECODE_MOVE_THRESHOLD

    # QT Overload
    def mouseMoveEvent(self, ev):  # noqa: C901
        """Update line with last point and current coordinates"""
        if self.is_loading:
            return
        try:
            pos = self.transform_pos(ev.localPos())
        except AttributeError:
            return

        if self.sequence_edit_locked:
            self.prev_move_point = pos
            if QtCore.Qt.LeftButton & ev.buttons():
                if (
                    self.pixmap
                    and self.pixmap.width()
                    and self.pixmap.height()
                ):
                    delta = ev.localPos() - self.prev_pan_point
                    self.prev_pan_point = ev.localPos()
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Vertical,
                        1,
                    )
            return

        prev_hover_shape = self.h_hape
        self.prev_move_point = pos
        self.repaint()

        # Handle auto decode mode
        if (
            self.auto_decode_mode
            and self.is_auto_labeling
            and self.auto_decode_tracklet
        ):
            if self._should_trigger_auto_decode(pos):
                self.last_mouse_pos = pos
                if not self.auto_decode_timer.isActive():
                    self.auto_decode_timer.start(AUTO_DECODE_DELAY_MS)

        # Polygon drawing.
        if self.drawing():
            line_color = utils.hex_to_rgb(self.cross_line_color)
            self.line.line_color = QtGui.QColor(*line_color)
            self.line.shape_type = self.create_mode

            if not self.current:
                self.override_cursor(CURSOR_DRAW)
                return

            if self.create_mode == "rectangle":
                shape_width = int(abs(self.current[0].x() - pos.x()))
                shape_height = int(abs(self.current[0].y() - pos.y()))
                self.show_shape.emit(shape_width, shape_height, pos)

            color = QtGui.QColor(0, 0, 255)
            if self.out_off_pixmap(pos) and self.create_mode not in [
                "rectangle",
                "rotation",
            ]:
                pos = self.intersection_point(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.create_mode == "polygon"
                and self.close_enough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            elif (
                self.create_mode == "rotation"
                and len(self.current) > 0
                and self.close_enough(pos, self.current[0])
            ):
                pos = self.current[0]
                color = self.current.line_color
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            else:
                self.override_cursor(CURSOR_DRAW)
            if self.create_mode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.create_mode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.create_mode == "rotation":
                self.line[1] = pos
                self.line.line_color = color
            elif self.create_mode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.create_mode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.create_mode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.repaint()
            self.current.highlight_clear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selected_shapes_copy and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes_copy, pos)
                self.repaint()
            elif self.selected_shapes:
                self.selected_shapes_copy = [
                    s.copy() for s in self.selected_shapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selected_vertex():
                self.is_move_editing = False
                try:
                    self.bounded_move_vertex(pos)
                    self.repaint()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_hape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
            elif self.selected_shapes and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes, pos)
                self.repaint()
                self.moving_shape = True
                if self.selected_shapes[-1].shape_type == "rectangle":
                    p1 = self.selected_shapes[-1][0]
                    p2 = self.selected_shapes[-1][2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
            else:
                if (
                    self.pixmap
                    and self.pixmap.width()
                    and self.pixmap.height()
                ):
                    self.override_cursor(CURSOR_MOVE)
                    delta = ev.localPos() - self.prev_pan_point
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Vertical,
                        1,
                    )
                    self.repaint()
            return

        if self.editing() and self.is_move_editing:
            self.override_cursor(CURSOR_MOVE)
            if self.selected_vertex():
                try:
                    self.bounded_move_vertex(pos)
                    self.repaint()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_hape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
            else:
                self.is_move_editing = False

            return

        self.show_shape.emit(-1, -1, pos)

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        # self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.is_visible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearest_vertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearest_edge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex = index
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                shape.highlight_vertex(index, shape.MOVE_VERTEX)
                self.override_cursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click & drag to move point of shape '%s'")
                    % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            if index_edge is not None and shape.can_add_point():
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge = index_edge
                self.override_cursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click to create point of shape '%s'")
                    % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            shape_hit = False
            if shape.shape_type in ["point", "line", "linestrip"]:
                nearest_index = shape.nearest_vertex(
                    pos, self.epsilon * 3 / self.scale
                )
                if nearest_index is not None:
                    shape_hit = True
            elif len(shape.points) > 1 and shape.contains_point(pos):
                shape_hit = True

            if shape_hit:
                if self.selected_vertex():
                    self.h_hape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_hape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                if shape.group_id and shape.shape_type == "rectangle":
                    tooltip_text = "Click & drag to move shape '{label} {group_id}'".format(
                        label=shape.label, group_id=shape.group_id
                    )
                    self.setToolTip(self.tr(tooltip_text))
                else:
                    self.setToolTip(
                        self.tr("Click & drag to move shape '%s'")
                        % shape.label
                    )
                self.setStatusTip(self.toolTip())
                self.override_cursor(CURSOR_GRAB)
                # [Feature] Automatically highlight shape when the mouse is moved inside it
                if self.h_shape_is_hovered:
                    group_mode = (
                        int(ev.modifiers()) == QtCore.Qt.ControlModifier
                    )
                    self.select_shape_point(
                        pos, multiple_selection_mode=group_mode
                    )
                self.update()

                if shape.shape_type == "rectangle":
                    p1 = self.h_hape[0]
                    p2 = self.h_hape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_width, shape_height, pos)
                break
        else:  # Nothing found, clear highlights, reset state.
            self.un_highlight()
            self.override_cursor(CURSOR_DEFAULT)
        self.vertex_selected.emit(self.h_vertex is not None)

        if prev_hover_shape != self.h_hape:
            self.shape_hover_changed.emit()

    def add_point_to_edge(self):
        """Add a point to current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_edge
        point = self.prev_move_point
        if shape is None or index is None or point is None:
            return
        shape.insert_point(index, point)
        shape.highlight_vertex(index, shape.MOVE_VERTEX)
        self.h_hape = shape
        self.h_vertex = index
        self.h_edge = None
        self.moving_shape = True

    def remove_selected_point(self):
        """Remove a point from current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_vertex
        if shape is None or index is None:
            return
        shape.remove_point(index)
        shape.highlight_clear()
        self.h_hape = shape
        self.prev_h_vertex = None
        self.moving_shape = True  # Save changes

    def on_auto_decode_timeout(self):
        """Handle auto decode timeout"""
        if (
            not self.auto_decode_mode
            or self.auto_labeling_mode.shape_type != AutoLabelingMode.POINT
        ):
            return

        flag = -1
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            flag = 1
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            flag = 0
        if flag == -1:
            return

        if self.auto_decode_mode and self.last_mouse_pos:
            if len(self.auto_decode_tracklet) >= MAX_AUTO_DECODE_MARKS:
                self.auto_decode_tracklet.pop(0)

            marks = {
                "type": "point",
                "data": [
                    int(self.last_mouse_pos.x()),
                    int(self.last_mouse_pos.y()),
                ],
                "label": flag,
            }
            self.auto_decode_tracklet.append(marks)
            self.auto_decode_requested.emit(self.auto_decode_tracklet)

    # QT Overload
    def mousePressEvent(self, ev):  # noqa: C901
        """Mouse press event"""
        if self.is_loading:
            return
        if self.sequence_edit_locked:
            if ev.button() == QtCore.Qt.LeftButton:
                self.prev_pan_point = ev.localPos()
            return
        pos = self.transform_pos(ev.localPos())

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.create_mode == "polygon":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode in ["circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.create_mode == "rectangle":
                        if self.current.reach_max_points() is False:
                            init_pos = self.current[0]
                            min_x = init_pos.x()
                            min_y = init_pos.y()
                            target_pos = self.line[1]
                            max_x = target_pos.x()
                            max_y = target_pos.y()
                            self.current.add_point(
                                QtCore.QPointF(max_x, min_y)
                            )
                            self.current.add_point(target_pos)
                            self.current.add_point(
                                QtCore.QPointF(min_x, max_y)
                            )
                            self.finalise()
                    elif self.create_mode == "rotation":
                        initPos = self.current[0]
                        minX = initPos.x()
                        minY = initPos.y()
                        targetPos = self.line[1]
                        maxX = targetPos.x()
                        maxY = targetPos.y()
                        self.current.add_point(QtCore.QPointF(maxX, minY))
                        self.current.add_point(targetPos)
                        self.current.add_point(QtCore.QPointF(minX, maxY))
                        self.current.add_point(initPos)
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode == "linestrip":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    # [Feature] support for automatically switching to editing mode
                    # when the cursor moves over an object
                    if (
                        self.create_mode
                        in ["rectangle", "rotation", "circle", "line", "point"]
                        and not self.is_auto_labeling
                        and not self.current
                    ):
                        self.prev_pan_point = ev.localPos()
                        self.mode_changed.emit()
                elif not self.out_off_pixmap(pos):
                    # Handle auto decode mode first click
                    if self.auto_decode_mode and self.is_auto_labeling:
                        if (
                            self.auto_labeling_mode.shape_type
                            == AutoLabelingMode.POINT
                        ):
                            self.last_mouse_pos = pos
                            self.on_auto_decode_timeout()
                            return

                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(pos)
                    if self.create_mode == "point":
                        self.finalise()
                    else:
                        if self.create_mode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.set_hiding()
                        self.drawing_polygon.emit(True)
                        self.update()
                elif self.out_off_pixmap(pos) and self.create_mode in [
                    "rectangle",
                    "rotation",
                ]:
                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(pos)
                    self.line.points = [pos, pos]
                    self.set_hiding()
                    self.drawing_polygon.emit(True)
                    self.update()
            elif self.editing():
                if self.selected_edge():
                    self.add_point_to_edge()
                elif (
                    self.selected_vertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                    and self.h_hape.shape_type
                    not in ["rectangle", "rotation", "line"]
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.remove_selected_point()

                if self.selected_vertex():
                    self.is_move_editing = not self.is_move_editing
                    if self.is_move_editing:
                        self.override_cursor(CURSOR_MOVE)
                    else:
                        self.override_cursor(CURSOR_POINT)

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.select_shape_point(
                    pos, multiple_selection_mode=group_mode
                )
                self.prev_point = pos
                self.prev_pan_point = ev.localPos()
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selected_shapes or (
                self.h_hape is not None
                and self.h_hape not in self.selected_shapes
            ):
                self.select_shape_point(
                    pos, multiple_selection_mode=group_mode
                )
                self.repaint()
            self.prev_point = pos

    # QT Overload
    def mouseReleaseEvent(self, ev):
        """Mouse release event"""
        if self.is_loading:
            return
        if self.sequence_edit_locked:
            return

        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selected_shapes_copy) > 0]
            self.restore_cursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selected_shapes_copy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selected_shapes_copy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if (
                self.drawing()
                and self.pose_creation_class
                and self.create_mode == "rectangle"
                and self.current
            ):
                try:
                    pos = self.transform_pos(ev.localPos())
                except AttributeError:
                    return
                if self.out_off_pixmap(pos):
                    pos = self.intersection_point(self.current[0], pos)
                if self._finalise_pose_rectangle(pos):
                    self.mode_changed.emit()
                return

            if self.editing():
                if (
                    self.h_hape is not None
                    and self.h_shape_is_selected
                    and not self.moving_shape
                ):
                    self.selection_changed.emit(
                        [x for x in self.selected_shapes if x != self.h_hape]
                    )

        self.store_moving_shape()

    def end_move(self, copy):
        """End of move"""
        assert self.selected_shapes and self.selected_shapes_copy
        assert len(self.selected_shapes_copy) == len(self.selected_shapes)
        if copy:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.shapes.append(shape)
                self.selected_shapes[i].selected = False
                self.selected_shapes[i] = shape
        else:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.selected_shapes[i].points = shape.points
        self.selected_shapes_copy = []
        self.repaint()
        self.store_shapes()
        return True

    def hide_background_shapes(self, value):
        """Set hide background - hide other shapes when some shapes are selected"""
        self.hide_backround = value
        if self.selected_shapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.set_hiding(True)
            self.update()

    def set_hiding(self, enable=True):
        """Set background hiding"""
        self._hide_backround = self.hide_backround if enable else False

    def can_close_shape(self):
        """Check if a shape can be closed (number of points > 2)"""
        return self.drawing() and self.current and len(self.current) > 2

    # QT Overload
    def mouseDoubleClickEvent(self, _):
        """Mouse double click event"""
        if self.is_loading or self.sequence_edit_locked:
            return

        # Handle auto decode mode double click to finish
        if (
            self.auto_decode_mode
            and self.is_auto_labeling
            and self.auto_decode_tracklet
        ):
            self.auto_decode_finish_requested.emit()
            return

        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.can_close_shape()
            and len(self.current) > 3
        ):
            self.current.pop_point()
            self.finalise()

    def select_shapes(self, shapes):
        """Select some shapes"""
        self.set_hiding()
        self.selection_changed.emit(shapes)
        self.update()

    def select_shape_point(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selected_vertex():  # A vertex is marked for selection.
            index, shape = self.h_vertex, self.h_hape
            shape.highlight_vertex(index, shape.MOVE_VERTEX)
            if shape.shape_type == "rotation":
                self.set_hiding()
                if shape not in self.selected_shapes:
                    if multiple_selection_mode:
                        self.selection_changed.emit(
                            self.selected_shapes + [shape]
                        )
                    else:
                        self.selection_changed.emit([shape])
                    self.h_shape_is_selected = False
                else:
                    self.h_shape_is_selected = True
                self.calculate_offsets(point)
                return

        else:
            for shape in reversed(self.shapes):
                shape_selectable = False
                if shape.shape_type in ["point", "line", "linestrip"]:
                    if (
                        self.is_visible(shape)
                        and shape.nearest_vertex(
                            point, self.epsilon * 3 / self.scale
                        )
                        is not None
                    ):
                        shape_selectable = True
                elif (
                    self.is_visible(shape)
                    and len(shape.points) > 1
                    and shape.contains_point(point)
                ):
                    shape_selectable = True

                if shape_selectable:
                    self.set_hiding()
                    if shape not in self.selected_shapes:
                        if multiple_selection_mode:
                            self.selection_changed.emit(
                                self.selected_shapes + [shape]
                            )
                        else:
                            self.selection_changed.emit([shape])
                        self.h_shape_is_selected = False
                    else:
                        self.h_shape_is_selected = True
                    self.calculate_offsets(point)
                    return
        self.deselect_shape()

    def calculate_offsets(self, point):
        """Calculate offsets of a point to pixmap borders"""
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selected_shapes:
            rect = s.bounding_rect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def get_adjoint_points(self, theta, p3, p1, index):
        a1 = math.tan(theta)
        if a1 == 0:
            if index % 2 == 0:
                p2 = QtCore.QPointF(p3.x(), p1.y())
                p4 = QtCore.QPointF(p1.x(), p3.y())
            else:
                p4 = QtCore.QPointF(p3.x(), p1.y())
                p2 = QtCore.QPointF(p1.x(), p3.y())
        else:
            a3 = a1
            a2 = -1 / a1
            a4 = -1 / a1
            b1 = p1.y() - a1 * p1.x()
            b2 = p1.y() - a2 * p1.x()
            b3 = p3.y() - a1 * p3.x()
            b4 = p3.y() - a2 * p3.x()

            if index % 2 == 0:
                p2 = self.get_cross_point(a1, b1, a4, b4)
                p4 = self.get_cross_point(a2, b2, a3, b3)
            else:
                p4 = self.get_cross_point(a1, b1, a4, b4)
                p2 = self.get_cross_point(a2, b2, a3, b3)

        return p2, p3, p4

    @staticmethod
    def get_cross_point(a1, b1, a2, b2):
        x = (b2 - b1) / (a1 - a2)
        y = (a1 * b2 - a2 * b1) / (a1 - a2)
        return QtCore.QPointF(x, y)

    def bounded_move_vertex(self, pos):
        """Move a vertex. Adjust position to be bounded by pixmap border"""
        index, shape = self.h_vertex, self.h_hape
        point = shape[index]
        old_pose_rect = (
            self._shape_rect_tuple(shape)
            if self._is_pose_rectangle(shape)
            else None
        )
        if (
            self.out_off_pixmap(pos)
            and shape.shape_type not in self.allowed_oop_shape_types
        ):
            pos = self.intersection_point(point, pos)

        if shape.shape_type == "rotation":
            sindex = (index + 2) % 4
            # Get the other 3 points after transformed
            p2, p3, p4 = self.get_adjoint_points(
                shape.direction, shape[sindex], pos, index
            )
            # if (
            #     self.out_off_pixmap(p2)
            #     or self.out_off_pixmap(p3)
            #     or self.out_off_pixmap(p4)
            # ):
            #     # No need to move if one pixal out of map
            #     return
            # Move 4 pixal one by one
            shape.move_vertex_by(index, pos - point)
            lindex = (index + 1) % 4
            rindex = (index + 3) % 4
            shape[lindex] = p2
            shape[rindex] = p4
            shape.close()
        elif shape.shape_type == "rectangle":
            shift_pos = pos - point
            shape.move_vertex_by(index, shift_pos)
            left_index = (index + 1) % 4
            right_index = (index + 3) % 4
            left_shift = None
            right_shift = None
            if index % 2 == 0:
                right_shift = QtCore.QPointF(shift_pos.x(), 0)
                left_shift = QtCore.QPointF(0, shift_pos.y())
            else:
                left_shift = QtCore.QPointF(shift_pos.x(), 0)
                right_shift = QtCore.QPointF(0, shift_pos.y())
            shape.move_vertex_by(right_index, right_shift)
            shape.move_vertex_by(left_index, left_shift)
        else:
            shape.move_vertex_by(index, pos - point)
        self._apply_pose_rectangle_transform(shape, old_pose_rect)

    def bounded_move_shapes(self, shapes, pos):
        """Move shapes. Adjust position to be bounded by pixmap border"""
        shape_types = []
        for shape in shapes:
            if shape.shape_type in self.allowed_oop_shape_types:
                shape_types.append(shape.shape_type)

        if self.out_off_pixmap(pos) and len(shape_types) == 0:
            return False  # No need to move
        if len(shape_types) > 0 and len(shapes) != len(shape_types):
            return False

        if len(shape_types) == 0:
            o1 = pos + self.offsets[0]
            if self.out_off_pixmap(o1):
                pos -= QtCore.QPoint(min(0, int(o1.x())), min(0, int(o1.y())))
            o2 = pos + self.offsets[1]
            if self.out_off_pixmap(o2):
                pos += QtCore.QPoint(
                    min(0, int(self.pixmap.width() - o2.x())),
                    min(0, int(self.pixmap.height() - o2.y())),
                )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prev_point
        if dp:
            pose_rects = {
                shape: self._shape_rect_tuple(shape)
                for shape in shapes
                if self._is_pose_rectangle(shape)
            }
            for shape in shapes:
                shape.move_by(dp)
            for shape, old_rect in pose_rects.items():
                self._apply_pose_rectangle_transform(
                    shape, old_rect, excluded_shapes=shapes
                )
            self.prev_point = pos
            return True
        return False

    def rotate_point(self, p, center, theta):
        order = p - center
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = -sinTheta * order.x() + cosTheta * order.y()
        pRes = QtCore.QPointF(center.x() + pResx, center.y() + pResy)
        return pRes

    def bounded_rotate_shapes(self, i, shape, theta):
        """Rotate shapes. Adjust position to be bounded by pixmap border"""
        new_shape = deepcopy(shape)
        if len(shape.points) == 2:
            new_shape.points[0] = shape.points[0]
            new_shape.points[1] = QtCore.QPointF(
                (shape.points[0].x() + shape.points[1].x()) / 2,
                shape.points[0].y(),
            )
            new_shape.points.append(shape.points[1])
            new_shape.points.append(
                QtCore.QPointF(
                    shape.points[1].x(),
                    (shape.points[0].y() + shape.points[1].y()) / 2,
                )
            )
        center = QtCore.QPointF(
            (new_shape.points[0].x() + new_shape.points[2].x()) / 2,
            (new_shape.points[0].y() + new_shape.points[2].y()) / 2,
        )
        for j, p in enumerate(new_shape.points):
            pos = self.rotate_point(p, center, theta)
            # TODO: Reserved for now
            # if self.out_off_pixmap(pos):
            #     return False  # No need to rotate
            new_shape.points[j] = pos
        new_shape.direction = (new_shape.direction - theta) % (2 * math.pi)
        self.selected_shapes[i].points = new_shape.points
        self.selected_shapes[i].direction = new_shape.direction
        return True

    def deselect_shape(self):
        """Deselect all shapes"""
        if self.selected_shapes:
            self.set_hiding(False)
            self.selection_changed.emit([])
            self.h_shape_is_selected = False
            self.update()

    def delete_selected(self):
        """Remove selected shapes"""
        return self.delete_shapes(self.selected_shapes)

    def delete_shapes(self, shapes):
        """Remove specific shapes from the canvas."""
        deleted_shapes = []
        selection_changed = False
        for shape in list(shapes):
            if shape in self.shapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            while shape in self.selected_shapes:
                self.selected_shapes.remove(shape)
                selection_changed = True
        if deleted_shapes:
            self.store_shapes()
        if deleted_shapes or selection_changed:
            self.update()
        return deleted_shapes

    def delete_shape(self, shape):
        """Remove a specific shape"""
        if shape in self.selected_shapes:
            self.selected_shapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.store_shapes()
        self.update()

    def duplicate_selected_shapes(self):
        """Duplicate selected shapes"""
        if self.selected_shapes:
            self.selected_shapes_copy = [
                s.copy() for s in self.selected_shapes
            ]
            self.bounded_shift_shapes(self.selected_shapes_copy)
            self.end_move(copy=True)
        return self.selected_shapes

    def bounded_shift_shapes(self, shapes):
        """
        Shift shapes by an offset. Adjust positions to be bounded
        by pixmap borders
        """
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.prev_point = point
        if not self.bounded_move_shapes(shapes, point - offset):
            self.bounded_move_shapes(shapes, point + offset)

    def _draw_pose_skeletons(self, painter):
        """Draw project pose skeleton lines without changing saved shapes."""
        if not self.pose_config:
            return
        drawable_shapes = [
            shape
            for shape in self.shapes
            if (shape.selected or not self._hide_backround)
            and self.is_visible(shape)
        ]
        segments = pose_config_utils.skeleton_segments(
            drawable_shapes, self.pose_config
        )
        if not segments:
            return

        pen = QtGui.QPen(QtGui.QColor(0, 180, 255, 220))
        pen.setStyle(Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setWidth(max(1, int(round(3.0 / Shape.scale))))
        painter.setPen(pen)
        for segment in segments:
            painter.drawLine(segment["start"], segment["end"])

    def _build_shape_label_text(self, shape):
        """Build the group, label, and optional score display text."""
        label_text = "" if shape.label is None else str(shape.label)
        if shape.group_id is not None:
            label_text = f"id:{shape.group_id} {label_text}"
        if shape.score is not None and self.show_scores:
            label_text += f" {float(shape.score):.2f}"
        return label_text

    @staticmethod
    def _shape_label_screen_rect(shape, transform):
        """Map a shape bounding rectangle into screen coordinates."""
        if not shape.points:
            return None
        try:
            shape_rect = shape.bounding_rect()
        except (AssertionError, IndexError):
            return None
        if not shape_rect.isValid() or shape_rect.isEmpty():
            return None
        screen_rect = transform.mapRect(shape_rect).normalized()
        if not screen_rect.isValid() or screen_rect.isEmpty():
            return None
        return screen_rect

    @staticmethod
    def _label_protection_distance():
        """Return the current screen-space clearance around a shape."""
        return max(LABEL_GAP_PX, Shape.point_size / 2.0) + LABEL_EDGE_MARGIN_PX

    def _shape_label_occupied_rects(self, shape, transform):
        """Map the shape's visible description and attributes to the screen."""
        try:
            shape_rect = shape.bounding_rect()
        except (AssertionError, IndexError):
            return {}

        occupied_rects = {}
        if self.show_texts and bool(getattr(shape, "description", None)):
            font_size = int(max(6.0, int(round(8.0 / Shape.scale))))
            font_metrics = QtGui.QFontMetrics(QtGui.QFont("Arial", font_size))
            text_width = font_metrics.tightBoundingRect(
                shape.description
            ).width()
            rect_width = text_width + 2 * LABEL_PADDING_X_PX
            rect_height = font_metrics.height() + 2 * LABEL_PADDING_Y_PX
            description_rect = QtCore.QRectF(
                int(shape_rect.x()),
                int(shape_rect.y() - rect_height),
                rect_width,
                rect_height,
            )
            occupied_rects["top"] = transform.mapRect(
                description_rect
            ).normalized()

        if self.show_attributes and bool(getattr(shape, "attributes", None)):
            font_size = int(max(8.0, int(round(10.0 / Shape.scale))))
            font = QtGui.QFont("Arial", font_size, QtGui.QFont.Bold)
            font_metrics = QtGui.QFontMetrics(font)
            attribute_lines = [
                f"{key}: {value}" for key, value in shape.attributes.items()
            ]
            max_width = max(
                font_metrics.tightBoundingRect(line).width()
                for line in attribute_lines
            )
            rect_width = max_width + 16
            rect_height = (
                len(attribute_lines) * font_metrics.height()
                + 2 * LABEL_PADDING_Y_PX
            )
            attributes_rect = QtCore.QRectF(
                int(shape_rect.x()),
                int(shape_rect.y() + shape_rect.height() + 1),
                rect_width,
                rect_height,
            )
            occupied_rects["bottom"] = transform.mapRect(
                attributes_rect
            ).normalized()

        return occupied_rects

    @staticmethod
    def _label_candidate_rects(
        shape_rect,
        label_size,
        protection_distance,
        prefer_top=True,
        prefer_bottom=True,
        top_occupied_rect=None,
        bottom_occupied_rect=None,
    ):
        """Return stable external label candidates and an inside fallback."""
        width = float(label_size.width())
        height = float(label_size.height())
        top_edge = shape_rect.top()
        if top_occupied_rect is not None:
            top_edge = min(top_edge, top_occupied_rect.top())
        bottom_edge = shape_rect.bottom()
        if bottom_occupied_rect is not None:
            bottom_edge = max(bottom_edge, bottom_occupied_rect.bottom())
        candidates = {
            "top": QtCore.QRectF(
                shape_rect.left(),
                top_edge - height - protection_distance,
                width,
                height,
            ),
            "bottom": QtCore.QRectF(
                shape_rect.left(),
                bottom_edge + protection_distance,
                width,
                height,
            ),
            "right": QtCore.QRectF(
                shape_rect.right() + protection_distance,
                shape_rect.top(),
                width,
                height,
            ),
            "left": QtCore.QRectF(
                shape_rect.left() - width - protection_distance,
                shape_rect.top(),
                width,
                height,
            ),
        }

        order = []
        if prefer_top:
            order.append("top")
        if prefer_bottom:
            order.append("bottom")
        order.extend(("right", "left"))
        if not prefer_top:
            order.append("top")
        if not prefer_bottom:
            order.append("bottom")

        inside = QtCore.QRectF(
            shape_rect.left(),
            shape_rect.top(),
            width,
            height,
        )
        return [candidates[name] for name in order] + [inside]

    @staticmethod
    def _rounded_label_rect(rect):
        """Round a label rectangle immediately before screen drawing."""
        return QtCore.QRectF(
            int(round(rect.x())),
            int(round(rect.y())),
            int(math.ceil(rect.width())),
            int(math.ceil(rect.height())),
        )

    def _place_shape_label(
        self,
        shape,
        shape_rect,
        label_size,
        image_rect,
        occupied_rects=None,
    ):
        """Choose a visible label position outside the shape when possible."""
        if occupied_rects is None:
            occupied_rects = {}
        prefer_top = not (
            self.show_texts and bool(getattr(shape, "description", None))
        )
        prefer_bottom = not (
            self.show_attributes and bool(getattr(shape, "attributes", None))
        )
        protection_distance = self._label_protection_distance()
        candidates = self._label_candidate_rects(
            shape_rect,
            label_size,
            protection_distance,
            prefer_top=prefer_top,
            prefer_bottom=prefer_bottom,
            top_occupied_rect=occupied_rects.get("top"),
            bottom_occupied_rect=occupied_rects.get("bottom"),
        )
        image_bounds = image_rect.normalized().adjusted(
            LABEL_EDGE_MARGIN_PX,
            LABEL_EDGE_MARGIN_PX,
            -LABEL_EDGE_MARGIN_PX,
            -LABEL_EDGE_MARGIN_PX,
        )
        if (
            label_size.width() > image_bounds.width()
            or label_size.height() > image_bounds.height()
        ):
            return None

        protected_rect = shape_rect.adjusted(
            -protection_distance,
            -protection_distance,
            protection_distance,
            protection_distance,
        )
        rounded_protected_rect = shape_rect.adjusted(
            -protection_distance,
            -protection_distance,
            protection_distance,
            protection_distance,
        )
        visible_occupied_rects = tuple(
            rect
            for rect in occupied_rects.values()
            if rect is not None and rect.isValid() and not rect.isEmpty()
        )

        for candidate in candidates[:-1]:
            if not image_bounds.contains(candidate):
                continue
            if candidate.intersects(protected_rect):
                continue
            if any(
                candidate.intersects(rect) for rect in visible_occupied_rects
            ):
                continue
            rounded = self._rounded_label_rect(candidate)
            if not image_bounds.contains(rounded):
                continue
            if rounded.intersects(rounded_protected_rect):
                continue
            if any(
                rounded.intersects(rect) for rect in visible_occupied_rects
            ):
                continue
            return rounded

        fallback = candidates[-1]
        fallback.moveLeft(
            min(
                max(fallback.left(), image_bounds.left()),
                image_bounds.right() - fallback.width(),
            )
        )
        fallback.moveTop(
            min(
                max(fallback.top(), image_bounds.top()),
                image_bounds.bottom() - fallback.height(),
            )
        )
        fallback = self._rounded_label_rect(fallback)
        if not image_bounds.contains(fallback):
            return None
        if any(fallback.intersects(rect) for rect in visible_occupied_rects):
            return None
        return fallback

    @staticmethod
    def _elide_label_text(label_text, font_metrics, image_rect):
        """Elide display text to the width currently available in the image."""
        max_text_width = int(
            math.floor(
                image_rect.width()
                - 2 * LABEL_EDGE_MARGIN_PX
                - 2 * LABEL_PADDING_X_PX
            )
        )
        if max_text_width <= 0:
            return ""
        return font_metrics.elidedText(
            label_text,
            Qt.ElideRight,
            max_text_width,
        )

    @staticmethod
    def _legacy_shape_label_rect(shape, label_size, transform):
        """Keep the established label anchors for non-bounding shapes."""
        if not shape.points:
            return None
        point = transform.map(shape.points[0])
        width = float(label_size.width())
        height = float(label_size.height())
        if shape.shape_type == "circle":
            rect = QtCore.QRectF(
                point.x() - width / 2.0,
                point.y() - height / 2.0,
                width,
                height,
            )
        elif shape.shape_type in ("line", "linestrip", "point"):
            rect = QtCore.QRectF(
                point.x() + Shape.point_size,
                point.y() - 15,
                width,
                height,
            )
        else:
            return None
        return Canvas._rounded_label_rect(rect)

    def _draw_shape_labels(self, painter):
        """Draw shape labels in screen coordinates with fixed UI sizing."""
        label_transform = painter.transform()
        image_rect = label_transform.mapRect(
            QtCore.QRectF(self.pixmap.rect())
        ).normalized()
        painter.save()
        try:
            painter.resetTransform()
            font = QtGui.QFont("Arial", int(round(LABEL_FONT_POINT_SIZE)))
            painter.setFont(font)
            font_metrics = QtGui.QFontMetrics(font)

            for shape in self.shapes:
                if not shape.visible or not self.is_visible(shape):
                    continue
                if shape.label in (
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                ):
                    continue

                label_text = self._build_shape_label_text(shape)
                if not label_text:
                    continue
                display_text = self._elide_label_text(
                    label_text, font_metrics, image_rect
                )
                if not display_text:
                    continue

                label_size = QtCore.QSizeF(
                    font_metrics.horizontalAdvance(display_text)
                    + 2 * LABEL_PADDING_X_PX,
                    font_metrics.height() + 2 * LABEL_PADDING_Y_PX,
                )
                if shape.shape_type in (
                    "rectangle",
                    "polygon",
                    "rotation",
                ):
                    shape_rect = self._shape_label_screen_rect(
                        shape, label_transform
                    )
                    if shape_rect is None:
                        continue
                    occupied_rects = self._shape_label_occupied_rects(
                        shape, label_transform
                    )
                    label_rect = self._place_shape_label(
                        shape,
                        shape_rect,
                        label_size,
                        image_rect,
                        occupied_rects,
                    )
                else:
                    label_rect = self._legacy_shape_label_rect(
                        shape, label_size, label_transform
                    )
                if label_rect is None:
                    continue

                painter.fillRect(label_rect, shape.line_color)
                painter.setPen(QtGui.QColor("#000000"))
                text_x = int(label_rect.left() + LABEL_PADDING_X_PX)
                text_y = int(
                    label_rect.top()
                    + LABEL_PADDING_Y_PX
                    + font_metrics.ascent()
                )
                painter.drawText(text_x, text_y, display_text)
        finally:
            painter.restore()

    # QT Overload
    def paintEvent(self, event):  # noqa: C901
        """Paint event for canvas"""
        if (
            self.pixmap is None
            or self.pixmap.width() == 0
            or self.pixmap.height() == 0
        ):
            super().paintEvent(event)
            return

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)

        p.scale(self.scale, self.scale)
        p.translate(self.offset_to_center())

        p.drawPixmap(0, 0, self.pixmap)

        # Draw compare view: left side shows compare image, right side shows original
        # split_position: 0 = all original, 1 = all compare
        if (
            self.compare_pixmap is not None
            and not self.compare_pixmap.isNull()
        ):
            split_x = int(self.split_position * self.pixmap.width())
            if split_x > 0:
                p.drawPixmap(
                    0,
                    0,
                    self.compare_pixmap,
                    0,
                    0,
                    split_x,
                    self.pixmap.height(),
                )

        Shape.scale = self.scale

        # Draw loading/waiting screen
        if self.is_loading:
            # Draw a semi-transparent rectangle
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 20))
            p.drawRect(self.pixmap.rect())

            # Draw a spinning wheel
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setBrush(Qt.NoBrush)
            p.save()
            p.translate(self.pixmap.width() / 2, self.pixmap.height() / 2 - 50)
            p.rotate(self.loading_angle)
            p.drawEllipse(-20, -20, 40, 40)
            p.drawLine(0, 0, 0, -20)
            p.restore()
            self.loading_angle += 30
            if self.loading_angle >= 360:
                self.loading_angle = 0

            # Draw the loading text
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setFont(QtGui.QFont("Arial", 20))
            p.drawText(
                self.pixmap.rect(),
                Qt.AlignCenter,
                self.loading_text,
            )
            p.end()
            self.update()
            return

        # Draw groups
        if self.show_groups:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.SolidLine)
            p.setPen(pen)
            grouped_shapes = {}
            for shape in self.shapes:
                if not shape.visible:
                    continue
                if shape.group_id is None:
                    continue
                if shape.group_id not in grouped_shapes:
                    grouped_shapes[shape.group_id] = []
                grouped_shapes[shape.group_id].append(shape)

            for group_id in grouped_shapes:
                shapes = grouped_shapes[group_id]
                min_x = float("inf")
                min_y = float("inf")
                max_x = 0
                max_y = 0
                for shape in shapes:
                    rect = shape.bounding_rect()
                    if shape.shape_type == "point":
                        points = shape.points[0]
                        min_x = min(min_x, points.x())
                        min_y = min(min_y, points.y())
                        max_x = max(max_x, points.x())
                        max_y = max(max_y, points.y())
                    else:
                        min_x = min(min_x, rect.x())
                        min_y = min(min_y, rect.y())
                        max_x = max(max_x, rect.x() + rect.width())
                        max_y = max(max_y, rect.y() + rect.height())
                    group_color = LABEL_COLORMAP[
                        int(group_id) % len(LABEL_COLORMAP)
                    ]
                    pen.setStyle(Qt.SolidLine)
                    pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                    pen.setColor(QtGui.QColor(*group_color))
                    p.setPen(pen)

                    # Calculate the center point of the bounding rectangle
                    cx = rect.x() + rect.width() / 2
                    cy = rect.y() + rect.height() / 2
                    triangle_radius = max(1, int(round(3.0 / Shape.scale)))

                    # Define the points of the triangle
                    triangle_points = [
                        QtCore.QPointF(cx, cy - triangle_radius),
                        QtCore.QPointF(
                            cx - triangle_radius, cy + triangle_radius
                        ),
                        QtCore.QPointF(
                            cx + triangle_radius, cy + triangle_radius
                        ),
                    ]

                    # Draw the triangle
                    p.drawPolygon(triangle_points)

                pen.setStyle(Qt.DashLine)
                pen.setWidth(max(1, int(round(1.0 / Shape.scale))))
                pen.setColor(QtGui.QColor("#EEEEEE"))
                p.setPen(pen)
                wrap_rect = QtCore.QRectF(
                    min_x, min_y, max_x - min_x, max_y - min_y
                )
                p.drawRect(wrap_rect)

        # Draw KIE linking
        if self.show_linking:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.SolidLine)
            p.setPen(pen)
            gid2point = {}
            linking_pairs = []
            group_color = (255, 128, 0)
            for shape in self.shapes:
                if not shape.visible:
                    continue

                try:
                    linking_pairs += shape.kie_linking
                except Exception:
                    pass

                if shape.group_id is None or shape.shape_type not in [
                    "rectangle",
                    "polygon",
                    "rotation",
                ]:
                    continue
                rect = shape.bounding_rect()
                cx = rect.x() + (rect.width() / 2.0)
                cy = rect.y() + (rect.height() / 2.0)
                gid2point[shape.group_id] = (cx, cy)

            for linking in linking_pairs:
                pen.setStyle(Qt.SolidLine)
                pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                pen.setColor(QtGui.QColor(*group_color))
                p.setPen(pen)
                key, value = linking
                # Adapt to the 'ungroup_selected_shapes' operation
                if key not in gid2point or value not in gid2point:
                    continue
                kp, vp = gid2point[key], gid2point[value]
                # Draw a link from key point to value point
                p.drawLine(QtCore.QPointF(*kp), QtCore.QPointF(*vp))
                # Draw the triangle arrowhead
                arrow_size = max(
                    1, int(round(10.0 / Shape.scale))
                )  # Size of the arrowhead
                angle = math.atan2(
                    vp[1] - kp[1], vp[0] - kp[0]
                )  # Angle towards the value point
                arrow_points = [
                    QtCore.QPointF(vp[0], vp[1]),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle - math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle - math.pi / 6),
                    ),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle + math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle + math.pi / 6),
                    ),
                ]
                p.drawPolygon(arrow_points)

        # Draw shape masks
        if self.show_masks:
            for shape in self.shapes:
                if not shape.visible:
                    continue
                if shape.shape_type not in [
                    "polygon",
                    "rectangle",
                    "rotation",
                    "circle",
                ]:
                    continue
                if shape.shape_type == "polygon" and len(shape.points) < 3:
                    continue
                if shape.shape_type == "rectangle" and len(shape.points) < 2:
                    continue
                if shape.shape_type == "rotation" and len(shape.points) < 2:
                    continue
                if shape.shape_type == "circle" and len(shape.points) < 2:
                    continue
                if not (
                    (shape.selected or not self._hide_backround)
                    and self.is_visible(shape)
                ):
                    continue

                mask_path = QtGui.QPainterPath()
                if shape.shape_type == "polygon":
                    mask_path.moveTo(shape.points[0])
                    for point in shape.points[1:]:
                        mask_path.lineTo(point)
                    if shape.is_closed() or len(shape.points) >= 3:
                        mask_path.closeSubpath()
                elif shape.shape_type == "rectangle":
                    if len(shape.points) == 2:
                        rectangle = shape.get_rect_from_line(*shape.points)
                        mask_path.addRect(rectangle)
                    elif len(shape.points) == 4:
                        mask_path.moveTo(shape.points[0])
                        for point in shape.points[1:]:
                            mask_path.lineTo(point)
                        mask_path.closeSubpath()
                elif shape.shape_type == "rotation":
                    if len(shape.points) == 2:
                        rectangle = shape.get_rect_from_line(*shape.points)
                        mask_path.addRect(rectangle)
                    elif len(shape.points) == 4:
                        mask_path.moveTo(shape.points[0])
                        for point in shape.points[1:]:
                            mask_path.lineTo(point)
                        mask_path.closeSubpath()
                elif shape.shape_type == "circle":
                    if len(shape.points) == 2:
                        rectangle = shape.get_circle_rect_from_line(
                            shape.points
                        )
                        mask_path.addEllipse(rectangle)

                fill_color = (
                    shape.select_line_color
                    if shape.selected
                    else shape.line_color
                )
                fill_color_alpha = QtGui.QColor(
                    fill_color.red(),
                    fill_color.green(),
                    fill_color.blue(),
                    self.mask_opacity,
                )
                p.setPen(Qt.NoPen)
                p.setBrush(fill_color_alpha)
                p.drawPath(mask_path)

                outline_color = (
                    shape.select_line_color
                    if shape.selected
                    else shape.line_color
                )
                pen = QtGui.QPen(outline_color)
                pen.setWidth(
                    max(1, int(round(shape.line_width / Shape.scale)))
                )
                if shape.difficult:
                    pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawPath(mask_path)

        self._draw_pose_skeletons(p)

        # Draw degrees
        for shape in self.shapes:
            if (
                shape.selected or not self._hide_backround
            ) and self.is_visible(shape):
                shape.fill = (
                    self._fill_drawing
                    and (shape.selected or shape == self.h_hape)
                    and not (self.selected_vertex() and self.moving_shape)
                )
                shape.paint(p)

            if (
                shape.shape_type == "rotation"
                and len(shape.points) == 4
                and self.is_visible(shape)
            ):
                d = shape.point_size / shape.scale
                center = QtCore.QPointF(
                    (shape.points[0].x() + shape.points[2].x()) / 2,
                    (shape.points[0].y() + shape.points[2].y()) / 2,
                )
                if self.show_degrees:
                    degrees = math.degrees(shape.direction)
                    if abs(degrees - 360.0) < 0.1:
                        degrees = 0.0
                    degrees = f"{degrees:.2f}°"
                    p.setFont(
                        QtGui.QFont(
                            "Arial",
                            int(max(6.0, int(round(8.0 / Shape.scale)))),
                        )
                    )
                    pen = QtGui.QPen(
                        QtGui.QColor("#FF9900"), 8, QtCore.Qt.SolidLine
                    )
                    p.setPen(pen)
                    fm = QtGui.QFontMetrics(p.font())
                    rect = fm.boundingRect(degrees)
                    p.fillRect(
                        int(rect.x() + center.x() - d),
                        int(rect.y() + center.y() + d),
                        int(rect.width()),
                        int(rect.height()),
                        QtGui.QColor("#FF9900"),
                    )
                    pen = QtGui.QPen(
                        QtGui.QColor("#FFFFFF"), 7, QtCore.Qt.SolidLine
                    )
                    p.setPen(pen)
                    p.drawText(
                        int(center.x() - d),
                        int(center.y() + d),
                        degrees,
                    )
                else:
                    cp = QtGui.QPainterPath()
                    cp.addRect(
                        int(center.x() - d / 2),
                        int(center.y() - d / 2),
                        int(d),
                        int(d),
                    )
                    p.drawPath(cp)
                    p.fillPath(cp, QtGui.QColor(255, 153, 0, 255))

        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selected_shapes_copy:
            for s in self.selected_shapes_copy:
                s.paint(p)

        if (
            self.fill_drawing()
            and self.create_mode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.add_point(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        # Draw texts
        if self.show_texts:
            text_color = "#FFFFFF"
            background_color = "#007BFF"
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            pen = QtGui.QPen(QtGui.QColor(background_color), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                description = shape.description
                if description:
                    bbox = shape.bounding_rect()
                    fm = QtGui.QFontMetrics(p.font())
                    text_rect = fm.tightBoundingRect(description)

                    padding_x = 4
                    padding_y = 2
                    rect_width = text_rect.width() + 2 * padding_x
                    rect_height = fm.height() + 2 * padding_y

                    bg_x = int(bbox.x())
                    bg_y = int(bbox.y() - rect_height)

                    p.fillRect(
                        bg_x,
                        bg_y,
                        rect_width,
                        rect_height,
                        QtGui.QColor(background_color),
                    )

            pen = QtGui.QPen(QtGui.QColor(text_color), 8, Qt.SolidLine)
            p.setPen(pen)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                description = shape.description
                if description:
                    bbox = shape.bounding_rect()
                    fm = QtGui.QFontMetrics(p.font())

                    padding_x = 4
                    padding_y = 2

                    text_x = int(bbox.x() + padding_x)
                    text_y = int(bbox.y() - padding_y - fm.descent())

                    p.drawText(
                        text_x,
                        text_y,
                        description,
                    )

        # Draw labels
        if self.show_labels:
            self._draw_shape_labels(p)

        # Draw mouse coordinates
        if self.cross_line_show:
            pen = QtGui.QPen(
                QtGui.QColor(self.cross_line_color),
                max(1, int(round(self.cross_line_width / Shape.scale))),
                Qt.DashLine,
            )
            p.setPen(pen)
            p.setOpacity(self.cross_line_opacity)
            p.drawLine(
                QtCore.QPointF(self.prev_move_point.x(), 0),
                QtCore.QPointF(self.prev_move_point.x(), self.pixmap.height()),
            )
            p.drawLine(
                QtCore.QPointF(0, self.prev_move_point.y()),
                QtCore.QPointF(self.pixmap.width(), self.prev_move_point.y()),
            )

        # Draw attributes
        if self.show_attributes:
            font_size = int(max(8.0, int(round(10.0 / Shape.scale))))
            font = QtGui.QFont("Arial", font_size, QtGui.QFont.Bold)
            p.setFont(font)
            attributes_list = []

            for shape in self.shapes:
                if not shape.visible:
                    continue
                if not hasattr(shape, "attributes") or not shape.attributes:
                    continue
                if shape.label in [
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                ]:
                    continue

                attrs_text = []
                for key, value in shape.attributes.items():
                    attrs_text.append(f"{key}: {value}")
                if not attrs_text:
                    continue

                max_attrs_per_line = 1
                attribute_lines = []
                for i in range(0, len(attrs_text), max_attrs_per_line):
                    line_attrs = attrs_text[i : i + max_attrs_per_line]
                    attribute_lines.append(" | ".join(line_attrs))

                fm = QtGui.QFontMetrics(font)
                max_width = 0
                line_heights = []
                for line in attribute_lines:
                    line_rect = fm.tightBoundingRect(line)
                    max_width = max(max_width, line_rect.width())
                    line_heights.append(fm.height())
                total_height = sum(line_heights)

                padding_x = 8
                padding_y = 2
                rect_width = max_width + 2 * padding_x
                rect_height = total_height + 2 * padding_y
                d_react = shape.point_size / shape.scale

                if shape.shape_type in ["rectangle", "polygon", "rotation"]:
                    try:
                        bbox = shape.bounding_rect()
                    except IndexError:
                        continue

                    rect = QtCore.QRect(
                        int(bbox.x()),
                        int(bbox.y() + bbox.height() + 1),
                        rect_width,
                        rect_height,
                    )

                    text_positions = []
                    y_offset = 0
                    for i, line_height in enumerate(line_heights):
                        text_pos = QtCore.QPoint(
                            int(bbox.x() + padding_x),
                            int(
                                bbox.y()
                                + bbox.height()
                                + 1
                                + padding_y
                                + y_offset
                                + fm.ascent()
                            ),
                        )
                        text_positions.append(text_pos)
                        y_offset += line_height

                elif shape.shape_type in [
                    "circle",
                    "line",
                    "linestrip",
                    "point",
                ]:
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]

                    rect = QtCore.QRect(
                        int(point.x() + d_react),
                        int(point.y() + 1),
                        rect_width,
                        rect_height,
                    )

                    text_positions = []
                    y_offset = 0
                    for i, line_height in enumerate(line_heights):
                        text_pos = QtCore.QPoint(
                            int(point.x() + d_react + padding_x),
                            int(
                                point.y()
                                + 1
                                + padding_y
                                + y_offset
                                + fm.ascent()
                            ),
                        )
                        text_positions.append(text_pos)
                        y_offset += line_height
                else:
                    continue

                attributes_list.append(
                    (shape, rect, text_positions, attribute_lines)
                )

            for shape, rect, _, _ in attributes_list:
                if not shape.visible:
                    continue

                background_color = QtGui.QColor(*self.attr_background_color)
                p.fillRect(rect, background_color)

                pen = QtGui.QPen(
                    QtGui.QColor(*self.attr_border_color), 1, Qt.SolidLine
                )
                p.setPen(pen)
                p.drawRect(rect)

            pen = QtGui.QPen(
                QtGui.QColor(*self.attr_text_color), 1, Qt.SolidLine
            )
            p.setPen(pen)
            p.setFont(font)

            for _, _, text_positions, attribute_lines in attributes_list:
                for i, (text_pos, line_text) in enumerate(
                    zip(text_positions, attribute_lines)
                ):
                    p.drawText(text_pos, line_text)

        # Draw compare view split line
        if (
            self.compare_pixmap is not None
            and not self.compare_pixmap.isNull()
        ):
            split_x = int(self.split_position * self.pixmap.width())
            img_h = self.pixmap.height()
            if 0 < split_x < self.pixmap.width():
                p.setRenderHint(QtGui.QPainter.Antialiasing)
                line_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 1.5)
                line_pen.setCapStyle(Qt.RoundCap)
                p.setPen(line_pen)
                p.drawLine(split_x, 0, split_x, img_h)
                handle_radius = 16
                handle_y = img_h // 2
                gradient = QtGui.QRadialGradient(
                    split_x, handle_y, handle_radius
                )
                gradient.setColorAt(0, QtGui.QColor(255, 255, 255, 200))
                gradient.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(gradient)
                p.drawEllipse(
                    split_x - handle_radius,
                    handle_y - handle_radius,
                    handle_radius * 2,
                    handle_radius * 2,
                )
                p.setBrush(QtGui.QColor(255, 255, 255, 200))
                p.drawEllipse(split_x - 8, handle_y - 8, 16, 16)
                arrow_pen = QtGui.QPen(QtGui.QColor(100, 100, 100, 180), 1.5)
                arrow_pen.setCapStyle(Qt.RoundCap)
                arrow_pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(arrow_pen)
                arrow_size = 4
                p.drawLine(
                    split_x - arrow_size, handle_y, split_x - 1, handle_y
                )
                p.drawLine(
                    split_x - arrow_size,
                    handle_y,
                    split_x - arrow_size + 2,
                    handle_y - 2,
                )
                p.drawLine(
                    split_x - arrow_size,
                    handle_y,
                    split_x - arrow_size + 2,
                    handle_y + 2,
                )
                p.drawLine(
                    split_x + 1, handle_y, split_x + arrow_size, handle_y
                )
                p.drawLine(
                    split_x + arrow_size,
                    handle_y,
                    split_x + arrow_size - 2,
                    handle_y - 2,
                )
                p.drawLine(
                    split_x + arrow_size,
                    handle_y,
                    split_x + arrow_size - 2,
                    handle_y + 2,
                )
                p.setRenderHint(QtGui.QPainter.Antialiasing, False)

        p.end()

    def transform_pos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offset_to_center()

    def offset_to_center(self):
        """Calculate offset to the center"""
        if self.pixmap is None:
            return QtCore.QPointF()
        s = self.scale
        area = super().size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        area_width, area_height = area.width(), area.height()
        x = (area_width - w) / (2 * s) if area_width > w else 0
        y = (area_height - h) / (2 * s) if area_height > h else 0
        return QtCore.QPointF(x, y)

    def out_off_pixmap(self, p):
        """Check if a position is out of pixmap"""
        if self.pixmap is None:
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        """Finish drawing for a shape"""
        assert self.current
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            self.current.label = self.auto_labeling_mode.edit_mode
        if self.current.label is None:
            self.current.label = ""
        self.current.close()
        if self.current.shape_type == "rectangle":
            if not self.clip_rectangle_to_pixmap(self.current):
                self.current = None
                self.set_hiding(False)
                self.drawing_polygon.emit(False)
                self.update()
                return

        self.shapes.append(self.current)
        self.store_shapes()
        self.current = None
        self.set_hiding(False)
        self.new_shape.emit()
        self.update()
        if self.is_auto_labeling:
            self.update_auto_labeling_marks()

    def update_auto_labeling_marks(self):
        """Update the auto labeling marks"""
        marks = []
        for shape in self.shapes:
            if shape.label == AutoLabelingMode.ADD:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 1,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 1,
                        }
                    )
            elif shape.label == AutoLabelingMode.REMOVE:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 0,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 0,
                        }
                    )

        self.auto_labeling_marks_updated.emit(marks)

    def close_enough(self, p1, p2):
        """Check if 2 points are close enough (by an threshold epsilon)"""
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersection_point(self, p1, p2):
        """Cycle through each image edge in clockwise fashion,
        and find the one intersecting the current line segment.
        """
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        _, i, (x, y) = min(self.intersecting_edges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        x3, y3 = int(x3), int(y3)
        x4, y4 = int(x4), int(y4)
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            # y3 == y4
            return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(int(x), int(y))

    def intersecting_edges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    # QT Overload
    def sizeHint(self):
        """Get size hint"""
        return self.minimumSizeHint()

    # QT Overload
    def minimumSizeHint(self):
        """Get minimum size hint"""
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super().minimumSizeHint()

    # QT Overload
    def wheelEvent(self, ev: QWheelEvent):
        """Mouse wheel event"""
        mods = ev.modifiers()
        delta = ev.angleDelta()

        if (
            self.editing()
            and not self.sequence_edit_locked
            and self.enable_wheel_rectangle_editing
            and len(self.selected_shapes) == 1
            and self.selected_shapes[0].shape_type == "rectangle"
            and not (QtCore.Qt.ControlModifier & int(mods))
        ):

            try:
                pos = self.transform_pos(ev.posF())
            except AttributeError:
                pos = self.transform_pos(ev.localPos())

            shape = self.selected_shapes[0]
            wheel_up = delta.y() > 0
            old_pose_rect = (
                self._shape_rect_tuple(shape)
                if self._is_pose_rectangle(shape)
                else None
            )

            if shape.contains_point(pos):
                self._scale_rectangle(shape, wheel_up)
            else:
                self._adjust_rectangle_edge(shape, pos, wheel_up)

            self._apply_pose_rectangle_transform(shape, old_pose_rect)
            self.store_shapes()
            self.shape_moved.emit()
            self.update()
            ev.accept()
            return

        # Shift+wheel: adjust compare view split position
        if (
            QtCore.Qt.ShiftModifier == int(mods)
            and self.compare_pixmap is not None
            and not self.compare_pixmap.isNull()
        ):
            step = 0.02 if delta.y() > 0 else -0.02
            self.split_position = max(
                0.0, min(1.0, self.split_position + step)
            )
            self.split_position_changed.emit(self.split_position)
            self.update()
            ev.accept()
            return

        if QtCore.Qt.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoom_request.emit(delta.y(), ev.pos())
        else:
            # scroll
            self.scroll_request.emit(delta.x(), QtCore.Qt.Horizontal, 0)
            self.scroll_request.emit(delta.y(), QtCore.Qt.Vertical, 0)
        ev.accept()

    def _scale_rectangle(self, shape, scale_up):
        """Scale rectangle from center while keeping within image boundaries"""
        if len(shape.points) < 4:
            return

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        x_coords = [p.x() for p in shape.points]
        y_coords = [p.y() for p in shape.points]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        center = QtCore.QPointF(center_x, center_y)

        scale_factor = (
            1.0 + self.rect_scale_step
            if scale_up
            else 1.0 - self.rect_scale_step
        )
        scale_factor = max(0.1, scale_factor)

        new_points = []
        for i in range(len(shape.points)):
            point = shape.points[i]
            offset = point - center
            scaled_offset = offset * scale_factor
            new_point = center + scaled_offset

            if (
                new_point.x() < 0
                or new_point.x() >= img_width
                or new_point.y() < 0
                or new_point.y() >= img_height
            ):
                return

            new_points.append(new_point)

        for i, new_point in enumerate(new_points):
            shape.points[i] = new_point

    def _adjust_rectangle_edge(self, shape, cursor_pos, move_outward):
        """Adjust the rectangle edge closest to cursor position within image boundaries"""
        if len(shape.points) < 4:
            return

        rect = shape.bounding_rect()
        min_x, max_x = rect.left(), rect.right()
        min_y, max_y = rect.top(), rect.bottom()

        distances = {}

        if cursor_pos.x() < min_x:
            distances["left"] = min_x - cursor_pos.x()
        elif cursor_pos.x() > max_x:
            distances["right"] = cursor_pos.x() - max_x
        else:
            distances["left"] = abs(cursor_pos.x() - min_x)
            distances["right"] = abs(cursor_pos.x() - max_x)

        if cursor_pos.y() < min_y:
            distances["top"] = min_y - cursor_pos.y()
        elif cursor_pos.y() > max_y:
            distances["bottom"] = cursor_pos.y() - max_y
        else:
            distances["top"] = abs(cursor_pos.y() - min_y)
            distances["bottom"] = abs(cursor_pos.y() - max_y)

        if (
            cursor_pos.x() < min_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            closest_edge = "left"
        elif (
            cursor_pos.x() > max_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            closest_edge = "right"
        elif (
            cursor_pos.y() < min_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            closest_edge = "top"
        elif (
            cursor_pos.y() > max_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            closest_edge = "bottom"
        else:
            closest_edge = min(distances, key=distances.get)

        step = (
            self.rect_adjust_step if move_outward else -self.rect_adjust_step
        )

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        for i, point in enumerate(shape.points):
            new_point = None

            if closest_edge == "left" and abs(point.x() - min_x) < 1e-6:
                new_x = max(0, point.x() - step)
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "right" and abs(point.x() - max_x) < 1e-6:
                new_x = min(img_width - 1, point.x() + step)
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "top" and abs(point.y() - min_y) < 1e-6:
                new_y = max(0, point.y() - step)
                new_point = QtCore.QPointF(point.x(), new_y)
            elif closest_edge == "bottom" and abs(point.y() - max_y) < 1e-6:
                new_y = min(img_height - 1, point.y() + step)
                new_point = QtCore.QPointF(point.x(), new_y)

            if new_point is not None:
                shape.points[i] = new_point

    def move_by_keyboard(self, offset):
        """Move selected shapes by an offset (using keyboard)"""
        if self.selected_shapes:
            self.bounded_move_shapes(
                self.selected_shapes, self.prev_point + offset
            )
            self.repaint()
            self.moving_shape = True

    def rotate_by_keyboard(self, theta):
        """Rotate selected shapes by an theta (using keyboard)"""
        if self.selected_shapes:
            for i, shape in enumerate(self.selected_shapes):
                if shape._shape_type == "rotation":
                    self.bounded_rotate_shapes(i, shape, theta)
                    self.repaint()
                    self.rotating_shape = True

    # QT Overload
    def keyPressEvent(self, ev):
        """Key press event"""
        if self.sequence_edit_locked:
            return
        self._key_press_event_unlocked(ev)

    def _key_press_event_unlocked(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawing_polygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Backspace and self.current:
                if self.create_mode in ["polygon", "linestrip"]:
                    if len(self.current.points) > 1:
                        self.current.points.pop()
                        self.line[0] = self.current[-1]
                        self.update()
                    elif len(self.current.points) == 1:
                        self.current = None
                        self.drawing_polygon.emit(False)
                        self.update()
            elif key == QtCore.Qt.Key_Return and self.can_close_shape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.move_by_keyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.move_by_keyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.move_by_keyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.move_by_keyboard(QtCore.QPointF(MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Z:
                self.rotate_by_keyboard(self.large_rotation_increment)
            elif key == QtCore.Qt.Key_X:
                self.rotate_by_keyboard(self.small_rotation_increment)
            elif key == QtCore.Qt.Key_C:
                self.rotate_by_keyboard(-self.small_rotation_increment)
            elif key == QtCore.Qt.Key_V:
                self.rotate_by_keyboard(-self.large_rotation_increment)

    # QT Overload
    def keyReleaseEvent(self, ev):
        """Key release event"""
        if self.sequence_edit_locked:
            return
        self._key_release_event_unlocked(ev)

    def _key_release_event_unlocked(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            # NOTE: Temporary fix to avoid ValueError
            # when the selected shape is not in the shapes list
            if (
                (self.moving_shape or self.rotating_shape)
                and self.selected_shapes
                and self.selected_shapes[0] in self.shapes
            ):
                index = self.shapes.index(self.selected_shapes[0])
                if (
                    self.shapes_backups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.store_shapes()
                    if self.moving_shape:
                        self.shape_moved.emit()
                    if self.rotating_shape:
                        self.shape_rotated.emit()

                if self.moving_shape:
                    self.moving_shape = False
                if self.rotating_shape:
                    self.rotating_shape = False

    def set_last_label(self, text, flags, group_id):
        """Set label and flags for last shape"""
        assert text
        if self.is_auto_labeling:
            self.shapes[-1].label = self.auto_labeling_mode.edit_mode
        else:
            self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapes[-1].group_id = group_id
        self.shapes_backups.pop()
        self.store_shapes()
        return self.shapes[-1]

    def undo_last_line(self):
        """Undo last line"""
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.set_open()
        if self.create_mode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.create_mode in ["rectangle", "line", "circle", "rotation"]:
            self.current.points = self.current.points[0:1]
        elif self.create_mode == "point":
            self.current = None
        self.drawing_polygon.emit(True)

    def undo_last_point(self):
        """Undo last point"""
        if not self.current or self.current.is_closed():
            return
        self.current.pop_point()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawing_polygon.emit(False)
        self.update()

    def load_pixmap(self, pixmap, clear_shapes=True):
        """Load pixmap"""
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
        self.update()

    def load_shapes(self, shapes, replace=True):
        """Load shapes"""
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.store_shapes()
        self.current = None
        self.h_hape = None
        self.h_vertex = None
        self.h_edge = None
        self.update()

    def set_shape_visible(self, shape, value):
        """Set visibility for a shape"""
        self.visible[shape] = value
        self.update()

    def current_cursor(self):
        """Current cursor"""
        cursor = QtWidgets.QApplication.overrideCursor()
        cursor = cursor.shape() if cursor else None

        return cursor

    def override_cursor(self, cursor):
        """Override cursor"""
        current_cursor = self.current_cursor()
        if current_cursor != cursor:
            self._cursor = cursor
            if current_cursor is None:
                QtWidgets.QApplication.setOverrideCursor(cursor)
            else:
                QtWidgets.QApplication.changeOverrideCursor(cursor)

    def restore_cursor(self):
        """Restore override cursor"""
        QtWidgets.QApplication.restoreOverrideCursor()

    def reset_state(self):
        """Clear shapes and pixmap"""
        self.restore_cursor()
        self.pixmap = None
        self.shapes_backups = []
        self.is_move_editing = False
        self.compare_pixmap = None
        self.update()

    def set_cross_line(self, show, width, color, opacity):
        """Set cross line options"""
        self.cross_line_show = show
        self.cross_line_width = width
        self.cross_line_color = color
        self.cross_line_opacity = opacity
        self.update()

    def gen_new_group_id(self):
        """Generate new shape's group_id based on current shapes"""
        max_group_id = 0
        for shape in self.shapes:
            if shape.group_id is not None:
                max_group_id = max(max_group_id, shape.group_id)
        return max_group_id + 1

    def merge_group_ids(self, group_ids, new_group_id):
        """Merge multiple shapes' group_id into a new one"""
        for shape in self.shapes:
            if shape.group_id in group_ids:
                shape.group_id = new_group_id

    def group_selected_shapes(self):
        """Group selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        has_non_group_shape = False
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)
            else:
                has_non_group_shape = True

        # If there is at least 1 shape having a group id,
        # use that id as the new group id. Otherwise, generate a new group_id
        new_group_id = None
        if len(group_ids) > 0:
            new_group_id = min(group_ids)
        else:
            new_group_id = self.gen_new_group_id()

        # Merge group ids
        if len(group_ids) > 1:
            self.merge_group_ids(
                group_ids=group_ids, new_group_id=new_group_id
            )
        # Assign new_group_id to non-group shapes
        if has_non_group_shape:
            for shape in self.selected_shapes:
                if shape.group_id is None:
                    shape.group_id = new_group_id

        self.update()

    def ungroup_selected_shapes(self):
        """Ungroup selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)

        for group_id in group_ids:
            for shape in self.shapes:
                if shape.group_id == group_id:
                    shape.group_id = None

        self.update()
