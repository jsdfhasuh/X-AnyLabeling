"""Explicit empty-annotation confirmation for an embedded manual session."""

import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from . import new_icon
from .auto_labeling_commit import resolve_existing_label


STATUS_ROLE = QtCore.Qt.UserRole + 41
STATUS_TEXT = {
    "unlabeled": "未标注",
    "annotated": "已标注",
    "confirmed_empty": "已确认无目标",
    "empty": "空标注",
    "pending": "待审核",
    "needs_fix": "需修正",
    "stale": "标注已变化",
    "conflict": "标注冲突",
    "recovery": "待恢复",
}
TITLE = "确认无目标并下一张"


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


class AnnotationStatusDelegate(QtWidgets.QStyledItemDelegate):
    """Paint a status without changing the path used by navigation."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        status = index.data(STATUS_ROLE)
        if status:
            option.text = (
                "[" + STATUS_TEXT.get(status, status) + "]  " + option.text
            )


class EmptyAnnotationController(QtCore.QObject):
    def __init__(self, widget, shortcuts):
        super().__init__(widget)
        self.widget = widget
        self.busy = False
        self._cooldown_until = 0.0
        self._refresh_scheduled = False
        self._record_key = None
        self._record_value = None
        self.action = QtWidgets.QAction(new_icon("done"), TITLE, widget)
        self.action.setObjectName("confirm_empty_next")
        self.action.setAutoRepeat(False)
        self.action.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.action.setVisible(False)
        self.action.setEnabled(False)
        self.action.triggered.connect(self.confirm)
        self.requested_shortcut = shortcuts.get("confirm_empty_next", "Alt+N")
        self.shortcut_conflict = False
        self.action.installEventFilter(self)

    def start(self):
        self.widget.file_list_widget.setItemDelegate(
            AnnotationStatusDelegate(self.widget.file_list_widget)
        )
        app = QtWidgets.QApplication.instance()
        app.installEventFilter(self)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(150)
        # This timer checks only in-memory UI state, never files or stores.
        self.timer.timeout.connect(self.refresh_action)
        self.timer.start()
        self.configure_shortcut()
        self.refresh_action()

    def configure_shortcut(self):
        requested = self.requested_shortcut
        values = (
            requested if isinstance(requested, (list, tuple)) else [requested]
        )
        sequences = [QtGui.QKeySequence(value) for value in values if value]
        occupied = []
        for action in self.widget.window().findChildren(QtWidgets.QAction):
            if action is not self.action:
                occupied.extend(action.shortcuts())
        occupied.extend(
            shortcut.key()
            for shortcut in self.widget.window().findChildren(
                QtWidgets.QShortcut
            )
        )
        self.shortcut_conflict = any(
            sequence.matches(other) != QtGui.QKeySequence.NoMatch
            or other.matches(sequence) != QtGui.QKeySequence.NoMatch
            for sequence in sequences
            for other in occupied
            if not other.isEmpty()
        )
        self.action.setShortcuts([] if self.shortcut_conflict else sequences)
        self.refresh_action()

    def _record(self):
        context = getattr(self.widget, "auto_labeling_host_context", None)
        path = getattr(self.widget, "filename", None)
        records = getattr(context, "image_records_by_path", {})
        key = (id(context), path)
        if key != self._record_key:
            self._record_key = key
            self._record_value = (
                records.get(_canonical(path)) if path else None
            )
        return self._record_value

    def unavailable_reason(self):
        widget = self.widget
        context = getattr(widget, "auto_labeling_host_context", None)
        if not getattr(context, "confirm_empty_annotations_enabled", False):
            return "仅支持目标检测的人工标注会话"
        if context.session_purpose != "labeling":
            return "请使用审核界面的确认操作"
        if not context.images_ready or self._record() is None:
            return "图片尚未加载完成"
        lease = context.annotation_session_lease
        if getattr(lease, "released", True):
            return "标注会话权限已失效"
        if self.busy or time.monotonic() < self._cooldown_until:
            return "正在完成当前操作"
        image = getattr(widget, "image", None)
        canvas = widget.canvas
        if image is None or image.isNull() or not canvas.isEnabled():
            return "图片尚未加载完成"
        auto_widget = getattr(widget, "auto_labeling_widget", None)
        manager = getattr(auto_widget, "model_manager", None)
        inference_lease = getattr(manager, "inference_lease", None)
        if getattr(inference_lease, "is_active", False):
            return "自动标注正在运行"
        if (
            not widget.isEnabled()
            or getattr(canvas, "is_loading", False)
            or getattr(canvas, "sequence_edit_locked", False)
            or getattr(widget, "fast_auto_labeling_edit_locked", False)
            or getattr(widget, "auto_labeling_audit_active", False)
            or getattr(widget, "_sequence_presentation_session", None)
            or getattr(auto_widget, "_fast_run_session", None)
            or (
                getattr(
                    getattr(widget, "actions", None), "run_all_images", None
                )
                and widget.actions.run_all_images.isChecked()
            )
        ):
            return "当前处于只读、审核或自动标注状态"
        if getattr(widget, "auto_labeling_commit_blocked", False):
            return "请先恢复未完成的标注提交"
        if getattr(canvas, "current", None) is not None:
            return "请先完成或取消当前绘制"
        if canvas.shapes or len(widget.label_list):
            return "当前图片已有目标或临时标注"
        return ""

    def refresh_action(self):
        context = getattr(self.widget, "auto_labeling_host_context", None)
        visible = bool(
            getattr(context, "confirm_empty_annotations_enabled", False)
            and getattr(context, "session_purpose", None) == "labeling"
        )
        reason = self.unavailable_reason()
        self.action.setVisible(visible)
        self.action.setEnabled(visible and not reason)
        tooltip = TITLE
        shortcuts = self.action.shortcuts()
        if shortcuts:
            tooltip += "（" + shortcuts[0].toString() + "）"
        if self.shortcut_conflict:
            tooltip += "\n快捷键冲突：保留已有绑定，请使用按钮或菜单"
        if reason:
            tooltip += "\n" + reason
        self.action.setToolTip(tooltip)
        self.action.setStatusTip(tooltip)

    @staticmethod
    def _text_input_active():
        focus = QtWidgets.QApplication.focusWidget()
        while focus is not None:
            if isinstance(
                focus,
                (
                    QtWidgets.QLineEdit,
                    QtWidgets.QTextEdit,
                    QtWidgets.QPlainTextEdit,
                    QtWidgets.QAbstractSpinBox,
                ),
            ):
                return True
            focus = focus.parentWidget()
        return False

    def eventFilter(self, watched, event):
        kind = event.type()
        if watched is self.action and kind == QtCore.QEvent.Shortcut:
            if (
                self._text_input_active()
                or QtWidgets.QApplication.activeModalWidget() is not None
                or QtWidgets.QApplication.activePopupWidget() is not None
            ):
                return True
        if (
            kind == QtCore.QEvent.ShortcutOverride
            and isinstance(watched, QtWidgets.QWidget)
            and watched.window() is self.widget.window()
            and self.widget.isVisible()
            and self.action.isVisible()
            and self.action.isEnabled()
            and not self.unavailable_reason()
            and self._text_input_active()
        ):
            sequence = QtGui.QKeySequence(
                int(event.modifiers()) | event.key()
            )
            if sequence in self.action.shortcuts():
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def schedule_status_refresh(self):
        if not self._refresh_scheduled:
            self._refresh_scheduled = True
            QtCore.QTimer.singleShot(0, self.refresh_current_status)

    def set_row_status(self, path, status):
        items = self.widget.file_list_widget.findItems(
            path, QtCore.Qt.MatchExactly
        )
        for item in items:
            item.setData(STATUS_ROLE, status)
            item.setToolTip(path + "\n" + STATUS_TEXT.get(status, status))

    def refresh_current_status(self):
        self._refresh_scheduled = False
        widget = self.widget
        context = getattr(widget, "auto_labeling_host_context", None)
        if not getattr(context, "confirm_empty_annotations_enabled", False):
            return
        reader = getattr(context, "read_annotation_display_status", None)
        record = self._record()
        if record is None or not callable(reader):
            return
        try:
            status = reader(record.image_id)
        except Exception as exc:
            from ..logger import logger

            logger.warning("Cannot read annotation status: %s", exc)
            status = "recovery"
        if getattr(widget, "auto_labeling_commit_blocked", False):
            status = "recovery"
        self.set_row_status(widget.filename, status)
        self.refresh_action()

    def confirm(self, _checked=False):
        reason = self.unavailable_reason()
        if reason:
            self.widget.status(reason)
            return False
        widget = self.widget
        original = widget.filename
        context = widget.auto_labeling_host_context
        record = self._record()
        self.busy = True
        self.refresh_action()
        try:
            # Normal save owns CAS, intent, atomic write and checkpoint.
            status = context.read_annotation_display_status(record.image_id)
            if widget.dirty or status != "confirmed_empty":
                if not widget.save_file():
                    return False
            if (
                widget.filename != original
                or widget.auto_labeling_host_context is not context
                or getattr(widget, "auto_labeling_commit_blocked", False)
            ):
                return False
            current = resolve_existing_label(
                record.canonical_session_label_path
            )
            if current.presence != "VALID_EMPTY" or (
                context.read_annotation_display_status(record.image_id)
                != "confirmed_empty"
            ):
                widget.status("空标注提交尚未确认，请检查标注状态")
                return False
            self.refresh_current_status()
            index = widget.fn_to_index.get(str(original))
            if index is None or index + 1 >= widget.file_list_widget.count():
                widget.status("已确认无目标，当前已是最后一张")
                return True
            try:
                widget.open_next_image()
            except Exception:
                from ..logger import logger

                logger.exception(
                    "Navigation failed after empty annotation commit"
                )
                widget.status("当前图片已确认无目标；下一张加载失败，请重新选择图片")
                return True
            if (
                widget.filename == original
                or widget.image.isNull()
                or not widget.canvas.isEnabled()
            ):
                widget.status("当前图片已确认无目标；下一张加载失败，请重新选择图片")
            else:
                widget.status("上一张已确认无目标")
            return True
        except Exception as exc:
            from ..logger import logger

            logger.exception("Empty annotation confirmation failed")
            widget.status("确认无目标失败：" + str(exc))
            return False
        finally:
            self.busy = False
            self._cooldown_until = time.monotonic() + 0.5
            self.schedule_status_refresh()
            self.refresh_action()
