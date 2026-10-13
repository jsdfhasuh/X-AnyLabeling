"""Explicit empty-annotation confirmation for an embedded manual session."""

import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from . import new_icon
from .auto_labeling_commit import resolve_existing_label
from .interaction_performance import start_gui_probe


STATUS_ROLE = QtCore.Qt.UserRole + 41
STATUS_TEXT = {
    "checking": "正在核对",
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


class _StatusResult(QtCore.QObject):
    ready = QtCore.pyqtSignal(object, object, object)


class _StatusRead(QtCore.QRunnable):
    def __init__(self, context, image_id, token, cached):
        super().__init__()
        self.context, self.image_id = context, image_id
        self.token, self.cached = token, cached
        self.result = _StatusResult()

    def run(self):
        value = error = None
        try:
            signature = getattr(
                self.context, "annotation_display_signature", None
            )
            before = signature(self.image_id) if callable(signature) else None
            if (
                before is not None
                and self.cached is not None
                and self.cached.get("signature") == before
            ):
                value = self.cached
            else:
                reader = getattr(
                    self.context, "read_annotation_display_snapshot", None
                )
                if callable(reader):
                    value = reader(self.image_id)
                else:
                    value = dict(
                        status=self.context.read_annotation_display_status(
                            self.image_id
                        )
                    )
                after = (
                    signature(self.image_id) if callable(signature) else None
                )
                if before != after:
                    raise OSError("Annotation changed during display read")
                value = dict(value, signature=after)
        except Exception as exc:
            error = exc
        self.result.ready.emit(self.token, value, error)


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
        self._version = 0
        self._cache = None
        self._cache_key = None
        self._status_task = None
        self._pending_status = None
        self._validate_scheduled = False
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
        self.verify_timer = QtCore.QTimer(self)
        self.verify_timer.setInterval(2000)
        self.verify_timer.timeout.connect(self._verify_visible_status)
        self.verify_timer.start()
        self._diagnostic_probe = start_gui_probe(self)
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
        if (kind == QtCore.QEvent.WindowActivate
                and watched is self.widget.window()):
            self.invalidate_status()
            self.schedule_status_refresh()
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

    def invalidate_status(self):
        self._version += 1
        self._cache = self._cache_key = None
        self._pending_status = None

    def _verify_visible_status(self):
        if self.widget.isVisible():
            self.schedule_status_refresh(validate=True)

    def schedule_status_refresh(self, validate=False):
        self._validate_scheduled = self._validate_scheduled or validate
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
        validate, self._validate_scheduled = self._validate_scheduled, False
        widget = self.widget
        context = getattr(widget, "auto_labeling_host_context", None)
        if not getattr(context, "confirm_empty_annotations_enabled", False):
            return
        reader = getattr(context, "read_annotation_display_status", None)
        record = self._record()
        if record is None or not callable(reader):
            return
        if getattr(widget, "auto_labeling_commit_blocked", False):
            self.set_row_status(widget.filename, "recovery")
            self.refresh_action()
            return
        key = self._status_key(context, record)
        if self._cache_key == key and self._cache is not None and not validate:
            self.set_row_status(widget.filename, self._cache["status"])
            self.refresh_action()
            return
        request = (
            context, record.image_id, key,
            self._cache if self._cache_key == key else None,
        )
        if self._cache_key != key:
            self.set_row_status(widget.filename, "checking")
        if self._status_task is not None:
            if self._status_task.token != key:
                self._pending_status = request
            self.refresh_action()
            return
        self._start_status_read(request)
        self.refresh_action()

    def _status_key(self, context, record):
        return (id(context), getattr(context, "project_id", None),
                getattr(context, "active_session_id", None), record.image_id,
                self.widget.filename, self._version,
                getattr(self.widget, "_loaded_label_document_digest", None))

    def _start_status_read(self, request):
        self._status_task = _StatusRead(*request)
        self._status_task.result.ready.connect(
            self._status_ready, QtCore.Qt.QueuedConnection
        )
        QtCore.QThreadPool.globalInstance().start(self._status_task)

    @QtCore.pyqtSlot(object, object, object)
    def _status_ready(self, token, value, error):
        self._status_task = None
        context = getattr(self.widget, "auto_labeling_host_context", None)
        record = self._record()
        if record is not None and token == self._status_key(context, record):
            if error is not None:
                from ..logger import logger
                logger.warning("Cannot read annotation status: %s", error)
                self._cache = self._cache_key = None
                status = "recovery"
            else:
                if (value.get("document_digest") is not None
                        and token[-1] is not None
                        and value["document_digest"] != token[-1]):
                    value = dict(value, status="stale")
                self._cache_key, self._cache = token, value
                status = value["status"]
            if getattr(self.widget, "auto_labeling_commit_blocked", False):
                status = "recovery"
            self.set_row_status(self.widget.filename, status)
            self.refresh_action()
        pending, self._pending_status = self._pending_status, None
        if (
            pending is not None
            and record is not None
            and pending[2] == self._status_key(context, record)
        ):
            self._start_status_read(pending)

    def accept_checkpoint(self, context, image_path, snapshot):
        if context is not getattr(
            self.widget, "auto_labeling_host_context", None
        ):
            return
        self.invalidate_status()
        self.set_row_status(image_path, snapshot["status"])
        record = self._record()
        if (
            record is not None
            and record.image_id == snapshot["image_id"]
            and image_path == self.widget.filename
        ):
            self._cache_key = self._status_key(context, record)
            self._cache = dict(snapshot, signature=None)
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
            # The action just performed a fresh authoritative read above.
            self.invalidate_status()
            self.set_row_status(original, "confirmed_empty")
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
