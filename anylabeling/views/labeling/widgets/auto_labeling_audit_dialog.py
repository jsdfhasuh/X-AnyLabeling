"""Non-modal staged audit UI using stable image identities."""

from PyQt5 import QtCore, QtWidgets

from anylabeling.views.labeling.utils.auto_labeling_audit import (
    StagedAuditClientProtocolV1,
    StagedAuditError,
)
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_boolean_text_v1,
    auto_labeling_status_text_v1,
    auto_labeling_text_v1,
)


class AuditInteractionGuard:
    """Keep annotation editing available while locking identity-changing UI."""

    _ACTION_NAMES = (
        "open_next_image",
        "open_prev_image",
        "open_next_unchecked_image",
        "open_prev_unchecked_image",
        "save_as",
        "change_output_dir",
        "delete_image_file",
        "run_all_images",
    )
    _TRANSACTION_ACTION_NAMES = (
        "save",
        "delete",
        "undo",
        "undo_last_point",
        "remove_point",
        "duplicate",
        "copy",
        "paste",
        "edit_mode",
        "create_mode",
        "create_rectangle_mode",
        "create_pose_mode",
        "create_rotation_mode",
        "create_circle_mode",
        "create_line_mode",
        "create_point_mode",
        "create_line_strip_mode",
        "group_selected_shapes",
        "ungroup_selected_shapes",
    )
    _TRANSACTION_CONTROL_NAMES = (
        "canvas",
        "label_list",
        "unique_label_list",
        "label_filter_combobox",
        "shape_dock",
    )

    def __init__(self, widget):
        self.widget = widget
        self._states = []
        self._transaction_states = []
        self._active = False
        self._transaction_busy = False

    def activate(self):
        if self._active:
            return
        actions = getattr(self.widget, "actions", None)
        for name in self._ACTION_NAMES:
            action = getattr(actions, name, None)
            if action is not None and callable(
                getattr(action, "setEnabled", None)
            ):
                self._states.append((action, action.isEnabled()))
                action.setEnabled(False)
        for control_name in ("file_list_widget", "auto_labeling_widget"):
            control = getattr(self.widget, control_name, None)
            if control is not None and callable(
                getattr(control, "setEnabled", None)
            ):
                self._states.append((control, control.isEnabled()))
                control.setEnabled(False)
        self.widget.auto_labeling_audit_active = True
        self._active = True

    def release(self):
        if not self._active:
            return
        self.set_transaction_busy(False)
        for target, enabled in reversed(self._states):
            target.setEnabled(enabled)
        self._states = []
        self.widget.auto_labeling_audit_active = False
        self._active = False

    def set_transaction_busy(self, busy):
        busy = bool(busy)
        if busy == self._transaction_busy:
            return
        if busy:
            actions = getattr(self.widget, "actions", None)
            for name in self._TRANSACTION_ACTION_NAMES:
                action = getattr(actions, name, None)
                if action is not None and callable(
                    getattr(action, "setEnabled", None)
                ):
                    self._transaction_states.append(
                        (action, action.isEnabled())
                    )
                    action.setEnabled(False)
            for name in self._TRANSACTION_CONTROL_NAMES:
                control = getattr(self.widget, name, None)
                if control is not None and callable(
                    getattr(control, "setEnabled", None)
                ):
                    self._transaction_states.append(
                        (control, control.isEnabled())
                    )
                    control.setEnabled(False)
            self._transaction_busy = True
            return
        for target, enabled in reversed(self._transaction_states):
            target.setEnabled(enabled)
        self._transaction_states = []
        self._transaction_busy = False


class _StagedAuditOperationWorker(QtCore.QObject):
    completed = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, object)

    def __init__(self, client):
        super().__init__()
        self.client = client

    @QtCore.pyqtSlot(int, str, object)
    def execute(self, operation_id, kind, payload):
        try:
            result = self._execute(kind, payload)
        except Exception as exc:  # noqa: B902
            self.failed.emit(operation_id, exc)
            return
        self.completed.emit(operation_id, result)

    def _execute(self, kind, payload):
        if kind == "begin":
            items = sorted(
                self.client.list_review_items(),
                key=lambda value: value["sequence"],
            )
            if not items:
                return {"items": [], "item": None, "summary": None}
            item = self._read_item(items[0]["image_id"])
            return {
                "items": items,
                "item": item,
                "summary": self.client.summary(),
            }
        if kind == "load":
            return {
                "item": self._read_item(payload["image_id"]),
                "summary": self.client.summary(),
            }
        if kind in {"approve", "needs_fix"}:
            image_id = payload["image_id"]
            refreshed = self.client.integrity_refresh(image_id)
            self._raise_integrity_error(refreshed)
            revision = refreshed["item_revision"]
            if kind == "approve":
                item = self.client.approve(image_id, revision)
            else:
                item = self.client.needs_fix(image_id, revision)
            return {"item": item, "summary": self.client.summary()}
        raise ValueError(f"unknown audit operation: {kind}")

    def _read_item(self, image_id):
        item = self.client.read_review_item(image_id)
        self._raise_integrity_error(item)
        return item

    @staticmethod
    def _raise_integrity_error(item):
        error = item.get("integrity_error")
        if error:
            raise StagedAuditError(error)


def _build_audit_controls(view, *, compact=False):
    layout = QtWidgets.QVBoxLayout(view)
    margins = (8, 8, 8, 8) if compact else (16, 14, 16, 14)
    layout.setContentsMargins(*margins)
    layout.setSpacing(8 if compact else 10)

    view.identity_label = QtWidgets.QLabel()
    view.identity_label.setTextInteractionFlags(
        QtCore.Qt.TextSelectableByMouse
    )
    view.identity_label.setWordWrap(True)
    view.status_label = QtWidgets.QLabel()
    view.status_label.setWordWrap(True)
    view.detail_label = QtWidgets.QLabel()
    view.detail_label.setWordWrap(True)
    view.counts_label = QtWidgets.QLabel()
    view.counts_label.setWordWrap(True)
    layout.addWidget(view.identity_label)
    layout.addWidget(view.status_label)
    layout.addWidget(view.detail_label)
    layout.addWidget(view.counts_label)

    view.activity_bar = QtWidgets.QProgressBar()
    view.activity_bar.setRange(0, 0)
    view.activity_bar.setTextVisible(False)
    view.activity_bar.setFixedHeight(4)
    view.activity_bar.hide()
    layout.addWidget(view.activity_bar)

    navigation = QtWidgets.QHBoxLayout()
    view.previous_button = QtWidgets.QPushButton(
        auto_labeling_text_v1("previous")
    )
    view.next_button = QtWidgets.QPushButton(
        auto_labeling_text_v1("next_pending")
    )
    navigation.addWidget(view.previous_button)
    navigation.addWidget(view.next_button)
    layout.addLayout(navigation)

    decisions = QtWidgets.QGridLayout() if compact else QtWidgets.QHBoxLayout()
    view.needs_fix_button = QtWidgets.QPushButton(
        auto_labeling_text_v1("mark_needs_fix")
    )
    view.save_approve_button = QtWidgets.QPushButton(
        auto_labeling_text_v1("save_approve_next")
    )
    view.approve_button = QtWidgets.QPushButton(
        auto_labeling_text_v1("approve_next")
    )
    view.finish_button = QtWidgets.QPushButton(
        auto_labeling_text_v1("finish_audit")
    )
    if compact:
        decisions.addWidget(view.needs_fix_button, 0, 0)
        decisions.addWidget(view.save_approve_button, 0, 1)
        decisions.addWidget(view.approve_button, 1, 0)
        decisions.addWidget(view.finish_button, 1, 1)
    else:
        decisions.addWidget(view.needs_fix_button)
        decisions.addStretch(1)
        decisions.addWidget(view.save_approve_button)
        decisions.addWidget(view.approve_button)
        decisions.addWidget(view.finish_button)
    layout.addLayout(decisions)
    layout.addStretch(1)

    view.previous_button.clicked.connect(view.previous_requested)
    view.next_button.clicked.connect(view.next_requested)
    view.needs_fix_button.clicked.connect(view.needs_fix_requested)
    view.save_approve_button.clicked.connect(view.save_approve_requested)
    view.approve_button.clicked.connect(view.approve_requested)
    view.finish_button.clicked.connect(view.finish_requested)


class AutoLabelingAuditDialog(QtWidgets.QDialog):
    approve_requested = QtCore.pyqtSignal()
    save_approve_requested = QtCore.pyqtSignal()
    needs_fix_requested = QtCore.pyqtSignal()
    previous_requested = QtCore.pyqtSignal()
    next_requested = QtCore.pyqtSignal()
    finish_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle(auto_labeling_text_v1("audit_title"))
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setMinimumWidth(560)

        _build_audit_controls(self)

    def set_busy(self, busy, finishing=False):
        busy = bool(busy)
        for button in (
            self.previous_button,
            self.next_button,
            self.needs_fix_button,
            self.save_approve_button,
            self.approve_button,
        ):
            button.setEnabled(not busy)
        self.finish_button.setEnabled(not finishing)
        self.activity_bar.setVisible(busy)

    def set_item(self, item, dirty):
        self.identity_label.setText(
            auto_labeling_text_v1("audit_identity", **item)
        )
        self.status_label.setText(
            auto_labeling_text_v1(
                "audit_status",
                status=auto_labeling_status_text_v1(
                    item["review_status"],
                    "review",
                ),
                dirty=auto_labeling_boolean_text_v1(dirty),
            )
        )
        self.detail_label.setText(
            auto_labeling_text_v1(
                "audit_detail",
                staged=auto_labeling_status_text_v1(
                    item["staged_commit_status"],
                    "commit",
                ),
                source=auto_labeling_status_text_v1(
                    item["source_commit_status"],
                    "commit",
                ),
                target_count=item["target_count"],
                zero_target=auto_labeling_boolean_text_v1(item["zero_target"]),
            )
        )

    def set_summary(self, summary):
        self.counts_label.setText(
            auto_labeling_text_v1("audit_summary", **summary)
        )

    def allow_close(self):
        self._allow_close = True

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
            return
        self.finish_requested.emit()
        event.ignore()


class AutoLabelingAuditPanel(QtWidgets.QWidget):
    """Embedded audit controls for the labeling sidebar."""

    approve_requested = QtCore.pyqtSignal()
    save_approve_requested = QtCore.pyqtSignal()
    needs_fix_requested = QtCore.pyqtSignal()
    previous_requested = QtCore.pyqtSignal()
    next_requested = QtCore.pyqtSignal()
    finish_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._allow_close = False
        self.setMinimumWidth(280)
        _build_audit_controls(self, compact=True)

    set_busy = AutoLabelingAuditDialog.set_busy
    set_item = AutoLabelingAuditDialog.set_item
    set_summary = AutoLabelingAuditDialog.set_summary
    allow_close = AutoLabelingAuditDialog.allow_close

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
            return
        self.finish_requested.emit()
        event.ignore()


class StagedAuditUiSession(QtCore.QObject):
    """Sequence stable audit actions around one LabelingWidget."""

    safe_to_close = QtCore.pyqtSignal(int)
    _execute_requested = QtCore.pyqtSignal(int, str, object)

    dialog_class = AutoLabelingAuditDialog

    def __init__(self, widget, client):
        super().__init__(widget)
        if not isinstance(client, StagedAuditClientProtocolV1):
            raise TypeError(
                "client does not implement StagedAuditClientProtocolV1"
            )
        self.widget = widget
        self.client = client
        panel_factory = getattr(
            widget, "create_auto_labeling_audit_panel", None
        )
        if callable(panel_factory):
            self.dialog = panel_factory()
        else:
            self.dialog = self.dialog_class(widget)
        self.guard = AuditInteractionGuard(widget)
        self.items = []
        self.current_image_id = None
        self.worker_thread = None
        self._worker = None
        self._summary = None
        self._active_operation = None
        self._next_operation_id = 0
        self._finish_requested = False
        self._shutdown_generation = None
        self._next_shutdown_generation = 0
        self._closed = False
        self._connect()

    def _connect(self):
        self.dialog.approve_requested.connect(self.approve_and_next)
        self.dialog.save_approve_requested.connect(self.save_approve_and_next)
        self.dialog.needs_fix_requested.connect(self.mark_needs_fix)
        self.dialog.previous_requested.connect(self.previous)
        self.dialog.next_requested.connect(self.next_pending)
        self.dialog.finish_requested.connect(self.finish)

    def begin(self):
        if not self._start_worker():
            return False
        self.guard.activate()
        panel_presenter = getattr(
            self.widget,
            "show_auto_labeling_audit_panel",
            None,
        )
        if callable(panel_presenter):
            panel_presenter(self.dialog)
        else:
            self.dialog.show()
        if self._submit("begin", {}):
            return True
        self._finish_requested = True
        self._begin_shutdown()
        return False

    def _start_worker(self):
        if self.worker_thread is not None or self._closed:
            return False
        self.worker_thread = QtCore.QThread(self)
        self._worker = _StagedAuditOperationWorker(self.client)
        self._worker.moveToThread(self.worker_thread)
        self._execute_requested.connect(
            self._worker.execute,
            QtCore.Qt.QueuedConnection,
        )
        self._worker.completed.connect(
            self._on_operation_completed,
            QtCore.Qt.QueuedConnection,
        )
        self._worker.failed.connect(
            self._on_operation_failed,
            QtCore.Qt.QueuedConnection,
        )
        self.worker_thread.finished.connect(self._worker.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)
        self.worker_thread.start()
        return True

    def _submit(self, kind, payload):
        thread = self.worker_thread
        if (
            self._closed
            or self._finish_requested
            or self._active_operation is not None
            or thread is None
            or not thread.isRunning()
        ):
            return False
        self._next_operation_id += 1
        operation_id = self._next_operation_id
        self._active_operation = {
            "operation_id": operation_id,
            "kind": kind,
        }
        self._set_operation_busy(True)
        self._execute_requested.emit(operation_id, kind, dict(payload))
        return True

    def _set_operation_busy(self, busy):
        self.guard.set_transaction_busy(busy)
        self.dialog.set_busy(busy, finishing=self._finish_requested)

    @QtCore.pyqtSlot(int, object)
    def _on_operation_completed(self, operation_id, result):
        operation = self._take_operation(operation_id)
        if operation is None:
            return
        self._set_operation_busy(False)
        try:
            self._apply_operation_result(operation["kind"], result)
        except Exception as exc:  # noqa: B902
            self._warn(exc)
        if self._finish_requested and self._active_operation is None:
            self._begin_shutdown()

    @QtCore.pyqtSlot(int, object)
    def _on_operation_failed(self, operation_id, exc):
        operation = self._take_operation(operation_id)
        if operation is None:
            return
        self._set_operation_busy(False)
        self._warn(exc)
        if operation["kind"] == "begin":
            self._finish_requested = True
        if self._finish_requested:
            self._begin_shutdown()

    def _take_operation(self, operation_id):
        operation = self._active_operation
        if operation is None or operation["operation_id"] != operation_id:
            return None
        self._active_operation = None
        return operation

    def _apply_operation_result(self, kind, result):
        if kind == "begin":
            self.items = list(result["items"])
            if not self.items:
                self._finish_requested = True
                return
            self._summary = result["summary"]
            self._publish_pending_count(result["item"].get("image_id"))
            self._present_item(result["item"])
            return
        item = self._remember_item(result["item"])
        self._summary = result["summary"]
        self._publish_pending_count(item.get("image_id"))
        if kind == "load":
            self._present_item(item)
        elif kind == "approve":
            self._start_next_pending()
        elif kind == "needs_fix":
            self._refresh_display(item)
        else:
            raise ValueError(f"unknown audit result: {kind}")

    def _remember_item(self, item):
        image_id = item["image_id"]
        for index, existing in enumerate(self.items):
            if existing["image_id"] == image_id:
                merged = dict(existing)
                merged.update(item)
                self.items[index] = merged
                return merged
        remembered = dict(item)
        self.items.append(remembered)
        self.items.sort(key=lambda value: value["sequence"])
        return remembered

    def _cached_item(self, image_id):
        for item in self.items:
            if item["image_id"] == image_id:
                return item
        return None

    def _current_item(self):
        if self.current_image_id is None:
            raise StagedAuditError("audit_current_image_missing")
        item = self._cached_item(self.current_image_id)
        if item is None:
            raise StagedAuditError("audit_current_item_missing")
        return item

    def _ensure_clean_and_coordinated(self):
        if bool(getattr(self.widget, "dirty", False)):
            raise StagedAuditError("audit_dirty_requires_save")
        if bool(getattr(self.widget, "auto_labeling_commit_blocked", False)):
            raise StagedAuditError("audit_integrity_refresh_required")

    def _resolve_dirty_navigation(self):
        if bool(getattr(self.widget, "auto_labeling_commit_blocked", False)):
            raise StagedAuditError("audit_integrity_refresh_required")
        if not bool(getattr(self.widget, "dirty", False)):
            return True
        answer = QtWidgets.QMessageBox.warning(
            self.widget,
            auto_labeling_text_v1("audit_unsaved_title"),
            auto_labeling_text_v1("audit_unsaved_prompt"),
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            if not self.widget.save_file() or bool(
                getattr(self.widget, "dirty", False)
            ):
                raise StagedAuditError("audit_save_failed")
        else:
            filename = getattr(self.widget, "filename", None)
            self.widget.set_clean()
            if filename and not self.widget.load_file(filename):
                raise StagedAuditError("audit_discard_reload_failed")
        self._ensure_clean_and_coordinated()
        return True

    def _start_load(self, image_id):
        if not self._resolve_dirty_navigation():
            return False
        return self._submit("load", {"image_id": image_id})

    def _present_item(self, item):
        item = self._remember_item(item)
        image_id = item["image_id"]
        image_path = item["canonical_image_path"]
        file_list = getattr(self.widget, "file_list_widget", None)
        blocked = False
        if file_list is not None:
            blocked = file_list.blockSignals(True)
            index = getattr(self.widget, "fn_to_index", {}).get(
                str(image_path)
            )
            if index is not None:
                file_list.setCurrentRow(index)
        try:
            if not self.widget.load_file(image_path):
                raise StagedAuditError("audit_image_load_failed", image_id)
        finally:
            if file_list is not None:
                file_list.blockSignals(blocked)
        self.current_image_id = image_id
        self._refresh_display(item)

    def _refresh_display(self, item=None):
        if self._closed or self.current_image_id is None:
            return
        if item is None:
            item = self._current_item()
        self.dialog.set_item(item, bool(getattr(self.widget, "dirty", False)))
        if self._summary is not None:
            self.dialog.set_summary(self._summary)

    def _warn(self, exc):
        code = getattr(exc, "code", "audit_action_failed")
        detail = str(exc)
        QtWidgets.QMessageBox.warning(
            self.widget,
            auto_labeling_text_v1("audit_action_failed"),
            f"{code}\n{detail}" if detail != code else code,
        )

    def approve_and_next(self):
        try:
            self._ensure_clean_and_coordinated()
            return self._submit(
                "approve",
                {"image_id": self.current_image_id},
            )
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def save_approve_and_next(self):
        try:
            if not self.widget.save_file():
                raise StagedAuditError("audit_save_failed")
            self._ensure_clean_and_coordinated()
            return self._submit(
                "approve",
                {"image_id": self.current_image_id},
            )
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def mark_needs_fix(self):
        try:
            self._ensure_clean_and_coordinated()
            return self._submit(
                "needs_fix",
                {"image_id": self.current_image_id},
            )
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def previous(self):
        try:
            current = self._current_item()["sequence"]
            candidates = [
                item for item in self.items if item["sequence"] < current
            ]
            if candidates:
                return self._start_load(candidates[-1]["image_id"])
            return False
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def next_pending(self):
        try:
            return self._start_next_pending()
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def _start_next_pending(self):
        current = self._current_item()["sequence"]
        pending = sorted(
            (
                item
                for item in self.items
                if item["review_status"] in {"pending", "needs_fix", "stale"}
            ),
            key=lambda value: value["sequence"],
        )
        candidates = [item for item in pending if item["sequence"] > current]
        if not candidates:
            candidates = [
                item for item in pending if item["sequence"] < current
            ]
        if candidates:
            return self._start_load(candidates[0]["image_id"])
        self.finish()
        return True

    def finish(self):
        if self._closed:
            return True
        if self._active_operation is not None:
            self._finish_requested = True
            self.dialog.set_busy(True, finishing=True)
            return False
        try:
            if not self._resolve_dirty_navigation():
                return False
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False
        self._finish_requested = True
        self._begin_shutdown()
        return True

    def is_idle(self):
        return self._active_operation is None

    def requires_safe_shutdown(self):
        return self.worker_thread is not None

    def request_safe_shutdown(self, _reason):
        if self._shutdown_generation is None:
            self._next_shutdown_generation += 1
            self._shutdown_generation = self._next_shutdown_generation
        self._finish_requested = True
        if self._active_operation is None:
            self._begin_shutdown()
        else:
            self.dialog.set_busy(True, finishing=True)
        return self._shutdown_generation

    def _begin_shutdown(self):
        if not self._closed:
            self._closed = True
            self._set_operation_busy(False)
            self.guard.release()
            self.dialog.allow_close()
            self.dialog.close()
        thread = self.worker_thread
        if thread is None:
            self._finish_thread_cleanup()
        elif thread.isRunning():
            thread.quit()

    @QtCore.pyqtSlot()
    def _on_thread_finished(self):
        worker = self._worker
        if worker is not None:
            try:
                self._execute_requested.disconnect(worker.execute)
            except (TypeError, RuntimeError):
                pass
        thread = self.worker_thread
        self._worker = None
        self.worker_thread = None
        if thread is not None:
            thread.deleteLater()
        self._finish_thread_cleanup()

    def _finish_thread_cleanup(self):
        self._closed = True
        if getattr(self.widget, "_auto_labeling_audit_session", None) is self:
            self.widget._auto_labeling_audit_session = None
        panel_remover = getattr(
            self.widget,
            "remove_auto_labeling_audit_panel",
            None,
        )
        if callable(panel_remover):
            panel_remover(self.dialog)
        generation = self._shutdown_generation
        if generation is not None:
            QtCore.QTimer.singleShot(
                0,
                lambda: self.safe_to_close.emit(generation),
            )

    def _publish_pending_count(self, image_id=None):
        refresh_counts = getattr(
            self.widget,
            "refresh_auto_labeling_pending_review_counts",
            None,
        )
        if callable(refresh_counts) and refresh_counts(image_id) is not None:
            return
        if self._summary is None:
            return
        count = self._summary.get("pending_review")
        if count is None:
            count = sum(
                int(self._summary.get(field, 0) or 0)
                for field in ("pending", "needs_fix", "stale")
            )
        setter = getattr(
            self.widget,
            "set_auto_labeling_pending_review_count",
            None,
        )
        if callable(setter):
            setter(count, scope="session")
