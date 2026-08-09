"""Non-modal staged audit UI using stable image identities."""

from PyQt5 import QtCore, QtWidgets

from anylabeling.views.labeling.utils.auto_labeling_audit import (
    StagedAuditClientProtocolV1,
    StagedAuditError,
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

    def __init__(self, widget):
        self.widget = widget
        self._states = []
        self._active = False

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
        for target, enabled in reversed(self._states):
            target.setEnabled(enabled)
        self._states = []
        self.widget.auto_labeling_audit_active = False
        self._active = False


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
        self.setWindowTitle(self.tr("连续自动标注审计"))
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setMinimumWidth(560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.identity_label = QtWidgets.QLabel()
        self.identity_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.status_label = QtWidgets.QLabel()
        self.detail_label = QtWidgets.QLabel()
        self.detail_label.setWordWrap(True)
        self.counts_label = QtWidgets.QLabel()
        layout.addWidget(self.identity_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.counts_label)

        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton(self.tr("上一张"))
        self.next_button = QtWidgets.QPushButton(self.tr("下一张待审计"))
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        navigation.addStretch(1)
        layout.addLayout(navigation)

        decisions = QtWidgets.QHBoxLayout()
        self.needs_fix_button = QtWidgets.QPushButton(self.tr("标记需修改"))
        self.save_approve_button = QtWidgets.QPushButton(
            self.tr("保存、通过并下一张")
        )
        self.approve_button = QtWidgets.QPushButton(self.tr("通过并下一张"))
        self.finish_button = QtWidgets.QPushButton(self.tr("结束审计"))
        decisions.addWidget(self.needs_fix_button)
        decisions.addStretch(1)
        decisions.addWidget(self.save_approve_button)
        decisions.addWidget(self.approve_button)
        decisions.addWidget(self.finish_button)
        layout.addLayout(decisions)

        self.previous_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)
        self.needs_fix_button.clicked.connect(self.needs_fix_requested)
        self.save_approve_button.clicked.connect(self.save_approve_requested)
        self.approve_button.clicked.connect(self.approve_requested)
        self.finish_button.clicked.connect(self.finish_requested)

    def set_item(self, item, dirty):
        self.identity_label.setText(
            self.tr(
                "{filename}  |  image_id={image_id}  |  #{sequence}"
            ).format(**item)
        )
        self.status_label.setText(
            self.tr("审计状态：{status}  |  dirty={dirty}").format(
                status=item["review_status"],
                dirty="true" if dirty else "false",
            )
        )
        self.detail_label.setText(
            self.tr(
                "staged={staged}  |  source={source}  |  "
                "target_count={target_count}  |  zero_target={zero_target}"
            ).format(
                staged=item["staged_commit_status"],
                source=item["source_commit_status"],
                target_count=item["target_count"],
                zero_target=item["zero_target"],
            )
        )

    def set_summary(self, summary):
        self.counts_label.setText(
            self.tr(
                "当前 Session 已通过 {staged_approved}  |  "
                "项目源已通过 {source_approved}  |  "
                "需修改 {needs_fix}  |  待审计 {pending}  |  "
                "stale {stale}"
            ).format(**summary)
        )

    def allow_close(self):
        self._allow_close = True

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
            return
        self.finish_requested.emit()
        event.ignore()


class StagedAuditUiSession(QtCore.QObject):
    """Sequence stable audit actions around one LabelingWidget."""

    dialog_class = AutoLabelingAuditDialog

    def __init__(self, widget, client):
        super().__init__(widget)
        if not isinstance(client, StagedAuditClientProtocolV1):
            raise TypeError(
                "client does not implement StagedAuditClientProtocolV1"
            )
        self.widget = widget
        self.client = client
        self.dialog = self.dialog_class(widget)
        self.guard = AuditInteractionGuard(widget)
        self.items = []
        self.current_image_id = None
        self._closed = False
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._refresh_display)
        self._connect()

    def _connect(self):
        self.dialog.approve_requested.connect(self.approve_and_next)
        self.dialog.save_approve_requested.connect(self.save_approve_and_next)
        self.dialog.needs_fix_requested.connect(self.mark_needs_fix)
        self.dialog.previous_requested.connect(self.previous)
        self.dialog.next_requested.connect(self.next_pending)
        self.dialog.finish_requested.connect(self.finish)

    def begin(self):
        self.items = sorted(
            self.client.list_review_items(),
            key=lambda value: value["sequence"],
        )
        if not self.items:
            return False
        self.guard.activate()
        self.dialog.show()
        self._load(self.items[0]["image_id"], initial=True)
        self._refresh_timer.start()
        return True

    def _current_item(self):
        if self.current_image_id is None:
            raise StagedAuditError("audit_current_image_missing")
        return self.client.read_review_item(self.current_image_id)

    def _ensure_clean_and_coordinated(self):
        if bool(getattr(self.widget, "dirty", False)):
            raise StagedAuditError("audit_dirty_requires_save")
        if bool(getattr(self.widget, "auto_labeling_commit_blocked", False)):
            raise StagedAuditError("audit_integrity_refresh_required")

    def _load(self, image_id, initial=False):
        if not initial:
            self._ensure_clean_and_coordinated()
        item = self.client.read_review_item(image_id)
        if item.get("integrity_error"):
            raise StagedAuditError(item["integrity_error"])
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
        self._refresh_display()

    def _refresh_display(self):
        if self._closed or self.current_image_id is None:
            return
        item = self.client.read_review_item(self.current_image_id)
        self.dialog.set_item(item, bool(getattr(self.widget, "dirty", False)))
        self.dialog.set_summary(self.client.summary())

    def _warn(self, exc):
        code = getattr(exc, "code", "audit_action_failed")
        detail = str(exc)
        QtWidgets.QMessageBox.warning(
            self.widget,
            self.tr("无法完成审计动作"),
            f"{code}\n{detail}" if detail != code else code,
        )

    def _verified_revision(self):
        self._ensure_clean_and_coordinated()
        refreshed = self.client.integrity_refresh(self.current_image_id)
        if refreshed.get("integrity_error"):
            raise StagedAuditError(refreshed["integrity_error"])
        return refreshed["item_revision"]

    def approve_and_next(self):
        try:
            revision = self._verified_revision()
            self.client.approve(self.current_image_id, revision)
            self.next_pending()
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def save_approve_and_next(self):
        try:
            if not self.widget.save_file():
                raise StagedAuditError("audit_save_failed")
            revision = self._verified_revision()
            self.client.approve(self.current_image_id, revision)
            self.next_pending()
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def mark_needs_fix(self):
        try:
            revision = self._verified_revision()
            self.client.needs_fix(self.current_image_id, revision)
            self._refresh_display()
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def previous(self):
        try:
            self._ensure_clean_and_coordinated()
            current = self._current_item()["sequence"]
            candidates = [
                item for item in self.items if item["sequence"] < current
            ]
            if candidates:
                self._load(candidates[-1]["image_id"])
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def next_pending(self):
        try:
            self._ensure_clean_and_coordinated()
            current = self._current_item()["sequence"]
            pending = sorted(
                self.client.list_review_items(),
                key=lambda value: value["sequence"],
            )
            candidates = [
                item for item in pending if item["sequence"] > current
            ]
            if not candidates:
                candidates = [
                    item for item in pending if item["sequence"] < current
                ]
            if candidates:
                self._load(candidates[0]["image_id"])
            else:
                self.finish()
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def finish(self):
        if self._closed:
            return True
        try:
            self._ensure_clean_and_coordinated()
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False
        self._closed = True
        self._refresh_timer.stop()
        self.guard.release()
        self.dialog.allow_close()
        self.dialog.close()
        if getattr(self.widget, "_auto_labeling_audit_session", None) is self:
            self.widget._auto_labeling_audit_session = None
        return True
