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
        self.setWindowTitle(auto_labeling_text_v1("audit_title"))
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
        self.previous_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("previous")
        )
        self.next_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("next_pending")
        )
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        navigation.addStretch(1)
        layout.addLayout(navigation)

        decisions = QtWidgets.QHBoxLayout()
        self.needs_fix_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("mark_needs_fix")
        )
        self.save_approve_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("save_approve_next")
        )
        self.approve_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("approve_next")
        )
        self.finish_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("finish_audit")
        )
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
        return True

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
            item = self.client.read_review_item(self.current_image_id)
            item = self._remember_item(item)
        return item

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
        item = self._remember_item(item)
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
            item = self.client.read_review_item(self.current_image_id)
            item = self._remember_item(item)
        self.dialog.set_item(item, bool(getattr(self.widget, "dirty", False)))
        self.dialog.set_summary(self.client.summary())

    def _warn(self, exc):
        code = getattr(exc, "code", "audit_action_failed")
        detail = str(exc)
        QtWidgets.QMessageBox.warning(
            self.widget,
            auto_labeling_text_v1("audit_action_failed"),
            f"{code}\n{detail}" if detail != code else code,
        )

    def _verified_revision(self):
        self._ensure_clean_and_coordinated()
        refreshed = self.client.integrity_refresh(self.current_image_id)
        if refreshed.get("integrity_error"):
            raise StagedAuditError(refreshed["integrity_error"])
        self._remember_item(refreshed)
        return refreshed["item_revision"]

    def approve_and_next(self):
        try:
            revision = self._verified_revision()
            updated = self.client.approve(self.current_image_id, revision)
            self._remember_item(updated)
            self.next_pending()
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def save_approve_and_next(self):
        try:
            if not self.widget.save_file():
                raise StagedAuditError("audit_save_failed")
            revision = self._verified_revision()
            updated = self.client.approve(self.current_image_id, revision)
            self._remember_item(updated)
            self.next_pending()
        except Exception as exc:  # noqa: B902
            self._warn(exc)

    def mark_needs_fix(self):
        try:
            revision = self._verified_revision()
            updated = self.client.needs_fix(self.current_image_id, revision)
            self._remember_item(updated)
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
                (
                    item
                    for item in self.items
                    if item["review_status"]
                    in {"pending", "needs_fix", "stale"}
                ),
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
        self.guard.release()
        self.dialog.allow_close()
        self.dialog.close()
        if getattr(self.widget, "_auto_labeling_audit_session", None) is self:
            self.widget._auto_labeling_audit_session = None
        return True
