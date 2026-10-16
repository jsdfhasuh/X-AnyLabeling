"""Non-modal staged audit UI using stable image identities."""

import hashlib
import threading

from PyQt5 import QtCore, QtWidgets

from anylabeling.views.labeling.utils.auto_labeling_audit import (
    StagedAuditClientProtocolV1,
    StagedAuditError,
)
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    review_text_v1,
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
    batch_progress = QtCore.pyqtSignal(int, int, int)

    def __init__(self, client):
        super().__init__()
        self.client = client
        self._views = {}
        self._summary = None

    @QtCore.pyqtSlot(int, str, object)
    def execute(self, operation_id, kind, payload):
        try:
            result = self._execute(
                kind, dict(payload, operation_id=operation_id)
            )
        except Exception as exc:  # noqa: B902
            self.failed.emit(operation_id, exc)
            return
        self.completed.emit(operation_id, result)

    def _execute(self, kind, payload):
        if kind in {"begin", "batch_prepare"}:
            snapshot_reader = getattr(
                self.client, "read_review_snapshot", None
            )
            snapshot = (
                snapshot_reader() if callable(snapshot_reader) else None
            )
            items = sorted(
                snapshot["items"] if snapshot is not None
                else self.client.list_review_items(),
                key=lambda value: value["sequence"],
            )
            if kind == "begin":
                self._views = {}
            self._views.update({item["image_id"]: dict(item) for item in items})
            self._summary = dict(
                snapshot["summary"] if snapshot is not None
                else self.client.summary()
            )
            # Standalone clients count conflicts separately; the embedded
            # host already includes them in stale. Normalize display only.
            self._summary["stale"] = (
                self._summary.get("stale", 0)
                + self._summary.pop("conflict", 0)
            )
            self._update_pending_count()
            if kind == "batch_prepare":
                return {"items": items, "summary": dict(self._summary)}
            if not items:
                return {
                    "items": [], "item": None,
                    "summary": dict(self._summary),
                }
            item = self._read_item(items[0]["image_id"])
            self._remember_view(item)
            return {
                "items": items,
                "item": item,
                "summary": dict(self._summary),
            }
        if kind in {"load", "complete", "refresh"}:
            updated = []
            previous_id = payload.get("previous_image_id")
            if previous_id and previous_id != payload["image_id"]:
                # An invalid previous image must not prevent leaving it.
                # Keep its error in the display cache; validate the target.
                previous = self._read_item(previous_id)
                self._remember_view(previous)
                updated.append(previous)
            item = self._read_item(payload["image_id"])
            self._remember_view(item)
            updated.append(item)
            return {
                "item": item,
                "summary": dict(self._summary),
                "items": updated,
            }
        if kind in {"approve", "needs_fix"}:
            image_id = payload["image_id"]
            refreshed = self.client.integrity_refresh(image_id)
            self._raise_integrity_error(refreshed)
            for key, expected in payload.get("observed", {}).items():
                if refreshed.get(key) != expected:
                    raise StagedAuditError("audit_integrity_display_changed")
            revision = refreshed["item_revision"]
            if kind == "approve":
                item = self.client.approve(image_id, revision)
            else:
                item = self.client.needs_fix(image_id, revision)
            self._remember_view(item)
            return {"item": item, "summary": dict(self._summary)}
        if kind == "batch_approve":
            return self._approve_batch(payload)
        raise ValueError(f"unknown audit operation: {kind}")

    def _approve_batch(self, payload):
        candidates = payload["items"]
        updated, skipped = [], []
        approved = 0
        for expected in candidates:
            if payload["cancel"].is_set():
                break
            image_id = expected["image_id"]
            try:
                current = self.client.read_review_item(image_id)
                self._raise_integrity_error(current)
                if (
                    expected["review_status"] != "pending"
                    or current["review_status"] != "pending"
                    or current["item_revision"] != expected["item_revision"]
                    or current.get("audit_scope") != expected.get("audit_scope")
                ):
                    raise StagedAuditError("audit_batch_item_changed")
                refreshed = self.client.integrity_refresh(image_id)
                self._raise_integrity_error(refreshed)
                for key in ("review_semantic_digest", "review_image_digest"):
                    observed = expected.get("observed", {}).get(
                        key, expected.get(key)
                    )
                    if not observed or refreshed.get(key) != observed:
                        raise StagedAuditError("audit_integrity_display_changed")
                if (
                    refreshed["review_status"] != "pending"
                    or refreshed.get("audit_scope") != expected.get("audit_scope")
                ):
                    raise StagedAuditError("audit_batch_item_changed")
                current = self.client.approve(
                    image_id, refreshed["item_revision"]
                )
                if current["review_status"] not in {"approved", "staged_approved"}:
                    raise StagedAuditError("audit_batch_confirmation_unverified")
                approved += 1
            except Exception as exc:  # noqa: B902
                # Keep successful decisions; never retry an uncertain write.
                skipped.append({
                    "image_id": image_id, "filename": expected["filename"],
                    "error": str(exc),
                })
                current = self._read_item(image_id)
            self._remember_view(current)
            updated.append(current)
            self.batch_progress.emit(
                payload.get("operation_id", -1), len(updated), len(candidates),
            )
        return {
            "items": updated, "summary": dict(self._summary),
            "approved": approved, "skipped": skipped,
            "remaining": len(candidates) - len(updated),
        }

    @staticmethod
    def _summary_key(item):
        status = item["review_status"]
        if status == "approved":
            return (
                "staged_approved"
                if item.get("audit_scope") == "STAGED"
                else "source_approved"
            )
        if status == "conflict":
            return "stale"
        return status

    def _remember_view(self, item):
        # This cache is for display only. Approval always refreshes integrity
        # and obtains the current revision from the authority above.
        previous = self._views.get(item["image_id"])
        if previous is not None:
            for view, delta in ((previous, -1), (item, 1)):
                key = self._summary_key(view)
                if key in self._summary:
                    self._summary[key] += delta
        self._views[item["image_id"]] = dict(item)
        self._update_pending_count()

    def _update_pending_count(self):
        self._summary["pending_review"] = sum(
            self._summary.get(key, 0)
            for key in ("pending", "needs_fix", "stale")
        )

    def _read_item(self, image_id):
        # Preserve queue identity even when this particular file is broken.
        try:
            return self.client.read_review_item(image_id)
        except Exception as exc:  # noqa: B902
            if image_id not in self._views:
                raise
            return dict(
                self._views[image_id],
                integrity_error=str(exc),
            )

    @staticmethod
    def _raise_integrity_error(item):
        error = item.get("integrity_error")
        if error:
            raise StagedAuditError(error)


class _FilenameLabel(QtWidgets.QLabel):
    def set_filename(self, filename):
        self._filename = filename
        self.setToolTip(filename)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(
            getattr(self, "_filename", ""), QtCore.Qt.ElideMiddle,
            max(0, self.contentsRect().width()),
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


def _build_audit_controls(view):
    view._busy = False
    view._complete = False
    view._has_item = False
    layout = QtWidgets.QVBoxLayout(view)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(10)
    title = QtWidgets.QLabel(view.text("title"))
    font = title.font()
    font.setBold(True)
    title.setFont(font)
    layout.addWidget(title)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    body = QtWidgets.QWidget()
    content = QtWidgets.QVBoxLayout(body)
    content.setContentsMargins(0, 0, 4, 0)
    content.setSpacing(10)
    view.position_label = QtWidgets.QLabel()
    view.identity_label = _FilenameLabel()
    view.identity_label.setMinimumWidth(0)
    view.identity_label.setSizePolicy(
        QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
    )
    view.status_label = QtWidgets.QLabel()
    view.detail_label = QtWidgets.QLabel()
    view.counts_label = QtWidgets.QLabel()
    view.summary_detail_label = QtWidgets.QLabel()
    view.activity_label = QtWidgets.QLabel(view.text("begin"))
    view.error_label = QtWidgets.QLabel()
    view.batch_result_label = QtWidgets.QLabel()
    view.error_label.setStyleSheet("color: #b42318;")
    for label in (
        view.position_label, view.identity_label, view.status_label,
        view.detail_label, view.counts_label, view.summary_detail_label,
        view.activity_label, view.error_label, view.batch_result_label,
    ):
        label.setTextFormat(QtCore.Qt.PlainText)
        if label is not view.identity_label:
            label.setWordWrap(True)
        content.addWidget(label)
        label.hide()
    view.error_label.hide()
    view.activity_bar = QtWidgets.QProgressBar()
    view.activity_bar.setRange(0, 0)
    view.activity_bar.setTextVisible(False)
    view.activity_bar.setFixedHeight(4)
    view.activity_bar.hide()
    content.addWidget(view.activity_bar)
    view.cancel_batch_button = QtWidgets.QPushButton(view.text("batch_stop"))
    view.cancel_batch_button.clicked.connect(view.batch_stop_requested)
    view.cancel_batch_button.hide()
    content.addWidget(view.cancel_batch_button)

    view.reload_button = QtWidgets.QPushButton(view.text("reload"))
    view.reload_button.setIcon(view.style().standardIcon(
        QtWidgets.QStyle.SP_BrowserReload
    ))
    view.reload_button.hide()
    view.reload_button.clicked.connect(view.reload_requested)
    content.addWidget(view.reload_button)
    view.recover_button = QtWidgets.QPushButton(view.text("retry_save"))
    view.recover_button.setIcon(view.style().standardIcon(
        QtWidgets.QStyle.SP_DialogSaveButton
    ))
    view.recover_button.hide()
    view.recover_button.clicked.connect(view.recover_requested)
    content.addWidget(view.recover_button)
    view.details_button = QtWidgets.QToolButton()
    view.details_button.setText(view.text("details"))
    view.details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    view.details_button.setArrowType(QtCore.Qt.RightArrow)
    view.details_button.setCheckable(True)
    view.details_button.hide()
    view.technical_details = QtWidgets.QPlainTextEdit()
    view.technical_details.setReadOnly(True)
    view.technical_details.setFixedHeight(130)
    view.technical_details.hide()

    def toggle_details(expanded):
        view.technical_details.setVisible(expanded)
        view.details_button.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
        )

    view.details_button.toggled.connect(toggle_details)
    content.addWidget(view.details_button)
    content.addWidget(view.technical_details)
    content.addStretch(1)
    scroll.setWidget(body)
    layout.addWidget(scroll, 1)

    navigation = QtWidgets.QHBoxLayout()
    for name, key, icon, signal in (
        ("previous_button", "previous", QtWidgets.QStyle.SP_ArrowBack,
         view.previous_requested),
        ("next_button", "next", QtWidgets.QStyle.SP_ArrowForward,
         view.next_requested),
    ):
        button = QtWidgets.QToolButton()
        button.setIcon(view.style().standardIcon(icon))
        button.setToolTip(view.text(key))
        button.setAccessibleName(view.text(key))
        button.setMinimumSize(32, 32)
        button.clicked.connect(signal)
        setattr(view, name, button)
        navigation.addWidget(button)
    navigation.addStretch(1)
    view.defer_button = QtWidgets.QPushButton(view.text("defer"))
    view.defer_button.clicked.connect(view.defer_requested)
    navigation.addWidget(view.defer_button)
    layout.addLayout(navigation)

    view.approve_button = QtWidgets.QPushButton(view.text("confirm"))
    view.approve_button.setIcon(view.style().standardIcon(
        QtWidgets.QStyle.SP_DialogApplyButton
    ))
    view.approve_button.setMinimumHeight(38)
    view.approve_button.setStyleSheet(
        "QPushButton { background: #167568; color: white; border: none;"
        "border-radius: 4px; padding: 8px; }"
        "QPushButton:hover { background: #126358; }"
        "QPushButton:disabled { background: #d9dfdd; color: #56615d; }"
    )
    view.approve_button.clicked.connect(view.approve_requested)
    layout.addWidget(view.approve_button)
    view.batch_button = QtWidgets.QPushButton(view.text("batch_entry"))
    view.batch_button.setIcon(view.style().standardIcon(
        QtWidgets.QStyle.SP_DialogApplyButton
    ))
    view.batch_button.clicked.connect(view.batch_requested)
    layout.addWidget(view.batch_button)
    view.needs_fix_button = QtWidgets.QPushButton(view.text("needs_fix"))
    view.needs_fix_button.clicked.connect(view.needs_fix_requested)
    view.finish_button = QtWidgets.QPushButton(view.text("finish"))
    view.finish_button.clicked.connect(view.finish_requested)
    layout.addWidget(view.needs_fix_button)
    layout.addWidget(view.finish_button)
    view.set_busy(False)


class _AuditView:
    def text(self, key, **values):
        return review_text_v1(key, self._review_config, **values)

    def set_busy(self, busy, finishing=False, stage=None):
        self._busy = bool(busy)
        for button in (
            self.previous_button, self.next_button, self.needs_fix_button,
            self.approve_button, self.defer_button,
        ):
            button.setEnabled(
                not busy and self._has_item and not self._complete
            )
        if not getattr(self, "_can_decide", True):
            self.approve_button.setEnabled(False)
            self.needs_fix_button.setEnabled(False)
        self.batch_button.setEnabled(
            not busy and not finishing and self._has_item and not self._complete
        )
        self.cancel_batch_button.setVisible(busy and stage == "batch_approve")
        self.cancel_batch_button.setEnabled(not finishing)
        self.reload_button.setEnabled(not busy and not finishing)
        self.recover_button.setEnabled(not busy and not finishing)
        self.finish_button.setEnabled(not finishing)
        self.activity_bar.setVisible(bool(busy))
        self.activity_label.setVisible(bool(busy))
        self.activity_bar.setRange(0, 0)
        if busy:
            self.activity_label.setText(
                self.text("finishing" if finishing else (stage or "load"))
            )

    def set_item(self, item, dirty, *, clear_error=True, can_decide=True):
        self._has_item = True
        self._can_decide = can_decide
        self._complete = False
        for control in (
            self.identity_label, self.status_label, self.detail_label,
            self.details_button,
        ):
            control.show()
        self.identity_label.set_filename(item["filename"])
        state = {
            "pending": "pending", "approved": "approved",
            "staged_approved": "approved", "needs_fix": "needs_fix_status",
            "stale": "stale", "conflict": "conflict",
        }.get(item["review_status"], "unknown")
        self.status_label.setText(self.text(state))
        saved = (
            "source_saved"
            if item.get("source_commit_status") == "committed"
            else "saved"
            if item.get("staged_commit_status") == "committed"
            else "not_saved"
        )
        count = item.get("target_count")
        objects = (
            self.text("count_unknown") if count is None
            else self.text("no_targets") if count == 0
            else self.text("targets", count=count)
        )
        self.detail_label.setText(
            self.text("dirty" if dirty else saved) + "\n" + objects
        )
        self.approve_button.setText(
            self.text("save_confirm" if dirty else "confirm")
        )
        if clear_error or self.error_label.isHidden():
            self.technical_details.setPlainText(
                "\n".join(f"{key}: {item.get(key)}" for key in (
                    "image_id", "item_revision", "audit_scope",
                    "staged_commit_status", "source_commit_status",
                    "canonical_label_path",
                ))
            )
        if clear_error:
            self.error_label.hide()
            self.reload_button.hide()
            self.recover_button.hide()
        self.set_busy(self._busy)

    def set_position(self, index, total):
        self.position_label.show()
        self.position_label.setText(
            self.text("position", index=index, total=total)
        )
        self.previous_button.setEnabled(not self._busy and index > 1)

    def set_summary(self, summary):
        self.counts_label.show()
        self.summary_detail_label.show()
        pending = sum(summary.get(k, 0) for k in (
            "pending", "needs_fix", "stale"
        ))
        approved = sum(summary.get(k, 0) for k in (
            "staged_approved", "source_approved"
        ))
        self.counts_label.setText(
            self.text("summary", pending=pending, approved=approved)
        )
        self.summary_detail_label.setText(self.text(
            "summary_detail", needs_fix=summary.get("needs_fix", 0),
            stale=summary.get("stale", 0),
        ))

    def set_complete(self, summary):
        self._complete = True
        self.status_label.show()
        self.detail_label.show()
        self.status_label.setText(self.text("complete"))
        remaining = sum(summary.get(k, 0) for k in (
            "pending", "needs_fix", "stale"
        ))
        self.detail_label.setText(
            self.text("remaining", count=remaining)
            if remaining else self.text("all_done")
        )
        self.set_summary(summary)
        self.set_busy(False)

    def set_error(self, exc, recovery_required=False, item_unavailable=False):
        code = getattr(exc, "code", "")
        key = (
            "recovery_error" if recovery_required
            else "load_error" if item_unavailable
            else "save_error" if code == "audit_save_failed"
            else "integrity_error" if "integrity" in code
            else "error"
        )
        self.error_label.setText(self.text(key))
        self.error_label.show()
        self.details_button.show()
        self.technical_details.appendPlainText(str(exc))
        self.reload_button.setVisible(not recovery_required)
        self.recover_button.setVisible(recovery_required)

    def allow_close(self):
        self._allow_close = True

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
            return
        self.finish_requested.emit()
        event.ignore()


class AutoLabelingAuditDialog(_AuditView, QtWidgets.QDialog):
    batch_requested = QtCore.pyqtSignal()
    batch_stop_requested = QtCore.pyqtSignal()
    approve_requested = QtCore.pyqtSignal()
    save_approve_requested = QtCore.pyqtSignal()
    needs_fix_requested = QtCore.pyqtSignal()
    previous_requested = QtCore.pyqtSignal()
    next_requested = QtCore.pyqtSignal()
    defer_requested = QtCore.pyqtSignal()
    reload_requested = QtCore.pyqtSignal()
    recover_requested = QtCore.pyqtSignal()
    finish_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self._review_config = config or getattr(parent, "_config", {})
        self._allow_close = False
        self.setWindowTitle(self.text("title"))
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setMinimumWidth(280)
        self.resize(380, 600)
        _build_audit_controls(self)


class AutoLabelingAuditPanel(_AuditView, QtWidgets.QWidget):
    """Embedded review controls for the labeling sidebar."""

    approve_requested = QtCore.pyqtSignal()
    batch_requested = QtCore.pyqtSignal()
    batch_stop_requested = QtCore.pyqtSignal()
    save_approve_requested = QtCore.pyqtSignal()
    needs_fix_requested = QtCore.pyqtSignal()
    previous_requested = QtCore.pyqtSignal()
    next_requested = QtCore.pyqtSignal()
    defer_requested = QtCore.pyqtSignal()
    reload_requested = QtCore.pyqtSignal()
    recover_requested = QtCore.pyqtSignal()
    finish_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self._review_config = config or getattr(parent, "_config", {})
        self._allow_close = False
        self.setMinimumWidth(280)
        _build_audit_controls(self)


class BatchConfirmationDialog(QtWidgets.QDialog):
    """Explicit consent for a frozen set of pending annotations."""

    def __init__(self, items, current_sequence, parent=None, config=None):
        super().__init__(parent)
        self._items = items
        self._current_sequence = current_sequence
        self._config = config or {}
        self.setWindowTitle(self.text("batch_title"))
        self.resize(460, 260)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.addItem(self.text("batch_all"), "all")
        self.range_combo.addItem(self.text("batch_from_current"), "remaining")
        form = QtWidgets.QFormLayout()
        form.addRow(self.text("batch_range"), self.range_combo)
        layout.addLayout(form)
        self.count_label = QtWidgets.QLabel()
        font = self.count_label.font()
        font.setBold(True)
        self.count_label.setFont(font)
        self.notice_label = QtWidgets.QLabel(self.text("batch_notice"))
        for label in (self.count_label, self.notice_label):
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.PlainText)
            layout.addWidget(label)
        layout.addStretch()
        buttons = QtWidgets.QDialogButtonBox()
        buttons.setStyleSheet("QPushButton { padding: 6px 12px; }")
        self.confirm_button = buttons.addButton(
            self.text("batch_confirm", count=0),
            QtWidgets.QDialogButtonBox.AcceptRole,
        )
        self.confirm_button.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_DialogApplyButton
        ))
        cancel = buttons.addButton(
            self.text("batch_cancel"), QtWidgets.QDialogButtonBox.RejectRole,
        )
        self.confirm_button.setAutoDefault(False)
        cancel.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.range_combo.currentIndexChanged.connect(self._update_count)
        self._update_count()

    def text(self, key, **values):
        return review_text_v1(key, self._config, **values)

    def selected_items(self):
        if self.range_combo.currentData() == "all":
            return list(self._items)
        return [
            item for item in self._items
            if item["sequence"] >= self._current_sequence
        ]

    def _update_count(self):
        count = len(self.selected_items())
        self.count_label.setText(self.text("batch_count", count=count))
        self.confirm_button.setText(self.text("batch_confirm", count=count))
        self.confirm_button.setEnabled(count > 0)


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
        self._item_presented = False
        self._displayed_observation = {}
        self.worker_thread = None
        self._worker = None
        self._summary = None
        self._active_operation = None
        self._next_operation_id = 0
        self._finish_requested = False
        self._shutdown_generation = None
        self._next_shutdown_generation = 0
        self._closed = False
        self._deferred_ids = set()
        self._round_complete = False
        self._batch_cancel = None
        self._connect()

    def _connect(self):
        self.dialog.approve_requested.connect(self.confirm_and_next)
        self.dialog.batch_requested.connect(self.confirm_batch)
        self.dialog.batch_stop_requested.connect(self.stop_batch)
        self.dialog.save_approve_requested.connect(self.save_approve_and_next)
        self.dialog.needs_fix_requested.connect(self.mark_needs_fix)
        self.dialog.previous_requested.connect(self.previous)
        self.dialog.next_requested.connect(self.next_pending)
        self.dialog.finish_requested.connect(self.finish)
        self.dialog.defer_requested.connect(self.defer_current)
        self.dialog.reload_requested.connect(self.reload_current)
        self.dialog.recover_requested.connect(self.recover_save)
        save = getattr(getattr(self.widget, "actions", None), "save", None)
        if save is not None:
            save.changed.connect(self._refresh_dirty_display)
        changed = getattr(self.widget, "annotation_changed", None)
        if changed is not None:
            changed.connect(self._refresh_dirty_display)

    def _refresh_dirty_display(self):
        if self._closed or self._finish_requested or not self.is_idle():
            return
        dirty = bool(getattr(self.widget, "dirty", False))
        if self._round_complete:
            if not dirty:
                return
            self._round_complete = False
        self._refresh_display(clear_error=(
            not dirty
            and not getattr(self.widget, "auto_labeling_commit_blocked", False)
        ))

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
        self._worker.batch_progress.connect(self._on_batch_progress)
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
        self.guard.set_transaction_busy(
            busy or (self.current_image_id is not None and not self._item_presented)
        )
        kind = (self._active_operation or {}).get("kind", "load")
        stage = {
            "needs_fix": "marking", "complete": "load", "refresh": "load",
        }.get(kind, kind)
        self.dialog.set_busy(
            busy, finishing=self._finish_requested, stage=stage
        )

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
        if operation["kind"] == "batch_approve":
            self._batch_cancel = None
        self._set_operation_busy(False)
        self._warn(exc)
        if self._finish_requested:
            self._begin_shutdown()

    def _take_operation(self, operation_id):
        operation = self._active_operation
        if operation is None or operation["operation_id"] != operation_id:
            return None
        self._active_operation = None
        return operation

    def _apply_operation_result(self, kind, result):
        if kind == "batch_prepare":
            if self._finish_requested:
                return
            self._merge_items(result["items"])
            self._summary = result["summary"]
            self.dialog.set_summary(self._summary)
            candidates = [
                dict(item) for item in result["items"]
                if item["review_status"] == "pending"
                and item["image_id"] not in self._deferred_ids
                and not item.get("integrity_error")
                and (item["image_id"] != self.current_image_id or self._item_presented)
            ]
            for item in candidates:
                if item["image_id"] == self.current_image_id:
                    item["observed"] = self._decision_payload()["observed"]
            self.guard.set_transaction_busy(True)
            try:
                selected = self._choose_batch(candidates)
            finally:
                self._set_operation_busy(False)
            if selected and not self._closed and not self._finish_requested:
                self._batch_cancel = threading.Event()
                self.dialog.batch_result_label.hide()
                self._submit("batch_approve", {
                    "items": selected, "cancel": self._batch_cancel,
                })
            return
        if kind == "batch_approve":
            self._batch_cancel = None
            self._merge_items(result["items"])
            self._summary = result["summary"]
            self._publish_pending_count()
            self._refresh_display()
            if not self._summary.get("pending_review"):
                self._round_complete = True
                self.dialog.set_complete(self._summary)
            self.dialog.batch_result_label.setText(self.dialog.text(
                "batch_result", approved=result["approved"],
                skipped=len(result["skipped"]), remaining=result["remaining"],
            ))
            self.dialog.batch_result_label.show()
            self.dialog.batch_result_label.setToolTip("\n".join(
                f'{item["filename"]}: {item["error"]}'
                for item in result["skipped"]
            ))
            for item in result["skipped"]:
                self.dialog.technical_details.appendPlainText(
                    f'{item["filename"]}: {item["error"]}'
                )
            if (
                not self._round_complete and not self._finish_requested
                and self._current_item()["review_status"] in {
                    "approved", "staged_approved",
                }
            ):
                self._start_next_pending()
            return
        if kind == "begin":
            self.items = list(result["items"])
            self._summary = result["summary"]
            if not self.items:
                self._round_complete = True
                self.dialog.set_complete(self._summary)
                self._publish_pending_count()
                return
            self._publish_pending_count(result["item"].get("image_id"))
            self._present_item(result["item"])
            return
        for updated in result.get("items", []):
            self._remember_item(updated)
        item = self._remember_item(result["item"])
        self._summary = result["summary"]
        self._publish_pending_count(item.get("image_id"))
        if kind == "load":
            self._present_item(item)
        elif kind == "refresh":
            if self._round_complete:
                self.dialog.set_complete(self._summary)
            else:
                self._refresh_display(item)
        elif kind == "complete":
            self._round_complete = True
            self.dialog.set_complete(self._summary)
        elif kind == "approve":
            self._start_next_pending()
        elif kind == "needs_fix":
            self._refresh_display(item)
        else:
            raise ValueError(f"unknown audit result: {kind}")

    def _merge_items(self, items):
        by_id = {item["image_id"]: item for item in self.items}
        for item in items:
            by_id[item["image_id"]] = dict(item)
        self.items = sorted(by_id.values(), key=lambda item: item["sequence"])

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
        answer = self._ask_unsaved_changes()
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

    def _ask_unsaved_changes(self):
        message = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Warning,
            self.dialog.text("unsaved_title"),
            self.dialog.text("unsaved_prompt"),
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            self.widget,
        )
        for role, key in (
            (QtWidgets.QMessageBox.Save, "save"),
            (QtWidgets.QMessageBox.Discard, "discard"),
            (QtWidgets.QMessageBox.Cancel, "cancel"),
        ):
            message.button(role).setText(self.dialog.text(key))
        message.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        return message.exec_()

    def _start_load(self, image_id):
        if not self._can_act():
            return False
        if not self._resolve_dirty_navigation():
            return False
        return self._submit("load", {
            "image_id": image_id,
            "previous_image_id": self.current_image_id,
        })

    def _present_item(self, item):
        item = self._remember_item(item)
        image_id = item["image_id"]
        image_path = item["canonical_image_path"]
        self.current_image_id = image_id
        self._item_presented = False
        self._displayed_observation = {}
        self._round_complete = False
        self._deferred_ids.discard(image_id)
        self.guard.set_transaction_busy(True)
        self._refresh_display(item)
        _StagedAuditOperationWorker._raise_integrity_error(item)
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
        self._item_presented = True
        self.guard.set_transaction_busy(False)
        self._displayed_observation = {
            key: item[key] for key in (
                "review_semantic_digest", "review_image_digest",
            ) if key in item
        }
        if hasattr(self.widget, "label_file"):
            self._displayed_observation["review_semantic_digest"] = getattr(
                self.widget.label_file, "semantic_digest", None,
            )
        image_data = getattr(self.widget, "image_data", None)
        if isinstance(image_data, bytes):
            self._displayed_observation["review_image_digest"] = (
                hashlib.sha256(image_data).hexdigest()
            )
        self._refresh_display(item)

    def _refresh_display(self, item=None, *, clear_error=True):
        if self._closed or self.current_image_id is None:
            return
        if item is None:
            item = self._current_item()
        shapes = getattr(getattr(self.widget, "canvas", None), "shapes", None)
        if isinstance(shapes, (list, tuple)):
            item = dict(item, target_count=len(shapes))
        self.dialog.set_item(
            item, bool(getattr(self.widget, "dirty", False)),
            clear_error=clear_error,
            can_decide=self._item_presented,
        )
        self.dialog.set_position(
            next(i + 1 for i, value in enumerate(self.items)
                 if value["image_id"] == item["image_id"]),
            len(self.items),
        )
        if self._summary is not None:
            self.dialog.set_summary(self._summary)

    def _warn(self, exc):
        self.dialog.set_error(
            exc, recovery_required=bool(
                getattr(self.widget, "auto_labeling_commit_blocked", False)
            ),
            item_unavailable=(
                self.current_image_id is not None and not self._item_presented
            ),
        )

    def _can_act(self):
        return (
            not self._closed and not self._finish_requested
            and self.is_idle() and not self._round_complete
            and self.current_image_id is not None
        )

    def confirm_and_next(self):
        if not self._can_act():
            return False
        if bool(getattr(self.widget, "dirty", False)):
            return self.save_approve_and_next()
        return self.approve_and_next()

    def confirm_batch(self):
        if not self._can_act():
            return False
        try:
            if not self._resolve_dirty_navigation():
                return False
            return self._submit("batch_prepare", {})
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def _choose_batch(self, candidates):
        dialog = BatchConfirmationDialog(
            candidates, self._current_item()["sequence"],
            parent=self.widget, config=self.dialog._review_config,
        )
        try:
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                return dialog.selected_items()
            return []
        finally:
            dialog.deleteLater()

    def stop_batch(self):
        if self._batch_cancel is not None:
            self._batch_cancel.set()
            self.dialog.cancel_batch_button.setEnabled(False)
            self.dialog.activity_label.setText(self.dialog.text("batch_stopping"))

    @QtCore.pyqtSlot(int, int, int)
    def _on_batch_progress(self, operation_id, processed, total):
        operation = self._active_operation or {}
        if operation.get("operation_id") != operation_id:
            return
        self.dialog.activity_bar.setRange(0, total)
        self.dialog.activity_bar.setValue(processed)
        if self._batch_cancel is not None and not self._batch_cancel.is_set():
            self.dialog.activity_label.setText(self.dialog.text(
                "batch_progress", processed=processed, total=total,
            ))

    def reload_current(self):
        if self._closed or self._finish_requested or not self.is_idle():
            return False
        try:
            if not self._resolve_dirty_navigation():
                return False
            if self.current_image_id is None:
                return self._submit("begin", {})
            return self._submit("load", {
                "image_id": self.current_image_id,
            })
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def recover_save(self):
        if self._closed or self._finish_requested or not self.is_idle():
            return False
        if not getattr(self.widget, "auto_labeling_commit_blocked", False):
            return False
        try:
            # Explicit save uses the existing commit bridge's recovery and
            # verification. Never turn a recovery click into an approval.
            if not self.widget.save_file():
                raise StagedAuditError("audit_save_failed")
            self._ensure_clean_and_coordinated()
            self.dialog.error_label.hide()
            self.dialog.recover_button.hide()
            if self.current_image_id is None:
                self.dialog.reload_button.show()
                return True
            return self._submit("refresh", {
                "image_id": self.current_image_id,
            })
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def defer_current(self):
        if not self._can_act():
            return False
        try:
            if not self._resolve_dirty_navigation():
                return False
            self._deferred_ids.add(self.current_image_id)
            return self._start_next_pending()
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def approve_and_next(self):
        if not self._can_act() or not self._item_presented:
            return False
        try:
            self._ensure_clean_and_coordinated()
            return self._submit(
                "approve",
                self._decision_payload(),
            )
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def save_approve_and_next(self):
        if not self._can_act() or not self._item_presented:
            return False
        try:
            if not self.widget.save_file():
                raise StagedAuditError("audit_save_failed")
            self._ensure_clean_and_coordinated()
            return self._submit(
                "approve",
                self._decision_payload(),
            )
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def mark_needs_fix(self):
        if not self._can_act() or not self._item_presented:
            return False
        try:
            self._ensure_clean_and_coordinated()
            return self._submit(
                "needs_fix",
                self._decision_payload(),
            )
        except Exception as exc:  # noqa: B902
            self._warn(exc)
            return False

    def _decision_payload(self):
        observed = dict(self._displayed_observation)
        label = getattr(self.widget, "label_file", None)
        semantic = getattr(label, "semantic_digest", None)
        if semantic is not None:
            # Both load and successful save retain the exact in-memory
            # document digest. Do not adopt an external disk refresh here.
            observed["review_semantic_digest"] = semantic
        return {"image_id": self.current_image_id, "observed": observed}

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
        if not self._can_act():
            return False
        current = self._current_item()["sequence"]
        pending = sorted(
            (
                item
                for item in self.items
                if item["review_status"] in {
                    "pending", "needs_fix", "stale", "conflict"
                }
                and item["image_id"] not in self._deferred_ids
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
        if not self._resolve_dirty_navigation():
            return False
        return self._submit("complete", {
            "image_id": self.current_image_id,
        })

    def finish(self):
        if self._closed:
            return True
        if self._active_operation is not None:
            self.stop_batch()
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
        self.stop_batch()
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
        # The worker may already be deleted in its own thread. Qt removes
        # its signal connections; do not access its bound methods here.
        thread = self.worker_thread
        self._worker = None
        self.worker_thread = None
        if thread is not None:
            thread.deleteLater()
        self._finish_thread_cleanup()

    def _finish_thread_cleanup(self):
        self._closed = True
        changed = getattr(self.widget, "annotation_changed", None)
        if changed is not None:
            try:
                changed.disconnect(self._refresh_dirty_display)
            except (TypeError, RuntimeError):
                pass
        save = getattr(getattr(self.widget, "actions", None), "save", None)
        if save is not None:
            try:
                save.changed.disconnect(self._refresh_dirty_display)
            except (TypeError, RuntimeError):
                pass
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
