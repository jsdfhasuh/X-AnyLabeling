"""Phase 4 Fast setup, progress, and UI lifecycle coordination."""

import os

from PyQt5 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.views.labeling.utils.auto_labeling_host import (
    validate_auto_labeling_host_context,
)
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_boolean_text_v1,
    auto_labeling_error_category_v1,
    auto_labeling_status_text_v1,
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_audit import (
    InMemoryStagedAuditClientV1,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    resolve_existing_label,
)
from anylabeling.views.labeling.utils.auto_labeling_contracts import (
    validate_document_digest_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    FastRunOptionsV1,
    SequenceRunOptionsV1,
    build_model_fingerprint_v1,
    build_parameter_snapshot_v1,
    resolve_sequence_capabilities,
    validate_sequence_delay_seconds_v1,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    ContinuousAutoLabelingController,
    FastAutoLabelingController,
    FastCanvasStateGuard,
    FastControllerError,
    LabelingWidgetSequencePresenter,
    VisibleCanvasStateGuard,
    standalone_image_id_v1,
)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _record_value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


_DEFAULT_SEQUENCE_SETTINGS = {
    "delay_seconds": 2.0,
    "range": "CURRENT_TO_END",
    "filter": "ALL",
    "write_policy": "SKIP_EXISTING",
}


def continuous_start_failure_text_v1(exc):
    code = str(getattr(exc, "code", "sequence_run_start_failed"))
    detail = str(exc)
    return f"{code}\n{detail}" if detail != code else code


def continuous_auto_labeling_settings_v1(config):
    result = dict(_DEFAULT_SEQUENCE_SETTINGS)
    raw = config.get("continuous_auto_labeling", {})
    if type(raw) is not dict:
        return result
    try:
        result["delay_seconds"] = validate_sequence_delay_seconds_v1(
            raw.get("delay_seconds", result["delay_seconds"])
        )
    except Exception:
        pass
    for field, allowed in (
        ("range", {"ALL_IMAGES", "CURRENT_TO_END"}),
        ("filter", {"ALL", "ONLY_WITHOUT_VALID_ANNOTATION"}),
    ):
        if raw.get(field) in allowed:
            result[field] = raw[field]
    return result


class ContinuousRunSetupDialog(QtWidgets.QDialog):
    """Collect the unified Fast or Visible sequence options."""

    def __init__(
        self,
        summary,
        model_summary,
        *,
        current_anchor_available,
        initial_values=None,
        parent=None,
    ):
        super().__init__(parent)
        self.summary = dict(summary)
        self.setWindowTitle(auto_labeling_text_v1("continuous_title"))
        self.setMinimumWidth(560)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self.title_label)

        self.note_label = QtWidgets.QLabel()
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)

        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.addItem(
            auto_labeling_text_v1("all_images"), "ALL_IMAGES"
        )
        self.range_combo.addItem(
            auto_labeling_text_v1("current_to_end"), "CURRENT_TO_END"
        )
        values = dict(_DEFAULT_SEQUENCE_SETTINGS)
        if type(initial_values) is dict:
            values.update(initial_values)
        values["write_policy"] = "SKIP_EXISTING"
        if not current_anchor_available:
            model = self.range_combo.model()
            model.item(1).setEnabled(False)
            values["range"] = "ALL_IMAGES"
        self.range_combo.setCurrentIndex(
            max(0, self.range_combo.findData(values["range"]))
        )
        form.addRow(auto_labeling_text_v1("base_range"), self.range_combo)

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItem(auto_labeling_text_v1("all"), "ALL")
        self.filter_combo.addItem(
            auto_labeling_text_v1("only_without_valid_annotation"),
            "ONLY_WITHOUT_VALID_ANNOTATION",
        )
        self.filter_combo.setCurrentIndex(
            max(0, self.filter_combo.findData(values["filter"]))
        )
        form.addRow(auto_labeling_text_v1("filter"), self.filter_combo)

        self.write_policy_combo = QtWidgets.QComboBox()
        for text, value in (
            (auto_labeling_text_v1("skip_existing"), "SKIP_EXISTING"),
            (auto_labeling_text_v1("force_replace"), "FORCE_REPLACE"),
        ):
            self.write_policy_combo.addItem(text, value)
        self.write_policy_combo.setCurrentIndex(
            max(0, self.write_policy_combo.findData(values["write_policy"]))
        )
        form.addRow(
            auto_labeling_text_v1("write_policy"),
            self.write_policy_combo,
        )

        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 60.0)
        self.delay_spin.setSingleStep(0.5)
        self.delay_spin.setDecimals(1)
        self.delay_spin.setSuffix(auto_labeling_text_v1("seconds_suffix"))
        self.delay_spin.setValue(float(values["delay_seconds"]))
        form.addRow(auto_labeling_text_v1("dwell_time"), self.delay_spin)
        form.addRow(
            auto_labeling_text_v1("model_and_parameters"),
            QtWidgets.QLabel(model_summary),
        )
        form.addRow(
            auto_labeling_text_v1("workset_count"),
            QtWidgets.QLabel(str(self.summary.get("workset_total", 0))),
        )
        form.addRow(
            auto_labeling_text_v1("existing_annotation_count"),
            QtWidgets.QLabel(str(self.summary.get("existing_annotation", 0))),
        )
        form.addRow(
            auto_labeling_text_v1("host_prepare_failure"),
            QtWidgets.QLabel(str(self.summary.get("host_prepare_failed", 0))),
        )
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox()
        self.start_button = buttons.addButton(
            "",
            QtWidgets.QDialogButtonBox.AcceptRole,
        )
        self.start_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.delay_spin.valueChanged.connect(self._sync_delay_mode)
        self._sync_delay_mode(self.delay_spin.value())

    def _sync_delay_mode(self, delay):
        if float(delay) == 0.0:
            self.title_label.setText(auto_labeling_text_v1("fast_mode"))
            self.note_label.setText(auto_labeling_text_v1("fast_description"))
            self.start_button.setText(auto_labeling_text_v1("start_fast"))
        else:
            self.title_label.setText(auto_labeling_text_v1("visible_mode"))
            self.note_label.setText(
                auto_labeling_text_v1("visible_description")
            )
            self.start_button.setText(auto_labeling_text_v1("start_visible"))

    def selected_values(self):
        return {
            "delay_seconds": self.delay_spin.value(),
            "range": self.range_combo.currentData(),
            "filter": self.filter_combo.currentData(),
            "write_policy": self.write_policy_combo.currentData(),
        }


class FastRunSetupDialog(ContinuousRunSetupDialog):
    """Phase 4 compatibility dialog with a locked zero-second delay."""

    def __init__(self, summary, model_summary, **kwargs):
        values = dict(kwargs.pop("initial_values", {}) or {})
        values["delay_seconds"] = 0.0
        super().__init__(
            summary,
            model_summary,
            initial_values=values,
            **kwargs,
        )
        self.delay_spin.setEnabled(False)

    def selected_values(self):
        values = super().selected_values()
        values.pop("delay_seconds")
        return values


class FastRunProgressDialog(QtWidgets.QDialog):
    pause_requested = QtCore.pyqtSignal()
    resume_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    retry_requested = QtCore.pyqtSignal()
    skip_requested = QtCore.pyqtSignal()
    review_now_requested = QtCore.pyqtSignal()
    review_later_requested = QtCore.pyqtSignal()

    _METRICS = (
        ("workset_total", "workset_total"),
        ("selected_by_range", "selected_by_range"),
        ("eligible_for_inference", "eligible_for_inference"),
        ("succeeded", "succeeded"),
        ("zero_target", "zero_target"),
        ("skipped_existing", "skipped_existing"),
        ("skipped_outside_range", "skipped_outside_range"),
        ("host_prepare_failed", "host_prepare_failed"),
        ("failed_input", "failed_input"),
        ("model_failed_unresolved", "model_failed_unresolved"),
        ("explicit_error_skips", "explicit_error_skips"),
        ("conflicts", "conflicts"),
        ("pending_review", "pending_review"),
        ("remaining", "remaining"),
        ("processing_status", "processing_status"),
        ("completed_with_errors", "completed_with_errors"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._terminal = False
        self._paused = False
        self._pause_pending = False
        self._phase = "PREPARING"
        self._positioned = False
        self.setWindowTitle(auto_labeling_text_v1("progress_title"))
        self.setMinimumWidth(520)
        self.setWindowModality(QtCore.Qt.NonModal)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        self.state_label = QtWidgets.QLabel(auto_labeling_text_v1("preparing"))
        self.state_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.state_label)
        self.file_label = QtWidgets.QLabel("")
        self.file_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        layout.addWidget(self.file_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 0")
        layout.addWidget(self.progress_bar)

        timing = QtWidgets.QFormLayout()
        suffix = auto_labeling_text_v1("seconds_suffix")
        self.configured_delay_label = QtWidgets.QLabel(f"0.0{suffix}")
        self.remaining_delay_label = QtWidgets.QLabel(f"0.0{suffix}")
        timing.addRow(
            auto_labeling_text_v1("configured_dwell_time"),
            self.configured_delay_label,
        )
        timing.addRow(
            auto_labeling_text_v1("remaining_dwell_time"),
            self.remaining_delay_label,
        )
        layout.addLayout(timing)

        metrics = QtWidgets.QGridLayout()
        self.metric_labels = {}
        for index, (field, text_key) in enumerate(self._METRICS):
            name = QtWidgets.QLabel(auto_labeling_text_v1(text_key))
            value = QtWidgets.QLabel("0")
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            metrics.addWidget(name, index // 2, (index % 2) * 2)
            metrics.addWidget(value, index // 2, (index % 2) * 2 + 1)
            self.metric_labels[field] = value
        layout.addLayout(metrics)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        controls = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("pause")
        )
        self.pause_button.setToolTip(auto_labeling_text_v1("pause_tooltip"))
        self.pause_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause)
        )
        self.stop_button = QtWidgets.QPushButton(auto_labeling_text_v1("stop"))
        self.stop_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop)
        )
        self.retry_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("retry_current")
        )
        self.retry_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.skip_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("skip_current")
        )
        self.review_now_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("review_now")
        )
        self.review_later_button = QtWidgets.QPushButton(
            auto_labeling_text_v1("review_later")
        )
        self.retry_button.hide()
        self.skip_button.hide()
        self.review_now_button.hide()
        self.review_later_button.hide()
        controls.addWidget(self.pause_button)
        controls.addWidget(self.retry_button)
        controls.addWidget(self.skip_button)
        controls.addWidget(self.review_later_button)
        controls.addWidget(self.review_now_button)
        controls.addStretch(1)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self.stop_requested)
        self.retry_button.clicked.connect(self.retry_requested)
        self.skip_button.clicked.connect(self.skip_requested)
        self.review_now_button.clicked.connect(self.review_now_requested)
        self.review_later_button.clicked.connect(self.review_later_requested)

    def _toggle_pause(self):
        if self._paused:
            self.resume_requested.emit()
        elif not self._pause_pending:
            self._pause_pending = True
            self.pause_button.setEnabled(False)
            self.state_label.setText(auto_labeling_text_v1("phase_pausing"))
            self.pause_requested.emit()

    def set_phase(self, phase):
        self._phase = phase
        messages = {
            "PREPARING": auto_labeling_text_v1("preparing"),
            "LOADING": auto_labeling_text_v1("phase_loading"),
            "INFERENCING": auto_labeling_text_v1("phase_inferencing"),
            "COMMITTING": auto_labeling_text_v1("phase_committing"),
            "PRESENTING": auto_labeling_text_v1("phase_presenting"),
            "WAITING_ERROR": auto_labeling_text_v1("phase_waiting_error"),
            "PAUSED": auto_labeling_text_v1("phase_paused"),
            "FINISHED": auto_labeling_text_v1("phase_finished"),
        }
        self._paused = phase == "PAUSED"
        if self._paused or phase == "FINISHED":
            self._pause_pending = False
        if self._pause_pending:
            self.state_label.setText(auto_labeling_text_v1("phase_pausing"))
        else:
            self.state_label.setText(messages.get(phase, phase))
        self.pause_button.setText(
            auto_labeling_text_v1("continue")
            if self._paused
            else auto_labeling_text_v1("pause")
        )
        self.pause_button.setEnabled(
            phase != "FINISHED" and not self._pause_pending
        )
        self.pause_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.SP_MediaPlay
                if self._paused
                else QtWidgets.QStyle.SP_MediaPause
            )
        )

    def cancel_pause_request(self):
        if not self._pause_pending:
            return
        self._pause_pending = False
        self.set_phase(self._phase)

    @staticmethod
    def _edge_position(owner_rect, dialog_size, available_rect, margin=16):
        width = dialog_size.width()
        height = dialog_size.height()
        outside_x = owner_rect.right() + margin + 1
        if outside_x + width - 1 <= available_rect.right() - margin:
            x = outside_x
        else:
            x = owner_rect.right() - width - margin + 1
        y = owner_rect.top() + margin
        max_x = available_rect.right() - width - margin + 1
        max_y = available_rect.bottom() - height - margin + 1
        x = max(available_rect.left() + margin, min(x, max_x))
        y = max(available_rect.top() + margin, min(y, max_y))
        return QtCore.QPoint(x, y)

    def _position_near_parent_edge(self):
        parent = self.parentWidget()
        if parent is None:
            return
        owner = parent.window()
        owner_rect = owner.frameGeometry()
        screen = QtWidgets.QApplication.screenAt(owner_rect.center())
        if screen is None:
            available_rect = (
                QtWidgets.QApplication.desktop().availableGeometry(owner)
            )
        else:
            available_rect = screen.availableGeometry()
        position = self._edge_position(
            owner_rect,
            self.frameGeometry().size(),
            available_rect,
        )
        self.move(position)

    def showEvent(self, event):
        super().showEvent(event)
        if self._positioned:
            return
        self._positioned = True
        QtCore.QTimer.singleShot(0, self._position_near_parent_edge)

    def update_progress(self, progress):
        total = int(progress.get("total", 0))
        processed = int(progress.get("processed", 0))
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"{processed} / {total}")
        self.file_label.setText(str(progress.get("current_filename", "")))
        delay = float(progress.get("delay_seconds", 0.0) or 0.0)
        remaining = float(progress.get("remaining_seconds", 0.0) or 0.0)
        suffix = auto_labeling_text_v1("seconds_suffix")
        self.configured_delay_label.setText(f"{delay:.1f}{suffix}")
        self.remaining_delay_label.setText(f"{remaining:.1f}{suffix}")
        if self._paused and delay > 0.0:
            self.state_label.setText(
                auto_labeling_text_v1(
                    "paused_remaining",
                    remaining=remaining,
                )
            )
        for field, label in self.metric_labels.items():
            value = progress.get(field, 0)
            if field == "processing_status":
                value = auto_labeling_status_text_v1(value, "processing")
            elif field == "completed_with_errors":
                value = auto_labeling_boolean_text_v1(value)
            label.setText(str(value))

    def show_waiting_error(self, error):
        code = str(error.get("error_code") or "model_prediction_failed")
        message = str(error.get("error_message") or "")
        category = auto_labeling_error_category_v1(code)
        detail = f"{code}\n{message}" if message else code
        self.error_label.setText(f"{category}: {detail}")
        self.error_label.show()
        self.retry_button.show()
        self.skip_button.show()
        self.pause_button.setEnabled(not self._pause_pending)

    def clear_waiting_error(self):
        self.error_label.clear()
        self.error_label.hide()
        self.retry_button.hide()
        self.skip_button.hide()
        self.pause_button.setEnabled(not self._pause_pending)

    def finish_run(self, summary):
        self._terminal = True
        self.set_phase("FINISHED")
        total = int(summary.get("workset_total", 0))
        remaining = int(summary.get("remaining", 0))
        processed = total - remaining
        self.update_progress(
            dict(
                summary, total=total, processed=processed, current_filename=""
            )
        )
        status = summary.get("processing_status", "FAILED")
        if (
            status == "COMPLETED"
            and not summary.get("eligible_for_inference")
            and not summary.get("completed_with_errors")
        ):
            self.state_label.setText(
                auto_labeling_text_v1("nothing_to_process")
            )
        else:
            self.state_label.setText(
                auto_labeling_text_v1(
                    "run_result",
                    status=auto_labeling_status_text_v1(
                        status,
                        "processing",
                    ),
                )
            )
        self.pause_button.hide()
        self.retry_button.hide()
        self.skip_button.hide()
        self.stop_button.setText(auto_labeling_text_v1("close"))
        try:
            self.stop_button.clicked.disconnect()
        except TypeError:
            pass
        self.stop_button.clicked.connect(self.accept)
        if int(summary.get("pending_review", 0)) > 0:
            self.review_now_button.show()
            self.review_later_button.show()

    def closeEvent(self, event):
        if self._terminal:
            super().closeEvent(event)
            return
        self.stop_requested.emit()
        event.ignore()


class FastRunUiSession(QtCore.QObject):
    """Own one unified Fast/Visible lifecycle for a labeling widget."""

    setup_dialog_class = ContinuousRunSetupDialog
    progress_dialog_class = FastRunProgressDialog
    runner_class = PredictionRunner
    controller_class = ContinuousAutoLabelingController

    def __init__(self, labeling_widget, auto_widget, initial_delay=0.0):
        super().__init__(auto_widget)
        self.labeling_widget = labeling_widget
        self.auto_widget = auto_widget
        self.initial_delay = initial_delay
        self.runner = None
        self.controller = None
        self.progress = None
        self.guard = None
        self.presenter = None
        self._thread_cleanup_connected = False
        self._pending_review_action = None
        self._review_choice_finalized = False
        self._audit_client = None

    def begin(self):
        try:
            prepared = self._prepare_options()
            if prepared is None:
                self._cleanup()
                return False
            options, image_paths = prepared
            self.runner = self.runner_class(self.auto_widget.model_manager)
            if options.execution_mode == "VISIBLE":
                self.presenter = LabelingWidgetSequencePresenter(
                    self.labeling_widget
                )
            self.controller = self.controller_class(
                self.runner,
                options,
                host_context=self.auto_widget.auto_labeling_host_context,
                standalone_image_paths=image_paths,
                standalone_output_dir=self.labeling_widget.output_dir,
                context_replacer=self._replace_context,
                pose_config=getattr(self.labeling_widget, "pose_config", None),
                presentation_adapter=self.presenter,
            )
            self.progress = self.progress_dialog_class(self.labeling_widget)
            guard_class = (
                FastCanvasStateGuard
                if options.execution_mode == "FAST"
                else VisibleCanvasStateGuard
            )
            self.guard = guard_class(self.labeling_widget)
            self._connect_controls()
            self.auto_widget.prediction_runner = self.controller
            self.guard.set_running(True)
            self.progress.show()
            if not self.controller.start():
                raise FastControllerError("fast_controller_start_rejected")
            self._connect_thread_cleanup()
            return True
        except Exception as exc:  # noqa: B902
            self._handle_start_failure(exc)
            return False

    def _prepare_options(self):
        model_config = self.auto_widget.model_manager.loaded_model_config
        if type(model_config) is not dict or model_config.get("model") is None:
            raise FastControllerError("prediction_model_not_loaded")
        capability = resolve_sequence_capabilities(model_config)
        if not (
            capability.supports_fast_sequence
            or capability.supports_visible_sequence
        ):
            raise FastControllerError("sequence_not_supported")
        lease = self.auto_widget.model_manager.inference_lease
        if bool(getattr(lease, "is_active", False)):
            raise FastControllerError("inference_lease_busy")
        canvas = getattr(self.labeling_widget, "canvas", None)
        if getattr(canvas, "current", None) is not None:
            raise FastControllerError("unfinished_shape_blocks_fast_run")
        if (
            self._resolve_dirty(auto_labeling_text_v1("start_continuous"))
            is None
        ):
            return None

        context = self.auto_widget.auto_labeling_host_context
        image_paths = tuple(self.labeling_widget.image_list)
        fingerprint = build_model_fingerprint_v1(model_config)
        parameters = build_parameter_snapshot_v1(
            model_config, self.auto_widget
        )
        if context is not None:
            context = validate_auto_labeling_host_context(context)
            if not context.images_ready:
                raise FastControllerError("images_not_ready")
            workset_source = "SESSION_WORKSET"
        else:
            if not image_paths:
                raise FastControllerError("standalone_workset_empty")
            workset_source = "CURRENT_FILE_LIST_SNAPSHOT"

        anchor_id = self._current_anchor_id(context)
        preview = self._preview_summary(
            context, image_paths, self.labeling_widget.output_dir
        )
        model_name = parameters.get("model_name") or parameters["model_type"]
        conf = parameters.get("confidence_threshold")
        iou = parameters.get("iou_threshold")
        model_summary = f"{model_name} · conf={conf} · IOU={iou}"
        settings = continuous_auto_labeling_settings_v1(
            self.labeling_widget._config
        )
        if self.initial_delay is not None:
            try:
                settings["delay_seconds"] = validate_sequence_delay_seconds_v1(
                    self.initial_delay
                )
            except Exception:
                settings["delay_seconds"] = 2.0
        dialog = self.setup_dialog_class(
            preview,
            model_summary,
            current_anchor_available=anchor_id is not None,
            initial_values=settings,
            parent=self.labeling_widget,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        selected = dialog.selected_values()
        if (
            selected["write_policy"] == "FORCE_REPLACE"
            and preview["existing_annotation"] > 0
        ):
            answer = QtWidgets.QMessageBox.warning(
                self.labeling_widget,
                auto_labeling_text_v1("replace_confirmation_title"),
                auto_labeling_text_v1(
                    "replace_confirmation",
                    count=preview["existing_annotation"],
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return None
        delay = validate_sequence_delay_seconds_v1(selected["delay_seconds"])
        if delay == 0.0 and not capability.supports_fast_sequence:
            raise FastControllerError("fast_sequence_not_supported")
        if delay > 0.0 and not capability.supports_visible_sequence:
            raise FastControllerError("visible_sequence_not_supported")
        options = SequenceRunOptionsV1(
            delay_seconds=delay,
            range=selected["range"],
            filter=selected["filter"],
            write_policy=selected["write_policy"],
            current_anchor_image_id=(
                anchor_id if selected["range"] == "CURRENT_TO_END" else None
            ),
            workset_source=workset_source,
            model_fingerprint=fingerprint,
            parameter_snapshot=parameters,
        )
        self.labeling_widget._config["continuous_auto_labeling"] = {
            "delay_seconds": delay,
            "range": selected["range"],
            "filter": selected["filter"],
        }
        return options, image_paths

    def _current_anchor_id(self, context):
        filename = getattr(self.labeling_widget, "filename", None)
        if not filename:
            return None
        if context is None:
            canonical = _canonical(filename)
            matches = [
                path
                for path in self.labeling_widget.image_list
                if _canonical(path) == canonical
            ]
            if len(matches) != 1:
                return None
            return standalone_image_id_v1(matches[0])
        record = context.image_records_by_path.get(_canonical(filename))
        return (
            _record_value(record, "image_id") if record is not None else None
        )

    @staticmethod
    def _preview_summary(context, image_paths, output_dir):
        if context is None:
            label_paths = []
            for image_path in image_paths:
                label_name = os.path.splitext(os.path.basename(image_path))[0]
                label_paths.append(
                    os.path.join(output_dir, label_name + ".json")
                    if output_dir
                    else os.path.splitext(image_path)[0] + ".json"
                )
            total = len(image_paths)
            host_failed = 0
        else:
            records = list(context.image_records_by_path.values())
            label_paths = [
                _record_value(record, "canonical_session_label_path")
                for record in records
            ]
            private = getattr(context, "_workset_summary", {}) or {}
            total = int(private.get("workset_total", len(records)))
            host_failed = int(private.get("host_prepare_failed", 0))
        existing = sum(
            resolve_existing_label(path).presence
            in {"VALID_EMPTY", "VALID_NONEMPTY"}
            for path in label_paths
        )
        return {
            "workset_total": total,
            "existing_annotation": existing,
            "host_prepare_failed": host_failed,
        }

    def _resolve_dirty(self, title):
        if bool(
            getattr(
                self.labeling_widget,
                "auto_labeling_commit_blocked",
                False,
            )
        ):
            self.labeling_widget.annotation_commit_bridge.integrity_refresh()
            if bool(
                getattr(
                    self.labeling_widget,
                    "auto_labeling_commit_blocked",
                    False,
                )
            ):
                raise FastControllerError("manual_commit_recovery_failed")
        if not bool(getattr(self.labeling_widget, "dirty", False)):
            return "CLEAN"
        answer = QtWidgets.QMessageBox.question(
            self.labeling_widget,
            title,
            auto_labeling_text_v1("dirty_current_image"),
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return None
        if answer == QtWidgets.QMessageBox.Save:
            saved = self.labeling_widget.save_file()
            if (
                saved is False
                or bool(getattr(self.labeling_widget, "dirty", False))
                or bool(
                    getattr(
                        self.labeling_widget,
                        "auto_labeling_commit_blocked",
                        False,
                    )
                )
            ):
                raise FastControllerError("manual_save_failed")
            return "SAVED"
        filename = self.labeling_widget.filename
        self.labeling_widget.set_clean()
        if filename is not None and not self.labeling_widget.load_file(
            filename
        ):
            set_dirty = getattr(self.labeling_widget, "set_dirty", None)
            if callable(set_dirty):
                set_dirty()
            else:
                self.labeling_widget.dirty = True
            raise FastControllerError("manual_discard_reload_failed")
        return "DISCARDED"

    def _replace_context(self, context):
        self.labeling_widget.set_auto_labeling_host_context(context)

    def _connect_controls(self):
        self.controller.state_changed.connect(self._on_phase_changed)
        self.controller.progress_changed.connect(self._on_progress_changed)
        self.controller.waiting_error.connect(self.progress.show_waiting_error)
        self.controller.label_committed.connect(self._on_label_committed)
        self.controller.finished.connect(self._on_finished)
        self.progress.pause_requested.connect(self._pause)
        self.progress.resume_requested.connect(self._resume)
        self.progress.stop_requested.connect(self._stop)
        self.progress.retry_requested.connect(self._retry)
        self.progress.skip_requested.connect(self._skip)
        self.progress.review_now_requested.connect(self._review_now)
        self.progress.review_later_requested.connect(self._review_later)

    def _on_progress_changed(self, summary):
        self.progress.update_progress(summary)
        if type(summary) is not dict:
            return
        pending_review = summary.get("pending_review")
        if type(pending_review) is not int or pending_review < 0:
            return
        pending_setter = getattr(
            self.labeling_widget,
            "set_auto_labeling_pending_review_count",
            None,
        )
        if callable(pending_setter):
            pending_setter(pending_review, scope="session")

    def _on_label_committed(self, event):
        """Reflect a verified label commit in the file list only."""
        try:
            if type(event) is not dict or self.controller is None:
                return
            if event.get("run_id") != getattr(self.controller, "run_id", None):
                return
            image_id = event.get("image_id")
            attempt_id = event.get("attempt_id")
            image_path = event.get("canonical_image_path")
            if (
                type(image_id) is not str
                or not image_id
                or type(attempt_id) is not str
                or not attempt_id
                or type(image_path) is not str
                or _canonical(image_path) != image_path
            ):
                return
            validate_document_digest_v1(event.get("staged_document_digest"))
            records = getattr(self.controller, "records_by_id", {})
            record = records.get(image_id)
            if (
                record is None
                or _record_value(record, "image_id") != image_id
                or _record_value(record, "canonical_session_image_path")
                != image_path
            ):
                return
            matches = [
                (index, path)
                for index, path in enumerate(self.labeling_widget.image_list)
                if _canonical(path) == image_path
            ]
            if len(matches) != 1:
                return
            index, listed_path = matches[0]
            if self.labeling_widget.fn_to_index.get(str(listed_path)) != index:
                return
            file_list = self.labeling_widget.file_list_widget
            item = file_list.item(index)
            if item is None or _canonical(item.text()) != image_path:
                return
            item.setCheckState(QtCore.Qt.Checked)
        except Exception:
            # The label transaction is already complete; UI drift is non-fatal.
            return

    def _on_phase_changed(self, phase):
        self.progress.set_phase(phase)
        if phase == "PAUSED":
            self.guard.set_paused(True)
        elif phase in {
            "LOADING",
            "INFERENCING",
            "COMMITTING",
            "PRESENTING",
        }:
            self.guard.set_running(True)

    def _pause(self):
        if not self.controller.request_pause():
            self.progress.cancel_pause_request()

    def _resume(self):
        try:
            resolution = self._resolve_dirty(
                auto_labeling_text_v1("continue_continuous")
            )
        except Exception as exc:  # noqa: B902
            self._show_control_failure(exc)
            return
        if resolution is None:
            return
        self.progress.clear_waiting_error()
        self.controller.resume(resolution)

    def _retry(self):
        if self.controller.retry_current():
            self.progress.clear_waiting_error()

    def _skip(self):
        if self.controller.skip_current():
            self.progress.clear_waiting_error()

    def _stop(self):
        if self.controller.phase == "PAUSED":
            try:
                resolution = self._resolve_dirty(
                    auto_labeling_text_v1("stop_continuous")
                )
            except Exception as exc:  # noqa: B902
                self._show_control_failure(exc)
                return
            if resolution is None:
                return
        self.controller.request_stop()

    def _show_control_failure(self, exc):
        code = getattr(exc, "code", "manual_save_failed")
        detail = str(exc)
        QtWidgets.QMessageBox.warning(
            self.labeling_widget,
            auto_labeling_text_v1("cannot_continue"),
            f"{code}\n{detail}" if detail != code else code,
        )

    def _on_finished(self, summary):
        if self.controller.control_intent == "CLOSE":
            self.guard.release_for_close()
        elif isinstance(self.guard, VisibleCanvasStateGuard):
            self.guard.finish(bool(self.presenter.presented_image_ids))
        else:
            self.guard.restore(self.controller.modified_image_ids)
        self._reset_entry_action()
        self._prepare_audit_client(summary)
        pending_setter = getattr(
            self.labeling_widget,
            "set_auto_labeling_pending_review_count",
            None,
        )
        if callable(pending_setter):
            pending_setter(summary.get("pending_review", 0), scope="session")
        self.progress.finish_run(summary)
        self.auto_widget.refresh_continuous_run_availability()
        self._connect_thread_cleanup()
        if not self.controller.requires_safe_shutdown():
            self._cleanup()

    def _prepare_audit_client(self, summary):
        if int(summary.get("pending_review", 0)) <= 0:
            return None
        context = getattr(self.controller, "host_context", None)
        if context is not None:
            self._audit_client = getattr(context, "staged_audit_client", None)
            return self._audit_client
        self._audit_client = InMemoryStagedAuditClientV1(
            run_store=self.controller.run_store,
            run_id=self.controller.run_id,
            records_by_id=self.controller.records_by_id,
            commit_store=self.controller.commit_store,
        )
        self.labeling_widget._standalone_auto_labeling_audit_client = (
            self._audit_client
        )
        return self._audit_client

    def _review_now(self):
        if self._audit_client is None or self._review_choice_finalized:
            return False
        self._review_choice_finalized = True
        self._pending_review_action = "IMMEDIATE"
        self.progress.accept()
        thread = self.runner.worker_thread if self.runner is not None else None
        if thread is None or not thread.isRunning():
            return self._start_pending_review()
        return True

    def _review_later(self):
        if self._review_choice_finalized:
            return False
        self._review_choice_finalized = True
        self._pending_review_action = None
        self.progress.accept()
        return True

    def _start_pending_review(self):
        if self._pending_review_action != "IMMEDIATE":
            return False
        self._pending_review_action = None
        return self.labeling_widget.start_auto_labeling_review(
            self._audit_client
        )

    def _connect_thread_cleanup(self):
        thread = self.runner.worker_thread if self.runner is not None else None
        if thread is None or self._thread_cleanup_connected:
            return
        thread.finished.connect(self._cleanup)
        self._thread_cleanup_connected = True

    def _handle_start_failure(self, exc):
        if self.guard is not None:
            self.guard.restore(
                getattr(self.controller, "modified_image_ids", set())
            )
        if self.progress is not None:
            self.progress._terminal = True
            self.progress.close()
        if self.runner is not None:
            self.runner.shutdown_when_idle()
            self._connect_thread_cleanup()
        QtWidgets.QMessageBox.warning(
            self.labeling_widget,
            auto_labeling_text_v1("cannot_start"),
            continuous_start_failure_text_v1(exc),
        )
        if self.runner is None or not self.runner.requires_safe_shutdown():
            self._cleanup()

    def _reset_entry_action(self):
        actions = getattr(self.labeling_widget, "actions", None)
        action = getattr(actions, "run_all_images", None)
        if action is not None and action.isCheckable():
            action.setChecked(False)

    @QtCore.pyqtSlot()
    def _cleanup(self):
        self._reset_entry_action()
        if (
            getattr(self.auto_widget, "prediction_runner", None)
            is self.controller
        ):
            self.auto_widget.prediction_runner = None
        if getattr(self.auto_widget, "_fast_run_session", None) is self:
            self.auto_widget._fast_run_session = None
        self.auto_widget.refresh_continuous_run_availability()
        self._start_pending_review()


ContinuousRunProgressDialog = FastRunProgressDialog
ContinuousRunUiSession = FastRunUiSession
