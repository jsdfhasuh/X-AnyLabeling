"""Phase 4 Fast setup, progress, and UI lifecycle coordination."""

import os

from PyQt5 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    resolve_existing_label,
)
from anylabeling.views.labeling.utils.auto_labeling_host import (
    validate_auto_labeling_host_context,
)
from anylabeling.views.labeling.utils.auto_labeling_run_store import (
    build_annotation_commit_event_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    FastRunOptionsV1,
    build_model_fingerprint_v1,
    build_parameter_snapshot_v1,
    resolve_sequence_capabilities,
)
from anylabeling.views.labeling.utils.continuous_auto_labeling import (
    FastAutoLabelingController,
    FastCanvasStateGuard,
    FastControllerError,
    standalone_image_id_v1,
)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _record_value(record, field):
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


class FastRunSetupDialog(QtWidgets.QDialog):
    """Collect only the zero-delay options authorized for Phase 4."""

    def __init__(
        self,
        summary,
        model_summary,
        *,
        current_anchor_available,
        parent=None,
    ):
        super().__init__(parent)
        self.summary = dict(summary)
        self.setWindowTitle(self.tr("连续自动标注"))
        self.setMinimumWidth(560)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QtWidgets.QLabel(self.tr("快速批处理"))
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        note = QtWidgets.QLabel(
            self.tr("快速批处理会在后台逐张推理并保存，主画布不会逐张切换。")
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)

        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.addItem(self.tr("全部图片"), "ALL_IMAGES")
        self.range_combo.addItem(self.tr("当前到末尾"), "CURRENT_TO_END")
        if current_anchor_available:
            self.range_combo.setCurrentIndex(1)
        else:
            model = self.range_combo.model()
            model.item(1).setEnabled(False)
        form.addRow(self.tr("基础范围"), self.range_combo)

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItem(self.tr("全部"), "ALL")
        self.filter_combo.addItem(
            self.tr("仅无有效标注"),
            "ONLY_WITHOUT_VALID_ANNOTATION",
        )
        form.addRow(self.tr("过滤"), self.filter_combo)

        self.write_policy_combo = QtWidgets.QComboBox()
        for text, value in (
            (self.tr("继承模型"), "INHERIT_MODEL_POLICY"),
            (self.tr("跳过已有"), "SKIP_EXISTING"),
            (self.tr("强制替换"), "FORCE_REPLACE"),
            (self.tr("强制合并"), "FORCE_MERGE"),
        ):
            self.write_policy_combo.addItem(text, value)
        form.addRow(self.tr("写入策略"), self.write_policy_combo)

        delay = QtWidgets.QLabel(self.tr("0.0 秒（快速批处理，不逐张显示）"))
        form.addRow(self.tr("停留时间"), delay)
        form.addRow(self.tr("模型与参数"), QtWidgets.QLabel(model_summary))
        form.addRow(
            self.tr("实际工作集数量"),
            QtWidgets.QLabel(str(self.summary.get("workset_total", 0))),
        )
        form.addRow(
            self.tr("已有标注数量"),
            QtWidgets.QLabel(str(self.summary.get("existing_annotation", 0))),
        )
        form.addRow(
            self.tr("Host prepare failure"),
            QtWidgets.QLabel(str(self.summary.get("host_prepare_failed", 0))),
        )
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox()
        self.start_button = buttons.addButton(
            self.tr("开始快速标注"),
            QtWidgets.QDialogButtonBox.AcceptRole,
        )
        self.start_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_values(self):
        return {
            "range": self.range_combo.currentData(),
            "filter": self.filter_combo.currentData(),
            "write_policy": self.write_policy_combo.currentData(),
        }


class FastRunProgressDialog(QtWidgets.QDialog):
    pause_requested = QtCore.pyqtSignal()
    resume_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    retry_requested = QtCore.pyqtSignal()
    skip_requested = QtCore.pyqtSignal()

    _METRICS = (
        ("succeeded", "succeeded"),
        ("zero_target", "zero target"),
        ("skipped_existing", "skipped existing"),
        ("failed_input", "failed input"),
        ("explicit_error_skips", "explicit error skips"),
        ("conflicts", "conflicts"),
        ("remaining", "remaining"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._terminal = False
        self._paused = False
        self.setWindowTitle(self.tr("连续自动标注进度"))
        self.setMinimumWidth(520)
        self.setWindowModality(QtCore.Qt.NonModal)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        self.state_label = QtWidgets.QLabel(self.tr("正在准备"))
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

        metrics = QtWidgets.QGridLayout()
        self.metric_labels = {}
        for index, (field, text) in enumerate(self._METRICS):
            name = QtWidgets.QLabel(self.tr(text))
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
        self.pause_button = QtWidgets.QPushButton(self.tr("暂停"))
        self.pause_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause)
        )
        self.stop_button = QtWidgets.QPushButton(self.tr("结束"))
        self.stop_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop)
        )
        self.retry_button = QtWidgets.QPushButton(self.tr("重试当前图片"))
        self.retry_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.skip_button = QtWidgets.QPushButton(self.tr("跳过当前图片"))
        self.retry_button.hide()
        self.skip_button.hide()
        controls.addWidget(self.pause_button)
        controls.addWidget(self.retry_button)
        controls.addWidget(self.skip_button)
        controls.addStretch(1)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self.stop_requested)
        self.retry_button.clicked.connect(self.retry_requested)
        self.skip_button.clicked.connect(self.skip_requested)

    def _toggle_pause(self):
        if self._paused:
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def set_phase(self, phase):
        messages = {
            "LOADING": self.tr("正在准备运行"),
            "INFERENCING": self.tr("正在后台推理"),
            "COMMITTING": self.tr("正在安全保存"),
            "WAITING_ERROR": self.tr("等待错误处理"),
            "PAUSED": self.tr("已暂停"),
            "FINISHED": self.tr("运行已结束"),
        }
        self.state_label.setText(messages.get(phase, phase))
        self._paused = phase == "PAUSED"
        self.pause_button.setText(
            self.tr("继续") if self._paused else self.tr("暂停")
        )
        self.pause_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.SP_MediaPlay
                if self._paused
                else QtWidgets.QStyle.SP_MediaPause
            )
        )

    def update_progress(self, progress):
        total = int(progress.get("total", 0))
        processed = int(progress.get("processed", 0))
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"{processed} / {total}")
        self.file_label.setText(str(progress.get("current_filename", "")))
        for field, label in self.metric_labels.items():
            label.setText(str(progress.get(field, 0)))

    def show_waiting_error(self, error):
        code = str(error.get("error_code") or "model_prediction_failed")
        message = str(error.get("error_message") or "")
        self.error_label.setText(f"{code}\n{message}" if message else code)
        self.error_label.show()
        self.retry_button.show()
        self.skip_button.show()
        self.pause_button.setEnabled(False)

    def clear_waiting_error(self):
        self.error_label.clear()
        self.error_label.hide()
        self.retry_button.hide()
        self.skip_button.hide()
        self.pause_button.setEnabled(True)

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
            self.state_label.setText(self.tr("无需处理"))
        else:
            self.state_label.setText(self.tr("运行结果：") + str(status))
        self.pause_button.hide()
        self.retry_button.hide()
        self.skip_button.hide()
        self.stop_button.setText(self.tr("关闭"))
        try:
            self.stop_button.clicked.disconnect()
        except TypeError:
            pass
        self.stop_button.clicked.connect(self.accept)

    def closeEvent(self, event):
        if self._terminal:
            super().closeEvent(event)
            return
        self.stop_requested.emit()
        event.ignore()


class FastRunUiSession(QtCore.QObject):
    """Own one setup/progress/controller lifecycle for a labeling widget."""

    setup_dialog_class = FastRunSetupDialog
    progress_dialog_class = FastRunProgressDialog
    runner_class = PredictionRunner
    controller_class = FastAutoLabelingController

    def __init__(self, labeling_widget, auto_widget):
        super().__init__(auto_widget)
        self.labeling_widget = labeling_widget
        self.auto_widget = auto_widget
        self.runner = None
        self.controller = None
        self.progress = None
        self.guard = None
        self._thread_cleanup_connected = False

    def begin(self):
        try:
            prepared = self._prepare_options()
            if prepared is None:
                self._cleanup()
                return False
            options, image_paths = prepared
            self.runner = self.runner_class(self.auto_widget.model_manager)
            self.controller = self.controller_class(
                self.runner,
                options,
                host_context=self.auto_widget.auto_labeling_host_context,
                standalone_image_paths=image_paths,
                standalone_output_dir=self.labeling_widget.output_dir,
                context_replacer=self._replace_context,
                pose_config=getattr(self.labeling_widget, "pose_config", None),
            )
            self.progress = self.progress_dialog_class(self.labeling_widget)
            self.guard = FastCanvasStateGuard(self.labeling_widget)
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
        if not capability.supports_fast_sequence:
            raise FastControllerError("fast_sequence_not_supported")
        lease = self.auto_widget.model_manager.inference_lease
        if bool(getattr(lease, "is_active", False)):
            raise FastControllerError("inference_lease_busy")
        canvas = getattr(self.labeling_widget, "canvas", None)
        if getattr(canvas, "current", None) is not None:
            raise FastControllerError("unfinished_shape_blocks_fast_run")
        if self._resolve_dirty(self.tr("开始快速标注")) is None:
            return None

        context = self.auto_widget.auto_labeling_host_context
        image_paths = tuple(self.labeling_widget.image_list)
        if context is not None:
            context = validate_auto_labeling_host_context(context)
            if not context.images_ready:
                raise FastControllerError("images_not_ready")
            if context.active_run_id is not None:
                raise FastControllerError("active_run_exists")
            workset_source = "SESSION_WORKSET"
        else:
            if not image_paths:
                raise FastControllerError("standalone_workset_empty")
            workset_source = "CURRENT_FILE_LIST_SNAPSHOT"

        anchor_id = self._current_anchor_id(context)
        fingerprint = build_model_fingerprint_v1(model_config)
        parameters = build_parameter_snapshot_v1(
            model_config, self.auto_widget
        )
        preview = self._preview_summary(
            context, image_paths, self.labeling_widget.output_dir
        )
        model_name = parameters.get("model_name") or parameters["model_type"]
        conf = parameters.get("confidence_threshold")
        iou = parameters.get("iou_threshold")
        model_summary = f"{model_name} · conf={conf} · IOU={iou}"
        dialog = self.setup_dialog_class(
            preview,
            model_summary,
            current_anchor_available=anchor_id is not None,
            parent=self.labeling_widget,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        selected = dialog.selected_values()
        options = FastRunOptionsV1(
            delay_seconds=0.0,
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
        if not bool(getattr(self.labeling_widget, "dirty", False)):
            return "CLEAN"
        answer = QtWidgets.QMessageBox.question(
            self.labeling_widget,
            title,
            self.tr("当前图片有未保存修改。"),
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return None
        if answer == QtWidgets.QMessageBox.Save:
            self.labeling_widget.save_file()
            if bool(getattr(self.labeling_widget, "dirty", False)):
                raise FastControllerError("manual_save_failed")
            try:
                FastRunUiSession._publish_manual_save_if_bound(self)
            except Exception:
                set_dirty = getattr(self.labeling_widget, "set_dirty", None)
                if callable(set_dirty):
                    set_dirty()
                else:
                    self.labeling_widget.dirty = True
                raise
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

    def _publish_manual_save_if_bound(self):
        context = getattr(
            self.auto_widget,
            "auto_labeling_host_context",
            None,
        )
        if context is None or context.active_run_id is None:
            return None
        filename = getattr(self.labeling_widget, "filename", None)
        if not filename:
            raise FastControllerError("manual_save_image_identity_missing")
        record = context.image_records_by_path.get(_canonical(filename))
        if record is None:
            raise FastControllerError("manual_save_image_identity_missing")
        image_id = _record_value(record, "image_id")
        label_path = _record_value(record, "canonical_session_label_path")
        label_file = getattr(self.labeling_widget, "label_file", None)
        saved_path = getattr(label_file, "filename", None)
        if not saved_path or _canonical(saved_path) != label_path:
            raise FastControllerError("manual_save_non_authoritative_path")
        current = resolve_existing_label(label_path)
        if current.presence not in {"VALID_EMPTY", "VALID_NONEMPTY"}:
            raise FastControllerError("manual_save_document_unavailable")
        item = context.run_store.read_item(context.active_run_id, image_id)
        event = build_annotation_commit_event_v1(
            project_id=context.project_id,
            session_id=context.active_session_id,
            run_id=context.active_run_id,
            image_id=image_id,
            attempt_id=item.get("latest_attempt_id"),
            writer_kind="MANUAL_SAVE",
            commit_scope="STAGED",
            mutation_mode="APPLY_MANUAL_REVISION",
            document_digest=current.document_digest,
            semantic_digest=current.semantic_digest,
            source_image_digest=_record_value(
                record,
                "source_image_digest",
            ),
            base_item_revision=item["item_revision"],
        )
        sink = context.annotation_commit_sink
        if not callable(getattr(sink, "publish", None)):
            raise FastControllerError("manual_commit_sink_unavailable")
        return sink.publish(event, label_path=label_path)

    def _replace_context(self, context):
        self.labeling_widget.set_auto_labeling_host_context(context)

    def _connect_controls(self):
        self.controller.state_changed.connect(self._on_phase_changed)
        self.controller.progress_changed.connect(self.progress.update_progress)
        self.controller.waiting_error.connect(self.progress.show_waiting_error)
        self.controller.finished.connect(self._on_finished)
        self.progress.pause_requested.connect(self.controller.request_pause)
        self.progress.resume_requested.connect(self._resume)
        self.progress.stop_requested.connect(self._stop)
        self.progress.retry_requested.connect(self._retry)
        self.progress.skip_requested.connect(self._skip)

    def _on_phase_changed(self, phase):
        self.progress.set_phase(phase)
        if phase == "PAUSED":
            self.guard.set_paused(True)
        elif phase in {"LOADING", "INFERENCING", "COMMITTING"}:
            self.guard.set_running(True)

    def _resume(self):
        try:
            resolution = self._resolve_dirty(self.tr("继续快速标注"))
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
                resolution = self._resolve_dirty(self.tr("结束快速标注"))
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
            self.tr("无法继续快速标注"),
            f"{code}\n{detail}" if detail != code else code,
        )

    def _on_finished(self, summary):
        if self.controller.control_intent == "CLOSE":
            self.guard.release_for_close()
        else:
            self.guard.restore(self.controller.modified_image_ids)
        self._reset_entry_action()
        self.progress.finish_run(summary)
        self.auto_widget.refresh_continuous_run_availability()
        self._connect_thread_cleanup()
        if not self.controller.requires_safe_shutdown():
            self._cleanup()

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
        code = getattr(exc, "code", "fast_run_start_failed")
        detail = str(exc)
        QtWidgets.QMessageBox.warning(
            self.labeling_widget,
            self.tr("无法开始快速标注"),
            f"{code}\n{detail}" if detail != code else code,
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
