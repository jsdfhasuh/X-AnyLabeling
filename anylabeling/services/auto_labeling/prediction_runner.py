from collections import OrderedDict

from PyQt5 import QtCore

from anylabeling.services.auto_labeling.prediction_job import (
    PredictionOutcome,
    PredictionRequest,
)


class SequenceWorker(QtCore.QObject):
    outcome_ready = QtCore.pyqtSignal(object)

    def __init__(self, executor, commit_handler=None):
        super().__init__()
        self._executor = executor
        self._commit_handler = commit_handler

    @QtCore.pyqtSlot(object, object)
    def execute(self, request, lease_token):
        failure_code = "prediction_worker_failed"
        try:
            outcome = self._executor(request, lease_token)
            if not isinstance(outcome, PredictionOutcome):
                raise TypeError(
                    "request executor must return PredictionOutcome"
                )
            if outcome.context != request:
                outcome = PredictionOutcome.failed(
                    request,
                    "prediction_outcome_context_mismatch",
                    "worker outcome context does not match active request",
                )
            if self._commit_handler is not None:
                failure_code = "prediction_commit_failed"
                self._commit_handler(outcome)
        except Exception as exc:  # noqa: B902
            outcome = PredictionOutcome.failed(
                request,
                failure_code,
                str(exc) or type(exc).__name__,
            )
        self.outcome_ready.emit(outcome)


class PredictionRunner(QtCore.QObject):
    request_started = QtCore.pyqtSignal(object)
    outcome_ready = QtCore.pyqtSignal(object)
    runner_idle = QtCore.pyqtSignal(int)
    safe_to_close = QtCore.pyqtSignal(int)
    runner_error = QtCore.pyqtSignal(object)
    _execute_requested = QtCore.pyqtSignal(object, object)

    _INTENT_PRIORITY = {"NONE": 0, "PAUSE": 1, "STOP": 2, "CLOSE": 3}

    def __init__(
        self,
        model_manager,
        request_executor=None,
        commit_handler=None,
        outcome_handler=None,
        history_limit=128,
    ):
        super().__init__()
        if isinstance(history_limit, bool) or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        self.model_manager = model_manager
        self._request_executor = request_executor or self._execute_request
        self._commit_handler = commit_handler
        self._outcome_handler = outcome_handler
        self._history_limit = history_limit
        self._completed_attempts = OrderedDict()

        self.generation = 0
        self.active_request = None
        self.worker_thread = None
        self._worker = None
        self.worker_executing = False
        self.control_intent = "NONE"
        self._lease_token = None
        self._started = False
        self._dispose_requested = False
        self._shutdown_generation = None
        self._manager_safe_generation = None
        self._safe_emit_scheduled = False

        manager_safe = getattr(model_manager, "safe_to_close", None)
        if manager_safe is not None and hasattr(manager_safe, "connect"):
            manager_safe.connect(self._on_manager_safe_to_close)

    @property
    def completed_attempt_count(self):
        return len(self._completed_attempts)

    def start(self):
        if self._started or self._dispose_requested:
            return False
        if self.control_intent in {"STOP", "CLOSE"}:
            return False
        self.generation += 1
        self.worker_thread = QtCore.QThread(self)
        self._worker = SequenceWorker(
            self._request_executor, self._commit_handler
        )
        self._worker.moveToThread(self.worker_thread)
        self._execute_requested.connect(
            self._worker.execute, QtCore.Qt.QueuedConnection
        )
        self._worker.outcome_ready.connect(
            self._on_worker_outcome, QtCore.Qt.QueuedConnection
        )
        self.worker_thread.finished.connect(self._worker.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)
        self.worker_thread.start()
        self._started = True
        return True

    def submit(self, request):
        if not isinstance(request, PredictionRequest):
            return False
        if request.generation != self.generation:
            return False
        if not self.is_accepting_requests():
            return False
        acquire_lease = getattr(
            self.model_manager, "acquire_inference_lease", None
        )
        if callable(acquire_lease):
            token = acquire_lease("PREDICTION_RUNNER", request.run_id)
        else:
            token = self.model_manager.inference_lease.acquire(
                "PREDICTION_RUNNER", request.run_id
            )
        if token is None:
            return False

        self.active_request = request
        self._lease_token = token
        self.worker_executing = True
        self.request_started.emit(request)
        self._execute_requested.emit(request, token)
        return True

    def request_pause(self):
        self._raise_control_intent("PAUSE")
        return True

    def resume(self):
        if self.control_intent != "PAUSE" or self._dispose_requested:
            return False
        self.control_intent = "NONE"
        return True

    def request_stop(self):
        self._raise_control_intent("STOP")
        return True

    def request_safe_shutdown(self, reason):
        self._raise_control_intent("CLOSE")
        self._dispose_requested = True
        request_shutdown = getattr(
            self.model_manager, "request_safe_shutdown", None
        )
        if callable(request_shutdown):
            generation = request_shutdown(reason)
        else:
            generation = (self._shutdown_generation or 0) + 1
            QtCore.QTimer.singleShot(
                0, lambda: self._on_manager_safe_to_close(generation)
            )
        self._shutdown_generation = generation
        self._manager_safe_generation = None
        if self.active_request is None:
            self._dispose_thread()
        return generation

    def shutdown_when_idle(self):
        self._dispose_requested = True
        if self.active_request is None:
            self._dispose_thread()

    def dispose_after_idle(self):
        self.shutdown_when_idle()

    def is_idle(self):
        return self.active_request is None and not self.worker_executing

    def requires_safe_shutdown(self):
        """Return whether this runner still owns a live worker thread."""

        return self.worker_thread is not None

    def is_accepting_requests(self):
        return (
            self._started
            and self.worker_thread is not None
            and self.worker_thread.isRunning()
            and self.control_intent == "NONE"
            and self.active_request is None
            and not self.worker_executing
            and not self._dispose_requested
        )

    @QtCore.pyqtSlot(object)
    def _on_worker_outcome(self, outcome):
        if not isinstance(outcome, PredictionOutcome):
            return
        request = self.active_request
        if request is None:
            return
        identity = outcome.context.identity
        if (
            outcome.context.generation != self.generation
            or identity != request.identity
            or identity in self._completed_attempts
        ):
            return

        self._completed_attempts[identity] = None
        while len(self._completed_attempts) > self._history_limit:
            self._completed_attempts.popitem(last=False)

        if self._outcome_handler is not None:
            try:
                self._outcome_handler(outcome)
            except Exception as exc:  # noqa: B902
                self.runner_error.emit(exc)

        try:
            if (
                outcome.status == "succeeded"
                and request.delivery_mode == "LEGACY_CANVAS"
            ):
                try:
                    self.model_manager.deliver_auto_labeling_payload(
                        outcome.result
                    )
                except Exception as exc:  # noqa: B902
                    self.runner_error.emit(exc)
            self.outcome_ready.emit(outcome)
        finally:
            self.active_request = None
            self.worker_executing = False
            token = self._lease_token
            self._lease_token = None
            if token is not None:
                token.release()
            self._notify_manager_inference_idle()

            if self._dispose_requested or self.control_intent == "CLOSE":
                self._dispose_thread()
            self.runner_idle.emit(self.generation)

    def _raise_control_intent(self, requested):
        if (
            self._INTENT_PRIORITY[requested]
            > self._INTENT_PRIORITY[self.control_intent]
        ):
            self.control_intent = requested

    def _execute_request(self, request, lease_token):
        execute = getattr(
            self.model_manager, "execute_prediction_request_unleased", None
        )
        if not callable(execute):
            raise RuntimeError("model manager has no prediction executor")
        return execute(request, lease_token)

    def _notify_manager_inference_idle(self):
        notify = getattr(self.model_manager, "on_inference_idle", None)
        if callable(notify):
            notify()
            return
        # Compatibility for the small protocol fake used by contract tests.
        maybe_safe = getattr(self.model_manager, "_maybe_safe", None)
        if callable(maybe_safe):
            maybe_safe()

    def _dispose_thread(self):
        thread = self.worker_thread
        if thread is None:
            self._maybe_forward_safe_to_close()
            return
        if self.worker_executing:
            return
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
        self._started = False
        if thread is not None:
            thread.deleteLater()
        self._maybe_forward_safe_to_close()

    @QtCore.pyqtSlot(int)
    def _on_manager_safe_to_close(self, generation):
        if generation != self._shutdown_generation:
            return
        self._manager_safe_generation = generation
        self._maybe_forward_safe_to_close()

    def _maybe_forward_safe_to_close(self):
        generation = self._shutdown_generation
        if (
            generation is None
            or self._manager_safe_generation != generation
            or self.active_request is not None
            or self.worker_executing
            or self.model_manager.inference_lease.is_active
            or self.worker_thread is not None
            or self._safe_emit_scheduled
        ):
            return
        self._safe_emit_scheduled = True

        def emit_if_current():
            self._safe_emit_scheduled = False
            if generation == self._shutdown_generation:
                self.safe_to_close.emit(generation)

        QtCore.QTimer.singleShot(0, emit_if_current)
