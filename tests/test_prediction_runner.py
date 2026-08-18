import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image
from PyQt5 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.inference_lease import (
    InferenceLeaseRegistry,
)
from anylabeling.services.auto_labeling.prediction_job import (
    AutoLabelingPayload,
    PredictionOutcome,
    PredictionRequest,
)
from anylabeling.services.auto_labeling.prediction_runner import (
    PredictionRunner,
)
from anylabeling.views.labeling.utils.auto_labeling_commit import (
    commit_label_for_image_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_run_store import (
    InMemoryCommitStoreV1,
)


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_until(predicate, timeout_ms=2000):
    app = _application()
    deadline = time.monotonic() + (timeout_ms / 1000)
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)
    return predicate()


class _FakeManager(QtCore.QObject):
    new_auto_labeling_result = QtCore.pyqtSignal(object)
    safe_to_close = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.inference_lease = InferenceLeaseRegistry()
        self.download_active = False
        self.shutdown_generation = 0
        self.pending_shutdown = None

    def deliver_auto_labeling_payload(self, payload):
        self.new_auto_labeling_result.emit(payload)

    def request_safe_shutdown(self, _reason):
        self.shutdown_generation += 1
        self.pending_shutdown = self.shutdown_generation
        self._maybe_safe()
        return self.shutdown_generation

    def _maybe_safe(self):
        generation = self.pending_shutdown
        if (
            generation is None
            or self.download_active
            or self.inference_lease.is_active
        ):
            return
        QtCore.QTimer.singleShot(
            0,
            lambda: (
                self.safe_to_close.emit(generation)
                if generation == self.pending_shutdown
                else None
            ),
        )

    def finish_download(self):
        self.download_active = False
        self._maybe_safe()


def _request(runner, attempt="attempt-a", mode="RETURN_ONLY", **overrides):
    values = {
        "run_id": "run-a",
        "job_id": f"job-{attempt}",
        "attempt_id": attempt,
        "image_id": f"image-{attempt}",
        "canonical_image_path": os.path.abspath(__file__),
        "generation": runner.generation,
        "parameter_snapshot": {},
        "existing_shapes_input": [],
        "delivery_mode": mode,
    }
    values.update(overrides)
    return PredictionRequest(**values)


def _success(request, label=None):
    label = label or request.attempt_id
    payload = AutoLabelingPayload(
        [
            {
                "label": label,
                "points": [[1, 2]],
                "group_id": None,
                "shape_type": "point",
                "flags": {},
            }
        ],
        True,
        "",
    )
    return PredictionOutcome.succeeded(request, payload)


class PredictionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _application()

    def setUp(self):
        self.manager = _FakeManager()
        self.runners = []

    def tearDown(self):
        for runner in self.runners:
            runner.dispose_after_idle()
        _wait_until(
            lambda: all(
                runner.worker_thread is None for runner in self.runners
            )
        )

    def _runner(self, executor, **kwargs):
        runner = PredictionRunner(
            self.manager,
            request_executor=executor,
            **kwargs,
        )
        runner.start()
        self.runners.append(runner)
        return runner

    def test_two_requests_use_one_long_lived_thread_and_run_serially(self):
        thread_ids = []

        def execute(request, _token):
            thread_ids.append(int(QtCore.QThread.currentThreadId()))
            return _success(request)

        runner = self._runner(execute)
        worker_thread = runner.worker_thread
        outcomes = []
        idle_states = []
        runner.outcome_ready.connect(outcomes.append)
        runner.runner_idle.connect(
            lambda _generation: idle_states.append(
                (runner.active_request, self.manager.inference_lease.is_active)
            )
        )

        self.assertTrue(runner.submit(_request(runner, "attempt-a")))
        self.assertTrue(_wait_until(lambda: len(idle_states) == 1))
        self.assertTrue(runner.submit(_request(runner, "attempt-b")))
        self.assertTrue(_wait_until(lambda: len(idle_states) == 2))

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(thread_ids[0], thread_ids[1])
        self.assertIs(runner.worker_thread, worker_thread)
        self.assertEqual(idle_states, [(None, False), (None, False)])

    def test_duplicate_late_and_old_generation_results_are_discarded(self):
        release = threading.Event()

        def execute(request, _token):
            if request.attempt_id == "attempt-b":
                release.wait(1)
            return _success(request)

        runner = self._runner(execute)
        outcomes = []
        runner.outcome_ready.connect(outcomes.append)
        request_a = _request(runner, "attempt-a")
        self.assertTrue(runner.submit(request_a))
        self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
        runner._on_worker_outcome(_success(request_a, "duplicate"))
        self.assertEqual(len(outcomes), 1)

        request_b = _request(runner, "attempt-b")
        self.assertTrue(runner.submit(request_b))
        self.assertTrue(
            _wait_until(lambda: runner.active_request == request_b)
        )
        runner._on_worker_outcome(_success(request_a, "late"))
        old = PredictionRequest(
            run_id="run-a",
            job_id="old-job",
            attempt_id="old-attempt",
            image_id="old-image",
            canonical_image_path=os.path.abspath(__file__),
            generation=runner.generation - 1,
            parameter_snapshot={},
            existing_shapes_input=[],
            delivery_mode="RETURN_ONLY",
        )
        runner._on_worker_outcome(_success(old, "old-generation"))
        release.set()
        self.assertTrue(_wait_until(lambda: len(outcomes) == 2))
        self.assertEqual(
            [outcome.context.attempt_id for outcome in outcomes],
            ["attempt-a", "attempt-b"],
        )

    def test_return_only_never_uses_legacy_canvas_and_legacy_is_once(self):
        runner = self._runner(lambda request, _token: _success(request))
        legacy = []
        self.manager.new_auto_labeling_result.connect(legacy.append)
        self.assertTrue(runner.submit(_request(runner, "return-only")))
        self.assertTrue(_wait_until(runner.is_idle))
        self.assertEqual(legacy, [])
        self.assertTrue(
            runner.submit(_request(runner, "legacy", mode="LEGACY_CANVAS"))
        )
        self.assertTrue(_wait_until(lambda: len(legacy) == 1))
        self.assertEqual(len(legacy), 1)

    def test_idle_is_after_handler_cleanup_and_lease_release(self):
        handler_observation = []
        idle_observation = []
        runner = self._runner(
            lambda request, _token: _success(request),
            outcome_handler=lambda _outcome: handler_observation.append(
                (
                    runner.active_request is not None,
                    self.manager.inference_lease.is_active,
                )
            ),
        )
        runner.runner_idle.connect(
            lambda _generation: idle_observation.append(
                (
                    runner.active_request,
                    self.manager.inference_lease.is_active,
                    runner.worker_executing,
                )
            )
        )
        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(lambda: len(idle_observation) == 1))
        self.assertEqual(handler_observation, [(True, True)])
        self.assertEqual(idle_observation, [(None, False, False)])

    def test_handler_and_worker_exceptions_still_produce_safe_idle(self):
        errors = []
        outcomes = []

        def fail_handler(_outcome):
            raise RuntimeError("handler boom")

        runner = self._runner(
            lambda _request, _token: (_ for _ in ()).throw(
                RuntimeError("worker boom")
            ),
            outcome_handler=fail_handler,
        )
        runner.runner_error.connect(errors.append)
        runner.outcome_ready.connect(outcomes.append)
        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(runner.is_idle))
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].status, "failed")
        self.assertEqual(len(errors), 1)
        self.assertFalse(self.manager.inference_lease.is_active)

    def test_commit_handler_runs_once_in_worker_thread(self):
        calls = []
        gui_thread = int(QtCore.QThread.currentThreadId())

        def commit(outcome):
            calls.append(
                (
                    outcome.context.attempt_id,
                    int(QtCore.QThread.currentThreadId()),
                )
            )

        runner = PredictionRunner(
            self.manager,
            request_executor=lambda request, _token: _success(request),
            commit_handler=commit,
        )
        runner.start()
        self.runners.append(runner)
        outcomes = []
        runner.outcome_ready.connect(outcomes.append)

        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
        runner._on_worker_outcome(outcomes[0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "attempt-a")
        self.assertNotEqual(calls[0][1], gui_thread)

    def test_commit_handler_uses_phase1_in_memory_store_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            label_path = root / "sample.json"
            Image.new("RGB", (8, 6), color=(20, 30, 40)).save(image_path)
            store = InMemoryCommitStoreV1()
            store.create_item("image-attempt-a", "attempt-a")
            committed = []

            def commit(outcome):
                result = outcome.result
                committed.append(
                    commit_label_for_image_v1(
                        store=store,
                        image_id=outcome.context.image_id,
                        attempt_id=outcome.context.attempt_id,
                        label_path=label_path,
                        image_path=image_path,
                        image_height=6,
                        image_width=8,
                        prediction_outcome={
                            "status": outcome.status,
                            "shapes": result.shapes,
                            "replace": result.replace,
                            "description": result.description,
                            "zero_target": outcome.zero_target,
                            "error_code": outcome.error_code,
                        },
                        write_policy="FORCE_REPLACE",
                        allowed_root=root,
                    )
                )

            runner = PredictionRunner(
                self.manager,
                request_executor=lambda request, _token: _success(request),
                commit_handler=commit,
            )
            runner.start()
            self.runners.append(runner)
            outcomes = []
            runner.outcome_ready.connect(outcomes.append)

            self.assertTrue(runner.submit(_request(runner)))
            self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
            self.assertEqual(len(committed), 1)
            item = store.read_item("image-attempt-a")
            self.assertEqual(item["staged_commit_status"], "committed")
            self.assertEqual(item["source_commit_status"], "pending")
            self.assertTrue(label_path.is_file())

    def test_commit_handler_failure_becomes_one_failed_outcome(self):
        def fail_commit(_outcome):
            raise RuntimeError("commit boom")

        runner = PredictionRunner(
            self.manager,
            request_executor=lambda request, _token: _success(request),
            commit_handler=fail_commit,
        )
        runner.start()
        self.runners.append(runner)
        outcomes = []
        runner.outcome_ready.connect(outcomes.append)

        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
        self.assertEqual(outcomes[0].status, "failed")
        self.assertEqual(outcomes[0].error_code, "prediction_commit_failed")
        self.assertTrue(runner.is_idle())

    def test_worker_failure_keeps_worker_error_with_commit_handler(self):
        commits = []
        runner = PredictionRunner(
            self.manager,
            request_executor=lambda _request, _token: (_ for _ in ()).throw(
                RuntimeError("worker boom")
            ),
            commit_handler=commits.append,
        )
        runner.start()
        self.runners.append(runner)
        outcomes = []
        runner.outcome_ready.connect(outcomes.append)

        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
        self.assertEqual(outcomes[0].status, "failed")
        self.assertEqual(outcomes[0].error_code, "prediction_worker_failed")
        self.assertEqual(commits, [])

    def test_worker_context_mismatch_fails_the_active_attempt_once(self):
        def mismatched(request, _token):
            other = PredictionRequest(
                run_id=request.run_id,
                job_id="wrong-job",
                attempt_id="wrong-attempt",
                image_id="wrong-image",
                canonical_image_path=request.canonical_image_path,
                generation=request.generation,
                parameter_snapshot={},
                existing_shapes_input=[],
                delivery_mode="RETURN_ONLY",
            )
            return _success(other)

        runner = self._runner(mismatched)
        outcomes = []
        runner.outcome_ready.connect(outcomes.append)
        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
        self.assertEqual(outcomes[0].context.attempt_id, "attempt-a")
        self.assertEqual(outcomes[0].status, "failed")
        self.assertTrue(runner.is_idle())

    def test_busy_submit_does_not_replace_active_context(self):
        release = threading.Event()
        runner = self._runner(
            lambda request, _token: (release.wait(1), _success(request))[1]
        )
        first = _request(runner, "attempt-a")
        second = _request(runner, "attempt-b")
        self.assertTrue(runner.submit(first))
        self.assertTrue(_wait_until(lambda: runner.active_request == first))
        self.assertFalse(runner.submit(second))
        self.assertEqual(runner.active_request, first)
        release.set()
        self.assertTrue(_wait_until(runner.is_idle))

    def test_completed_attempt_history_is_bounded(self):
        runner = self._runner(
            lambda request, _token: _success(request), history_limit=2
        )
        for index in range(4):
            self.assertTrue(
                runner.submit(_request(runner, f"attempt-{index}"))
            )
            self.assertTrue(_wait_until(runner.is_idle))
        self.assertEqual(runner.completed_attempt_count, 2)

    def test_pause_stop_close_priority_and_resume_rules(self):
        release = threading.Event()
        runner = self._runner(
            lambda request, _token: (release.wait(1), _success(request))[1]
        )
        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(runner.request_pause())
        self.assertFalse(runner.is_accepting_requests())
        self.assertTrue(runner.request_stop())
        self.assertEqual(runner.control_intent, "STOP")
        self.assertFalse(runner.resume())
        generation = runner.request_safe_shutdown("window_close")
        self.assertGreater(generation, 0)
        self.assertEqual(runner.control_intent, "CLOSE")
        self.assertTrue(runner.request_pause())
        self.assertEqual(runner.control_intent, "CLOSE")
        release.set()
        self.assertTrue(_wait_until(lambda: runner.worker_thread is None))
        self.assertFalse(runner.resume())

    def test_pause_applies_after_request_and_resume_allows_submit(self):
        runner = self._runner(lambda request, _token: _success(request))
        self.assertTrue(runner.request_pause())
        self.assertFalse(runner.submit(_request(runner)))
        self.assertTrue(runner.resume())
        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(_wait_until(runner.is_idle))

    def test_pause_waits_for_active_request_and_repeated_intents_are_idempotent(
        self,
    ):
        release = threading.Event()
        runner = self._runner(
            lambda request, _token: (release.wait(1), _success(request))[1]
        )
        outcomes = []
        runner.outcome_ready.connect(outcomes.append)
        self.assertTrue(runner.submit(_request(runner)))
        self.assertTrue(runner.request_pause())
        self.assertTrue(runner.request_pause())
        self.assertEqual(runner.control_intent, "PAUSE")
        release.set()
        self.assertTrue(_wait_until(lambda: len(outcomes) == 1))
        self.assertFalse(runner.is_accepting_requests())
        self.assertTrue(runner.resume())
        self.assertTrue(runner.request_stop())
        self.assertTrue(runner.request_stop())
        self.assertEqual(runner.control_intent, "STOP")
        self.assertFalse(runner.resume())

    def test_safe_shutdown_is_deferred_and_waits_for_work_and_download(self):
        release = threading.Event()
        runner = self._runner(
            lambda request, _token: (release.wait(1), _success(request))[1]
        )
        safe = []
        runner.safe_to_close.connect(safe.append)
        self.manager.download_active = True
        self.assertTrue(runner.submit(_request(runner)))
        generation = runner.request_safe_shutdown("window_close")
        self.assertEqual(safe, [])
        release.set()
        self.assertTrue(_wait_until(lambda: runner.active_request is None))
        self.assertEqual(safe, [])
        self.manager.finish_download()
        self.assertTrue(_wait_until(lambda: safe == [generation]))
        self.assertFalse(self.manager.inference_lease.is_active)
        self.assertFalse(runner.worker_executing)
        self.assertIsNone(runner.worker_thread)

    def test_idle_safe_shutdown_uses_next_event_loop_tick_and_latest_generation(
        self,
    ):
        runner = self._runner(lambda request, _token: _success(request))
        safe = []
        runner.safe_to_close.connect(safe.append)
        first = runner.request_safe_shutdown("first")
        second = runner.request_safe_shutdown("second")
        self.assertEqual(safe, [])
        self.assertGreater(second, first)
        self.assertTrue(_wait_until(lambda: safe == [second]))


if __name__ == "__main__":
    unittest.main()
