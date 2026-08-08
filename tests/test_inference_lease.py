import threading
import unittest

from anylabeling.services.auto_labeling.inference_lease import (
    InferenceLeaseRegistry,
)


class InferenceLeaseTests(unittest.TestCase):
    def test_only_one_owner_can_hold_the_registry(self):
        registry = InferenceLeaseRegistry()
        first = registry.acquire("LEGACY_SINGLE", "single-a")
        self.assertIsNotNone(first)
        self.assertIsNone(registry.acquire("PREDICTION_RUNNER", "runner-a"))
        self.assertTrue(registry.is_active_token(first))
        self.assertTrue(first.release())
        self.assertFalse(registry.is_active)

    def test_old_token_cannot_release_a_new_generation(self):
        registry = InferenceLeaseRegistry()
        old = registry.acquire("LEGACY_SINGLE", "single-a")
        self.assertTrue(old.release())
        current = registry.acquire("PREDICTION_RUNNER", "runner-a")
        self.assertFalse(old.release())
        self.assertTrue(registry.is_active_token(current))
        self.assertTrue(current.release())

    def test_release_is_idempotent_and_snapshot_has_frozen_fields(self):
        registry = InferenceLeaseRegistry()
        token = registry.acquire("LEGACY_AUTO_RUN", "batch-a")
        snapshot = registry.snapshot()
        self.assertEqual(
            set(snapshot),
            {
                "lease_id",
                "owner_kind",
                "owner_id",
                "acquired_at",
                "released",
                "generation",
            },
        )
        self.assertTrue(token.release())
        self.assertTrue(token.released)
        self.assertTrue(token.release())

    def test_threaded_acquire_has_one_winner(self):
        registry = InferenceLeaseRegistry()
        barrier = threading.Barrier(3)
        winners = []

        def acquire(owner):
            barrier.wait()
            token = registry.acquire("LEGACY_SINGLE", owner)
            if token is not None:
                winners.append(token)

        threads = [
            threading.Thread(target=acquire, args=(f"owner-{index}",))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(len(winners), 1)
        winners[0].release()

    def test_context_manager_releases_on_exception(self):
        registry = InferenceLeaseRegistry()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with registry.hold("LEGACY_SINGLE", "single-a") as token:
                self.assertIsNotNone(token)
                raise RuntimeError("boom")
        self.assertFalse(registry.is_active)


if __name__ == "__main__":
    unittest.main()
