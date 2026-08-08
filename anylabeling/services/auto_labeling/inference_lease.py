import threading
import time
import uuid
from contextlib import contextmanager


class InferenceLeaseToken:
    """Generation-bound handle for one inference owner."""

    def __init__(
        self,
        registry,
        lease_id,
        owner_kind,
        owner_id,
        acquired_at,
        generation,
    ):
        self._registry = registry
        self.lease_id = lease_id
        self.owner_kind = owner_kind
        self.owner_id = owner_id
        self.acquired_at = acquired_at
        self.generation = generation
        self.released = False

    def release(self):
        return self._registry._release(self)


class InferenceLeaseRegistry:
    """Thread-safe, single-owner inference lease with monotonic generations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._generation = 0
        self._active_token = None
        self._release_callbacks = []

    @property
    def is_active(self):
        with self._lock:
            return self._active_token is not None

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def acquire(self, owner_kind, owner_id):
        if not isinstance(owner_kind, str) or not owner_kind:
            raise ValueError("owner_kind must be a non-empty string")
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id must be a non-empty string")
        with self._lock:
            if self._active_token is not None:
                return None
            self._generation += 1
            token = InferenceLeaseToken(
                registry=self,
                lease_id=str(uuid.uuid4()),
                owner_kind=owner_kind,
                owner_id=owner_id,
                acquired_at=time.time(),
                generation=self._generation,
            )
            self._active_token = token
            return token

    @contextmanager
    def hold(self, owner_kind, owner_id):
        token = self.acquire(owner_kind, owner_id)
        try:
            yield token
        finally:
            if token is not None:
                token.release()

    def is_active_token(self, token):
        with self._lock:
            return (
                token is not None
                and self._active_token is token
                and not token.released
                and token.generation == self._generation
            )

    def snapshot(self):
        with self._lock:
            token = self._active_token
            if token is None:
                return {
                    "lease_id": None,
                    "owner_kind": None,
                    "owner_id": None,
                    "acquired_at": None,
                    "released": True,
                    "generation": self._generation,
                }
            return {
                "lease_id": token.lease_id,
                "owner_kind": token.owner_kind,
                "owner_id": token.owner_id,
                "acquired_at": token.acquired_at,
                "released": token.released,
                "generation": token.generation,
            }

    def add_release_callback(self, callback):
        with self._lock:
            if callback not in self._release_callbacks:
                self._release_callbacks.append(callback)

    def _release(self, token):
        callbacks = []
        with self._lock:
            if token.released:
                return (
                    self._active_token is None
                    and token.generation == self._generation
                )
            if (
                self._active_token is not token
                or token.generation != self._generation
            ):
                return False
            token.released = True
            self._active_token = None
            callbacks = list(self._release_callbacks)
        for callback in callbacks:
            callback(token)
        return True
