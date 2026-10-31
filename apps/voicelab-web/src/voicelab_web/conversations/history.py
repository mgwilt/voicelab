"""Commit completed exchanges only after the output transport drains."""

import threading


class History[Turn]:
    def __init__(self, initial: tuple[Turn, ...] = (), limit: int = 6) -> None:
        self._turns: tuple[Turn, ...] = initial[-limit:]
        self._pending: tuple[Turn, Turn] | None = None
        self._limit = limit
        self._lock = threading.Lock()

    def snapshot(self) -> tuple[Turn, ...]:
        with self._lock:
            return self._turns

    def stage(self, user: Turn, assistant: Turn) -> None:
        with self._lock:
            self._pending = (user, assistant)

    def played(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._turns = (*self._turns, *self._pending)[-self._limit :]
                self._pending = None

    def interrupt(self) -> None:
        with self._lock:
            self._pending = None
