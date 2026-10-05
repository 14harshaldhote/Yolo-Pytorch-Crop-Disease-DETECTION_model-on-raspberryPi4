"""API key check and a small in-memory rate limiter."""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(expected: str, provided: str | None) -> None:
    """Raise 401 unless `provided` matches. No-op when no key is configured."""
    if not expected:
        return
    # compare_digest takes constant time, so the key can't be guessed byte by byte.
    if not provided or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API-Key"},
        )


class RateLimiter:
    """Sliding one-minute window per client. Good for a single process on a Pi;
    put a reverse proxy in front for anything bigger."""

    def __init__(self, per_minute: int, max_clients: int = 10_000) -> None:
        self.per_minute = per_minute
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, request: Request) -> None:
        if self.per_minute <= 0:
            return
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > self.max_clients:
                self._hits.clear()  # bound memory if flooded with many IPs
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                retry = int(60 - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)
