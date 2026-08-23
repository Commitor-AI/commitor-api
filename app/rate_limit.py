import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status

from app.config import get_settings


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            if len(self._hits) > 10_000:
                stale = [k for k, v in self._hits.items() if not v or now - v[-1] > self.window_seconds]
                for k in stale:
                    del self._hits[k]
            return True


def rate_limit_dependency(limiter: RateLimiter) -> Callable[[Request], Awaitable[None]]:
    async def dependency(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests, please try again later",
                headers={"Retry-After": str(limiter.window_seconds)},
            )

    return dependency


settings = get_settings()
login_limiter = RateLimiter(settings.login_rate_limit, settings.rate_limit_window_seconds)
signup_limiter = RateLimiter(settings.signup_rate_limit, settings.rate_limit_window_seconds)
analyze_limiter = RateLimiter(settings.analyze_rate_limit, settings.rate_limit_window_seconds)
