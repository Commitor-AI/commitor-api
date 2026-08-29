import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, Response, status

from app.config import get_settings
from app.dependencies.auth import get_current_user_from_api_key
from app.models.user import Plan, User


# --- Core limiter -----------------------------------------------------------
#
# In-memory sliding-window counter. Fine for early scale (single process).
#
# To move to Redis later: implement a backend with the same
# `check(key, limit, window_seconds) -> LimitDecision` contract (e.g. via
# an atomic Lua script or INCR+EXPIRE) and swap the `_limiter` singleton in
# `get_limiter()`. No call-site changes needed.


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    limit: int
    remaining: int  # quota left after this request (0 when denied)
    reset_at: datetime  # wall-clock UTC instant a slot frees up


class InMemoryLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> LimitDecision:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > window_seconds:
                hits.popleft()
            allowed = len(hits) < limit
            if allowed:
                hits.append(now)
            remaining = max(limit - len(hits), 0)
            if len(self._hits) > 10_000:
                stale = [k for k, v in self._hits.items() if not v or now - v[-1] > window_seconds]
                for k in stale:
                    del self._hits[k]

        # Convert the monotonic window position to a wall-clock instant.
        if hits:
            seconds_until_free = max((hits[0] + window_seconds) - now, 0.0)
        else:
            seconds_until_free = float(window_seconds)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=seconds_until_free)
        return LimitDecision(allowed=allowed, limit=limit, remaining=remaining, reset_at=reset_at)


_limiter = InMemoryLimiter()


def get_limiter() -> InMemoryLimiter:
    """Single seam for swapping in a Redis-backed implementation."""
    return _limiter


def _rate_headers(decision: LimitDecision) -> dict[str, str]:
    reset_epoch = int(decision.reset_at.timestamp())
    return {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(reset_epoch),
    }


# --- IP-based limiting for pre-auth routes (login/signup) -------------------


class RateLimiter:
    """Legacy bool-flavored wrapper kept for the auth routers."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def allow(self, key: str) -> bool:
        return get_limiter().check(key, self.max_requests, self.window_seconds).allowed


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


# --- Plan-gated per-user limiting for AI-calling routes ---------------------

# Rolling 24-hour windows; per-user, keyed off the User row so plan changes
# take effect on the next request without a redeploy.
ANALYZE_WINDOW_SECONDS = 86_400

PLAN_ANALYZE_DAILY_LIMIT: dict[Plan, int] = {
    Plan.free: 20,
    Plan.pro: 200,
}


def analyze_daily_limit(plan: Plan) -> int:
    # Unknown future plan values fall back to the free tier.
    return PLAN_ANALYZE_DAILY_LIMIT.get(plan, PLAN_ANALYZE_DAILY_LIMIT[Plan.free])


# --- Plan-gated diff size for AI endpoints ---------------------------------

# Absolute ceiling the wire schema accepts; the per-plan limit below is the
# real gate. Free stays at 200k (the CLI caps slightly under this); Pro gets
# far more headroom. Raise the schema ceiling if PRO ever grows past it.
PLAN_ANALYZE_MAX_DIFF_CHARS: dict[Plan, int] = {
    Plan.free: 200_000,
    Plan.pro: 1_000_000,
}

PRICING_URL = "https://commitor.dev/pricing"


def analyze_max_diff_chars(plan: Plan) -> int:
    # Unknown future plan values fall back to the free tier.
    return PLAN_ANALYZE_MAX_DIFF_CHARS.get(plan, PLAN_ANALYZE_MAX_DIFF_CHARS[Plan.free])


class DiffTooLarge(Exception):
    """413 — the diff exceeds the caller's plan limit (upgrade to analyze bigger)."""

    def __init__(self, limit: int, plan: Plan = Plan.free) -> None:
        message = (
            f"This change is too large to analyze on the {plan.value} plan "
            f"(limit {limit:,} characters). Upgrade to Pro at {PRICING_URL} "
            f"to analyze changes this large."
        )
        self.body = {
            "error": "diff_too_large",
            "message": message,
            "limit": limit,
            "upgrade_url": PRICING_URL,
        }
        super().__init__(message)


class RateLimitExceeded(Exception):
    def __init__(self, decision: LimitDecision, message: str) -> None:
        self.decision = decision
        self.message = message
        body = {
            "error": "rate_limit_exceeded",
            "message": message,
            "limit": decision.limit,
            "reset_at": decision.reset_at.isoformat(),
        }
        headers = _rate_headers(decision)
        headers["Retry-After"] = str(max(int((decision.reset_at - datetime.now(timezone.utc)).total_seconds()), 1))
        super().__init__(message)
        self.body = body
        self.headers = headers


async def enforce_analyze_rate_limit(
    response: Response,
    user: User = Depends(get_current_user_from_api_key),
) -> None:
    """Per-user daily quota for AI endpoints; sets X-RateLimit-* headers."""
    decision = get_limiter().check(
        f"user:{user.id}", analyze_daily_limit(user.plan), ANALYZE_WINDOW_SECONDS
    )
    for name, value in _rate_headers(decision).items():
        response.headers[name] = value
    if not decision.allowed:
        reset_iso = decision.reset_at.isoformat()
        raise RateLimitExceeded(
            decision,
            f"Daily analyze limit of {decision.limit} reached. Quota resets at {reset_iso}.",
        )


# --- Pre-auth route instances -----------------------------------------------

settings = get_settings()
login_limiter = RateLimiter(settings.login_rate_limit, settings.rate_limit_window_seconds)
signup_limiter = RateLimiter(settings.signup_rate_limit, settings.rate_limit_window_seconds)
