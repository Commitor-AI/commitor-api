import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


class OpenRouterError(Exception):
    pass


class OpenRouterTimeoutError(OpenRouterError):
    pass


class OpenRouterResponseError(OpenRouterError):
    pass


@dataclass
class ChatResult:
    content: dict[str, Any]
    raw_content: str
    model: str
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


def assistant_message(content: str, reasoning_details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "reasoning_details": reasoning_details,
    }


class OpenRouterClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured")
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _describe(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    async def _post_with_retry(self, payload: dict[str, Any], http_timeout: httpx.Timeout) -> httpx.Response:
        attempts = 2
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._http.post("/chat/completions", json=payload, timeout=http_timeout)
            except httpx.TimeoutException as exc:
                raise OpenRouterTimeoutError(f"OpenRouter request timed out after {http_timeout.read}s") from exc
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                logger.warning("transient OpenRouter transport error (%s); retrying", self._describe(exc))
                await asyncio.sleep(0.5 * attempt)
        raise OpenRouterError(
            f"OpenRouter request failed after {attempts} attempts ({self._describe(last_exc)})"
        ) from last_exc

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        reasoning_enabled: bool = False,
        timeout: float | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        effective_timeout = timeout or self._settings.openrouter_timeout_seconds
        http_timeout = httpx.Timeout(effective_timeout, connect=10.0)
        started = time.monotonic()
        response = await self._post_with_retry(payload, http_timeout)
        if response.status_code != 200:
            raise OpenRouterError(f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}")
        try:
            body = response.json()
            message = body["choices"][0]["message"]
            raw_content = message["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OpenRouterResponseError(f"Unexpected OpenRouter response shape: {response.text[:500]}") from exc
        return ChatResult(
            content=parse_json_object(raw_content),
            raw_content=raw_content,
            model=body.get("model", model),
            reasoning=message.get("reasoning"),
            reasoning_details=message.get("reasoning_details") or [],
            usage=body.get("usage") or {},
            latency_seconds=time.monotonic() - started,
        )

    async def chat_completion_with_reasoning(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, Any]] | None = None,
        follow_up: str | None = None,
    ) -> ChatResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_prompt})
        if follow_up is not None:
            messages.append({"role": "user", "content": follow_up})
        return await self.chat_completion(
            model=self._settings.openrouter_reasoning_model,
            messages=messages,
            reasoning_enabled=True,
            timeout=self._settings.openrouter_reasoning_timeout_seconds,
        )


def parse_json_object(raw: str) -> dict[str, Any]:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenRouterResponseError(f"Model did not return valid JSON: {raw[:300]}") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterResponseError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


_client: OpenRouterClient | None = None


def get_openrouter_client() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
