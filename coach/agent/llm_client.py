from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

from coach.agent.prompts import SYSTEM_PROMPT, planner_prompt, summary_prompt

try:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    ResourceExhausted = Exception  # type: ignore[assignment,misc]
    TooManyRequests = Exception  # type: ignore[assignment,misc]

try:  # pragma: no cover - optional dependency
    from upstash_redis import Redis as UpstashRedis
except Exception:  # pragma: no cover
    UpstashRedis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quota-error detection
# ---------------------------------------------------------------------------

_QUOTA_ERRORS: tuple[type[Exception], ...] = (ResourceExhausted, TooManyRequests)
_QUOTA_STATUS_CODES = {429, 503}


def _is_quota_error(exc: Exception) -> bool:
    if isinstance(exc, _QUOTA_ERRORS):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in _QUOTA_STATUS_CODES:
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("quota", "rate limit", "resource exhausted", "429", "too many requests"))


# ---------------------------------------------------------------------------
# Key loading (env-var fallback)
# ---------------------------------------------------------------------------

class _RedisKeyQueue:
    """
    Redis-backed circular key queue stored in a single list.

    The current key is always at index 0. On rotation we:

    1. LPOP the head element.
    2. RPUSH it back to the tail.

    This ensures global rotation across all processes using the same list key.
    """

    def __init__(self, redis: Any, list_key: str) -> None:
        self._redis = redis
        self._list_key = list_key

    def __len__(self) -> int:
        """Return the number of keys currently in the Redis list."""
        try:
            return int(self._redis.llen(self._list_key) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not check Redis list length: %s", exc)
            return 0

    def ensure_seeded(self, fallback_keys: list[str]) -> None:
        """Seed the Redis list with *fallback_keys* if it's currently empty."""
        if not fallback_keys:
            return
        try:
            length = int(self._redis.llen(self._list_key) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not check Redis list length for key rotation: %s", exc)
            return
        if length > 0:
            return
        try:
            # Push all keys in the given order so rotation is predictable.
            self._redis.rpush(self._list_key, *fallback_keys)
            logger.info(
                "Seeded Redis list %r with %d Gemini API key(s).",
                self._list_key,
                len(fallback_keys),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to seed Redis list for Gemini keys: %s", exc)

    def current(self) -> str | None:
        """Return the key at the head of the list (index 0)."""
        try:
            value = self._redis.lindex(self._list_key, 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read current Gemini key from Redis: %s", exc)
            return None
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    def rotate(self) -> str | None:
        """
        Rotate the list by one element and return the new head key.

        Implementation: LPOP head, RPUSH it to the tail, then LINDEX 0.
        """
        try:
            # Atomically move the head of the list to the tail.
            # This prevents key loss if the process fails mid-operation.
            moved = self._redis.lmove(self._list_key, self._list_key, "LEFT", "RIGHT")
            if moved is None:
                logger.warning("Redis key list %r is empty; cannot rotate Gemini API key.", self._list_key)
                return None
            new_head = self._redis.lindex(self._list_key, 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rotate Gemini key in Redis: %s", exc)
            return None

        if new_head is None:
            return None
        if isinstance(new_head, bytes):
            return new_head.decode()
        return str(new_head)


def _load_env_keys_only() -> list[str]:
    """Load Gemini API keys from environment variables only."""
    keys: list[str] = []
    idx = 1
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{idx}", "").strip()
        if not k:
            break
        keys.append(k)
        idx += 1

    if not keys:
        single = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if single:
            keys.append(single)
    return keys


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Gemini wrapper with **circular key rotation** backed by Upstash Redis.

    Key rotation strategy
    ---------------------
    Keys are arranged in a **circular queue** natively inside Upstash Redis.
    On a quota/rate-limit error, the client commands Redis to advance the queue
    by moving the current key to the back (LPOP + RPUSH), providing
    coordinated global rotation via Redis.
    If every key in one full cycle fails, a ``RuntimeError`` is raised.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._redis_queue: _RedisKeyQueue | None = None

        # If an explicit api_key is given, use it directly.
        if api_key:
            self._keys: list[str] = [api_key]
            self._index: int = 0
        else:
            # Prefer Upstash Redis list when configured; this gives global rotation.
            redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
            redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
            redis_list = os.getenv("GEMINI_KEYS_REDIS_LIST", "gemini_api_keys")

            if UpstashRedis is not None and redis_url and redis_token:
                try:
                    redis = UpstashRedis(url=redis_url, token=redis_token)
                    queue = _RedisKeyQueue(redis=redis, list_key=redis_list)
                    # Seed from env-based keys if the list is currently empty.
                    queue.ensure_seeded(_load_env_keys_only())
                    current = queue.current()
                    if current:
                        self._redis_queue = queue
                        self._keys = [current]
                        self._store = None
                        self._index = 0
                        logger.info(
                            "Using Upstash Redis list %r for Gemini key rotation.",
                            redis_list,
                        )
                    else:
                        logger.warning(
                            "Upstash Redis is configured but no Gemini keys found in list %r; "
                            "falling back to gist/env configuration.",
                            redis_list,
                        )
                        self._redis_queue = None
                        self._init_env_fallback()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to initialize Upstash Redis for Gemini key rotation (%s); "
                        "falling back to env configuration.",
                        exc,
                    )
                    self._init_env_fallback()
            else:
                self._init_env_fallback()

        self.enabled = bool(self._keys and genai is not None)
        self.client = self._make_client() if self.enabled else None

    def _init_env_fallback(self) -> None:
        """Initialize key rotation from local env (no Redis)."""
        self._keys = _load_env_keys_only()
        self._index = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """The currently active API key."""
        return self._keys[self._index] if self._keys else None

    def plan(self, user_query: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        response = self._generate_with_rotation(
            contents=planner_prompt(user_query),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0),
        )
        text = getattr(response, "text", None) or ""
        return _extract_json_payload(text) if text.strip() else None

    def summarize(self, question: str, computed_payload: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        response = self._generate_with_rotation(
            contents=summary_prompt(question, computed_payload),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
        )
        text = getattr(response, "text", None)
        return text if text and text.strip() else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self) -> Any:
        return genai.Client(api_key=self.api_key)

    def _advance(self) -> None:
        """Advance to the next key and rebuild the client."""
        # When Redis is configured, rotate the global list instead of a local index.
        if self._redis_queue is not None:
            new_key = self._redis_queue.rotate()
            if not new_key:
                raise RuntimeError("Redis-backed Gemini key queue is empty; cannot rotate API key.")
            self._keys = [new_key]
            self._index = 0
            self.client = self._make_client()
            logger.warning("Gemini quota hit — rotated to next key from Redis list.")
            return

        # Fallback: local circular list with no persistence.
        self._index = (self._index + 1) % len(self._keys)
        self.client = self._make_client()
        logger.warning(
            "Gemini quota hit — rotated to key #%d of %d (index %d).",
            self._index + 1,
            len(self._keys),
            self._index,
        )

    def _generate_with_rotation(self, contents: Any, config: Any) -> Any:
        """
        Call ``generate_content`` with automatic circular key rotation.

        Tries each key at most once per call.  Raises ``RuntimeError`` if
        the entire pool is exhausted.
        """
        n = len(self._redis_queue) if self._redis_queue is not None else len(self._keys)
        if n == 0:
             raise RuntimeError("No Gemini API keys available.")
        last_exc: Exception | None = None

        for attempt in range(n):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                if _is_quota_error(exc):
                    last_exc = exc
                    self._advance()
                    # If we've looped all the way back to the key we started
                    # on, every key is exhausted.
                    if attempt == n - 1:
                        break
                    continue
                raise  # non-quota errors propagate immediately

        raise RuntimeError(
            f"All {n} Gemini API key(s) are quota-exhausted."
        ) from last_exc


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in response.", text, 0)
