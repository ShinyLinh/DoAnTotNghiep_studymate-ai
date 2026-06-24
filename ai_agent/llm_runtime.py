"""Shared LLM runtime limits, error classification, and provider readiness state."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


LLM_REQUEST_TIMEOUT_SECONDS = _float_env("LLM_REQUEST_TIMEOUT", 20.0, 1.0, 120.0)
LLM_TURN_TIMEOUT_SECONDS = _float_env("LLM_TURN_TIMEOUT", 35.0, 2.0, 300.0)
LLM_MAX_RETRIES = int(_float_env("LLM_MAX_RETRIES", 0, 0, 3))
PROVIDER_FAILURE_COOLDOWN_SECONDS = _float_env(
    "PROVIDER_FAILURE_COOLDOWN_SECONDS",
    60.0,
    1.0,
    3600.0,
)

_lock = threading.Lock()
_last_success_at: float | None = None
_last_failure_at: float | None = None
_last_error_type: str | None = None
_last_error_code: str | None = None


class LLMTurnTimeoutError(TimeoutError):
    pass


def classify_provider_error(exc: Exception, used_user_key: bool = False) -> dict:
    """Map provider-specific failures to stable API codes safe for the UI."""
    message = str(exc).lower()

    if isinstance(exc, LLMTurnTimeoutError) or "turn exceeded" in message or "timed out" in message:
        return {
            "code": "AI_TIMEOUT",
            "status": 504,
            "retryable": True,
            "message": "AI phản hồi quá thời gian cho phép. Hãy rút gọn yêu cầu hoặc thử lại.",
        }
    if "capacity queue is full" in message:
        return {
            "code": "AI_CAPACITY_FULL",
            "status": 503,
            "retryable": True,
            "message": "AI đang có quá nhiều yêu cầu cùng lúc. Vui lòng thử lại sau vài giây.",
        }
    if any(marker in message for marker in (
        "429", "rate-limit", "rate limit", "temporarily rate-limited",
        "too many requests", "quota exceeded", "resource_exhausted",
    )):
        return {
            "code": "AI_RATE_LIMITED" if used_user_key else "AI_FREE_QUOTA_EXHAUSTED",
            "status": 429,
            "retryable": True,
            "message": (
                "Provider đang giới hạn tốc độ hoặc quota của API key bạn đã nhập. "
                "Hãy kiểm tra quota/key hoặc thử lại sau."
                if used_user_key else
                "Hạn mức AI miễn phí của hệ thống đang tạm hết. "
                "Hãy thử lại sau, nâng cấp gói hoặc dùng API key riêng."
            ),
        }
    if any(marker in message for marker in (
        "402", "insufficient_quota", "insufficient quota", "credits",
        "credit balance", "afford", "billing", "payment required",
    )):
        return {
            "code": "AI_CREDIT_EXHAUSTED",
            "status": 402,
            "retryable": False,
            "message": (
                "API key của bạn đã hết hoặc không đủ credit cho model đã chọn."
                if used_user_key else
                "Tài khoản AI của hệ thống đã hết credit. Quản trị viên cần nạp thêm hoặc đổi model."
            ),
        }
    if any(marker in message for marker in (
        "401", "403", "api key", "authentication", "unauthorized",
        "permission denied", "invalid key",
    )):
        return {
            "code": "AI_AUTH_ERROR",
            "status": 401,
            "retryable": False,
            "message": "API key không hợp lệ, đã hết hạn hoặc không có quyền dùng model đã chọn.",
        }
    if any(marker in message for marker in (
        "context length", "maximum context", "max tokens", "token limit",
        "prompt is too long", "input is too long",
    )):
        return {
            "code": "AI_TOKEN_LIMIT",
            "status": 413,
            "retryable": False,
            "message": "Nội dung vượt giới hạn token của model. Hãy chia nhỏ tài liệu hoặc rút gọn yêu cầu.",
        }
    return {
        "code": "AI_PROVIDER_UNAVAILABLE",
        "status": 503,
        "retryable": True,
        "message": "AI provider tạm thời không phản hồi. Vui lòng thử lại sau.",
    }


def record_provider_success() -> None:
    global _last_success_at
    with _lock:
        _last_success_at = time.time()


def record_provider_failure(exc: Exception) -> None:
    global _last_failure_at, _last_error_type, _last_error_code
    with _lock:
        _last_failure_at = time.time()
        _last_error_type = type(exc).__name__
        _last_error_code = classify_provider_error(exc)["code"]


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def provider_status() -> dict:
    now = time.time()
    with _lock:
        last_success = _last_success_at
        last_failure = _last_failure_at
        last_error_type = _last_error_type
        last_error_code = _last_error_code

    recent_unrecovered_failure = bool(
        last_failure
        and (not last_success or last_failure > last_success)
        and now - last_failure < PROVIDER_FAILURE_COOLDOWN_SECONDS
    )
    return {
        "ready": not recent_unrecovered_failure,
        "configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "last_success_at": _iso_timestamp(last_success),
        "last_failure_at": _iso_timestamp(last_failure),
        "last_error_type": last_error_type,
        "last_error_code": last_error_code,
        "failure_cooldown_seconds": PROVIDER_FAILURE_COOLDOWN_SECONDS,
        "request_timeout_seconds": LLM_REQUEST_TIMEOUT_SECONDS,
        "turn_timeout_seconds": LLM_TURN_TIMEOUT_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
    }