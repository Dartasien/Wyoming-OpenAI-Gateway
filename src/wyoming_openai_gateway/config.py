"""Environment-driven configuration for Wyoming-OpenAI-Gateway."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


def _get_bool(key: str, default: str | bool = False) -> bool:
    if isinstance(default, bool):
        default_str = str(default).lower()
    else:
        default_str = default
    val = os.environ.get(key, default_str).lower()
    return val in ("true", "1", "t", "yes")


def _get_int(key: str, default: str | int | None) -> int | None:
    raw = os.environ.get(key)
    if raw is None:
        if isinstance(default, int):
            return default
        return default
    return int(raw)


def _get_str(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _get_log_level(key: str, default: str = "INFO") -> str:
    level = os.environ.get(key, default).upper()
    valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    return level if level in valid else default


def normalize_prefix(prefix: str | None) -> str:
    """Normalize an API prefix to '' or '/segment' with no trailing slash.

    ``/v1`` -> ``/v1``, ``v1`` -> ``/v1``, ``/`` and empty -> ``''`` (routes at the root).
    """
    if not prefix:
        return ""
    cleaned = prefix.strip()
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    return cleaned.rstrip("/")


def route_prefixes(primary: str) -> tuple[str, ...]:
    """Every prefix the OpenAI-compatible routes are served under.

    OpenAI-compatible clients (Open Notebook via esperanto, the ``openai`` SDK,
    curl) append ``/models``, ``/audio/speech``, ``/audio/transcriptions`` to the
    base URL the user typed — they never add ``/v1`` themselves. Mounting the
    routes under the configured prefix, the canonical ``/v1`` **and** the root
    means any of these work as a base URL:

        http://host:8555      http://host:8555/      http://host:8555/v1
    """
    prefixes: dict[str, None] = {}
    for candidate in (primary, "/v1", ""):
        prefixes.setdefault(candidate, None)
    return tuple(prefixes)


def openai_base_urls(host: str, port: int, prefixes: tuple[str, ...]) -> list[str]:
    """Base URLs a client can be pointed at, for startup logging."""
    display_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    return [f"http://{display_host}:{port}{prefix}" for prefix in prefixes]


def validate(settings: Settings) -> list[str]:
    """Validate settings and return a list of error messages (empty = valid)."""
    errors: list[str] = []

    tts_set = settings.tts_host is not None
    tts_port_set = settings.tts_port is not None

    stt_set = settings.stt_host is not None
    stt_port_set = settings.stt_port is not None

    # Partial TTS configuration
    if tts_set != tts_port_set:
        if tts_set:
            errors.append("TTS_HOST is set but TTS_PORT is not defined (or vice versa)")
        else:
            errors.append("TTS_PORT is set but TTS_HOST is not defined (or vice versa)")

    # Partial STT configuration
    if stt_set != stt_port_set:
        if stt_set:
            errors.append("STT_HOST is set but STT_PORT is not defined (or vice versa)")
        else:
            errors.append("STT_PORT is set but STT_HOST is not defined (or vice versa)")

    # At least one service must be configured
    if not tts_set and not stt_set:
        errors.append(
            "Neither TTS_HOST nor STT_HOST are defined. "
            "At least one speech service must be configured."
        )

    return errors


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings parsed from environment variables."""

    tts_host: str | None = None
    tts_port: int | None = None
    stt_host: str | None = None
    stt_port: int | None = None
    host: str = "0.0.0.0"
    port: int = 8555
    debug: bool = False
    log_level: str = "INFO"
    prefix: str = "/v1"
    voice_fallback: bool = True
    ffmpeg_fallback: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Create Settings from environment variables or a mapping (for testing)."""
        if env is not None:
            old_environ = os.environ.copy()
            os.environ.update({k: v for k, v in env.items() if v is not None})
            try:
                return cls._parse()
            finally:
                os.environ.clear()
                os.environ.update(old_environ)
        return cls._parse()

    @classmethod
    def _parse(cls) -> Settings:
        return cls(
            tts_host=_get_str("TTS_HOST"),
            tts_port=_get_int("TTS_PORT", None),
            stt_host=_get_str("STT_HOST"),
            stt_port=_get_int("STT_PORT", None),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=_get_int("PORT", 8555),
            debug=_get_bool("DEBUG", False),
            log_level=_get_log_level("LOG_LEVEL", "INFO"),
            prefix=normalize_prefix(os.environ.get("PREFIX", "/v1")),
            voice_fallback=_get_bool("VOICE_FALLBACK", True),
            ffmpeg_fallback=_get_bool("FFMPEG_FALLBACK", True),
        )
