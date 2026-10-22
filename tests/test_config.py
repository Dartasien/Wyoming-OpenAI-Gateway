"""Tests for the Settings configuration."""

from __future__ import annotations

from wyoming_openai_gateway.config import (
    Settings,
    normalize_prefix,
    openai_base_urls,
    route_prefixes,
    validate,
)


def test_defaults():
    """Test Settings with default values (all None)."""
    settings = Settings()
    assert settings.tts_host is None
    assert settings.tts_port is None
    assert settings.stt_host is None
    assert settings.stt_port is None
    assert settings.host == "0.0.0.0"
    assert settings.port == 8555
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.prefix == "/v1"


def test_custom_values():
    """Test Settings with custom values via constructor."""
    settings = Settings(
        tts_host="10.0.0.1",
        tts_port=10200,
        host="192.168.1.1",
        port=8080,
        debug=True,
        log_level="DEBUG",
        prefix="/api/v1",
    )
    assert settings.tts_host == "10.0.0.1"
    assert settings.tts_port == 10200
    assert settings.host == "192.168.1.1"
    assert settings.port == 8080
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.prefix == "/api/v1"


def test_from_env():
    """Test Settings.from_env() with a mapping."""
    settings = Settings.from_env({
        "TTS_HOST": "10.0.0.2",
        "TTS_PORT": "10206",
        "STT_HOST": "10.0.0.3",
        "STT_PORT": "10300",
        "HOST": "0.0.0.0",
        "PORT": "9000",
        "DEBUG": "true",
        "LOG_LEVEL": "WARNING",
    })
    assert settings.tts_host == "10.0.0.2"
    assert settings.tts_port == 10206
    assert settings.stt_host == "10.0.0.3"
    assert settings.stt_port == 10300
    assert settings.port == 9000
    assert settings.debug is True
    assert settings.log_level == "WARNING"


def test_validate_neither():
    """Validate returns errors when neither TTS nor STT is configured."""
    settings = Settings()
    errors = validate(settings)
    assert len(errors) == 1
    assert "Neither TTS_HOST nor STT_HOST" in errors[0]


def test_validate_partial_tts():
    """Validate returns error when TTS_HOST is set but TTS_PORT is not."""
    settings = Settings(tts_host="10.0.0.1")
    errors = validate(settings)
    assert len(errors) >= 1
    assert any("TTS_HOST" in e and "TTS_PORT" in e for e in errors)


def test_validate_partial_tts_port():
    """Validate returns error when TTS_PORT is set but TTS_HOST is not."""
    settings = Settings(tts_port=10200)
    errors = validate(settings)
    assert len(errors) >= 1
    assert any("TTS_PORT" in e and "TTS_HOST" in e for e in errors)


def test_validate_ok_tts_only():
    """Validate succeeds with only TTS configured."""
    settings = Settings(tts_host="10.0.0.1", tts_port=10200)
    assert validate(settings) == []


def test_validate_ok_stt_only():
    """Validate succeeds with only STT configured."""
    settings = Settings(stt_host="10.0.0.1", stt_port=10300)
    assert validate(settings) == []


def test_validate_ok_both():
    """Validate succeeds with both TTS and STT configured."""
    settings = Settings(tts_host="10.0.0.1", tts_port=10200, stt_host="10.0.0.2", stt_port=10300)
    assert validate(settings) == []


def test_prefix_normalisation():
    """PREFIX is normalised to a leading slash with no trailing slash."""
    for raw, expected in (
        ("/v1", "/v1"),
        ("v1", "/v1"),
        ("/v1/", "/v1"),
        ("", ""),
        ("/", ""),
        ("/api/v1/", "/api/v1"),
    ):
        assert Settings.from_env({"PREFIX": raw}).prefix == expected


def test_voice_fallback_default_true():
    """VOICE_FALLBACK defaults on, and accepts the usual truthy spellings."""
    assert Settings().voice_fallback is True
    assert Settings.from_env({}).voice_fallback is True
    for raw in ("false", "False", "0", "no", ""):
        assert Settings.from_env({"VOICE_FALLBACK": raw}).voice_fallback is False
    for raw in ("true", "TRUE", "1", "yes"):
        assert Settings.from_env({"VOICE_FALLBACK": raw}).voice_fallback is True


def test_ffmpeg_fallback_default_true():
    """FFMPEG_FALLBACK defaults on and can be disabled."""
    assert Settings().ffmpeg_fallback is True
    assert Settings.from_env({"FFMPEG_FALLBACK": "false"}).ffmpeg_fallback is False


def test_normalize_prefix():
    assert normalize_prefix("/v1") == "/v1"
    assert normalize_prefix("v1/") == "/v1"
    assert normalize_prefix("/") == ""
    assert normalize_prefix(None) == ""


def test_route_prefixes_covers_v1_and_root():
    """A custom PREFIX still answers on /v1 and on the root path."""
    assert route_prefixes("/api") == ("/api", "/v1", "")
    assert route_prefixes("/v1") == ("/v1", "")
    assert route_prefixes("") == ("", "/v1")
    assert route_prefixes("/api/v1") == ("/api/v1", "/v1", "")


def test_openai_base_urls():
    urls = openai_base_urls("0.0.0.0", 8555, route_prefixes("/v1"))
    assert "http://localhost:8555/v1" in urls
    assert "http://localhost:8555" in urls
