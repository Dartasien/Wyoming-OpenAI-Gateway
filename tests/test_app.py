"""Tests for the FastAPI application routes."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from wyoming_openai_gateway.app import create_app
from wyoming_openai_gateway.config import Settings

# A port with no Wyoming server behind it, so routes answer without upstream audio.
_DEAD_PORT = 19999

# Valid WAV: 16-bit mono 16000 Hz, 160 samples (10ms) of silence
# Total file = 44 header + 320 data = 364 bytes
# RIFF chunk size = 364 - 8 = 356 (0x164)
# Data chunk size = 320 (0x140)
_WAV_SIZE = 356  # total file size - 8
_WAV_DATA_SIZE = 160 * 2  # 320 bytes

_WAV_SILENCE = (
    b"RIFF"
    + _WAV_SIZE.to_bytes(4, "little")
    + b"WAVE"
    + b"fmt "
    + b"\x10\x00\x00\x00"  # fmt chunk size (16)
    + b"\x01\x00"  # PCM
    + b"\x01\x00"  # mono
    + b"\x80\x3e\x00\x00"  # 16000 Hz
    + (16000 * 1 * 2).to_bytes(4, "little")  # byte rate
    + b"\x02\x00"  # block align (2)
    + b"\x10\x00"  # 16-bit
    + b"data"
    + _WAV_DATA_SIZE.to_bytes(4, "little")
    + b"\x00\x00" * 160  # 160 samples (10 ms at 16 kHz)
)


# ── Health / Readiness ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_healthz(client):
    """Test the /healthz endpoint returns OK."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_readyz_no_wyoming(client):
    """Test the /readyz endpoint returns 503 when no Wyoming server."""
    response = await client.get("/readyz")
    assert response.status_code == 503


# ── Voices ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_voices_no_wyoming(client):
    """Test the /v1/voices endpoint returns 503 when no Wyoming server."""
    response = await client.get("/v1/voices")
    assert response.status_code == 503


# ── TTS ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tts_missing_body(client):
    """Test POST /v1/audio/speech with missing body returns 422."""
    response = await client.post("/v1/audio/speech", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tts_invalid_voice(client):
    """Test POST /v1/audio/speech with invalid data."""
    response = await client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello",
            "voice": "test-voice",
            "model": "wyoming",
        },
    )
    # Should fail with 503 since no Wyoming server is available
    assert response.status_code in (422, 503)


# ── STT (Transcriptions) ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stt_transcriptions_no_wyoming(client):
    """Test POST /v1/audio/transcriptions returns 503 when no Wyoming server."""
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _WAV_SILENCE, "audio/wav")},
        data={"model": "whisper-1"},
    )
    # Should fail with 503 since no Wyoming ASR server is available,
    # or 422 if validation fails first on the file.
    assert response.status_code in (422, 503)


@pytest.mark.asyncio
async def test_stt_transcriptions_missing_file(client):
    """Test POST /v1/audio/transcriptions without a file returns 422."""
    response = await client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_stt_transcriptions_with_language(client):
    """Test POST /v1/audio/transcriptions with language and prompt fields."""
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", _WAV_SILENCE, "audio/wav")},
        data={
            "model": "whisper-1",
            "language": "en",
            "prompt": "This is a test.",
            "response_format": "json",
            "temperature": "0.2",
        },
    )
    # No Wyoming ASR server, so expect 503 (or 422 on file parse)
    assert response.status_code in (422, 503)


# ── STT (Translations) ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stt_translations_no_wyoming(client):
    """Test POST /v1/audio/translations returns 503 when no Wyoming server."""
    response = await client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", _WAV_SILENCE, "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert response.status_code in (422, 503)


@pytest.mark.asyncio
async def test_stt_translations_missing_file(client):
    """Test POST /v1/audio/translations without a file returns 422."""
    response = await client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1"},
    )
    assert response.status_code == 422


# ── Model listing and base URLs (Open Notebook integration) ──────────────────
#
# Open Notebook's "Test connection" and "Discover Models" both call
# GET {base_url}/models. A 404 there is what broke the integration, and the body
# has to be the OpenAI {"object": "list", "data": [...]} shape.


@pytest.mark.asyncio
async def test_models_returns_openai_shape(client):
    """GET /v1/models lists the gateway's models in OpenAI format."""
    response = await client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"

    ids = [model["id"] for model in body["data"]]
    assert "wyoming-tts" in ids
    assert "whisper-1" in ids
    for model in body["data"]:
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model


@pytest.mark.asyncio
async def test_model_ids_are_classifiable_by_open_notebook(client):
    """Open Notebook sorts discovered models by name substring: 'tts' / 'whisper'."""
    body = (await client.get("/v1/models")).json()
    ids = [model["id"] for model in body["data"]]
    assert any("tts" in model_id for model_id in ids)
    assert any("whisper" in model_id for model_id in ids)


@pytest.mark.asyncio
async def test_models_at_every_base_url_form(client):
    """Bare host, trailing slash and /v1 must all work as an Open Notebook base URL."""
    for path in ("/models", "/v1/models"):
        response = await client.get(path)
        assert response.status_code == 200, path


@pytest.mark.asyncio
async def test_speech_at_root_and_v1(client):
    """POST /audio/speech is reachable with or without the /v1 prefix."""
    for path in ("/audio/speech", "/v1/audio/speech"):
        response = await client.post(path, json={"input": "Hello", "model": "wyoming-tts"})
        assert response.status_code == 503, path


@pytest.mark.asyncio
async def test_transcriptions_at_root_and_v1(client):
    """POST /audio/transcriptions is reachable with or without the /v1 prefix."""
    for path in ("/audio/transcriptions", "/v1/audio/transcriptions"):
        response = await client.post(path, files={"file": ("t.wav", _WAV_SILENCE, "audio/wav")})
        assert response.status_code == 503, path


@pytest.mark.asyncio
async def test_voices_alias_path(client):
    """esperanto asks for GET {base_url}/audio/voices, not just /voices."""
    for path in ("/audio/voices", "/v1/audio/voices", "/v1/voices", "/voices"):
        response = await client.get(path)
        assert response.status_code == 503, path


@pytest.mark.asyncio
async def test_unknown_path_is_404(client):
    response = await client.get("/nope/models")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_root_documents_base_urls(client):
    """GET / tells the user which base URL to paste into Open Notebook."""
    response = await client.get("/")
    assert response.status_code == 200
    assert any(url.endswith("/v1") for url in response.json()["openai_base_urls"])


@pytest.mark.asyncio
async def test_readyz_names_the_unready_service(client):
    response = await client.get("/readyz")
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]


@pytest.mark.asyncio
async def test_custom_prefix_still_serves_v1_and_root():
    """A non-default PREFIX does not remove the /v1 and root routes."""
    settings = Settings(
        tts_host="127.0.0.1",
        tts_port=_DEAD_PORT,
        stt_host="127.0.0.1",
        stt_port=_DEAD_PORT,
        prefix="api",
    )
    async with AsyncClient(transport=ASGITransport(app=create_app(settings)), base_url="http://test") as ac:
        for path in ("/api/models", "/v1/models", "/models"):
            response = await ac.get(path)
            assert response.status_code == 200, path
