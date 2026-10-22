"""Tests for the Wyoming <-> OpenAI translation layer."""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

from wyoming_openai_gateway.config import Settings
from wyoming_openai_gateway.errors import WyomingConnectionError
from wyoming_openai_gateway.gateway import (
    STT_MODEL_ID,
    TTS_MODEL_ID,
    OpenAIGateway,
    _filename_suffix,
)
from wyoming_openai_gateway.openai_models import SpeechRequest, TranscriptionRequest

_PCM = b"\x00\x00" * 160  # 10 ms of 16-bit silence
_AUDIO_PARAMS = {"rate": 16000, "width": 2, "channels": 1}


def _gateway(**overrides) -> OpenAIGateway:
    values = {
        "tts_host": "127.0.0.1",
        "tts_port": 10200,
        "stt_host": "127.0.0.1",
        "stt_port": 10300,
    }
    values.update(overrides)
    return OpenAIGateway(Settings(**values))


def _info(voices: list[str], installed: bool = True) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="wyoming-piper",
                attribution=Attribution(name="test", url=""),
                installed=True,
                description=None,
                version="1.0",
                voices=[
                    TtsVoice(
                        name=name,
                        attribution=Attribution(name="test", url=""),
                        installed=installed,
                        description=None,
                        version=None,
                        languages=["en"],
                    )
                    for name in voices
                ],
            )
        ]
    )


class _FakeClient:
    """Stand-in for WyomingStreamClient that only answers Describe."""

    def __init__(self, info: Info | None, error: Exception | None = None) -> None:
        self._info = info
        self._error = error

    async def describe(self) -> Info:
        if self._error:
            raise self._error
        assert self._info is not None
        return self._info


# ── response_format negotiation ───────────────────────────────────────────────


def test_wav_is_the_default_format():
    gateway = _gateway()
    response = gateway._audio_response(_PCM, _AUDIO_PARAMS, SpeechRequest(input="hi"))
    assert response.media_type == "audio/wav"
    assert response.body[:4] == b"RIFF"


def test_pcm_is_returned_raw():
    gateway = _gateway()
    request = SpeechRequest(input="hi", response_format="pcm")
    response = gateway._audio_response(_PCM, _AUDIO_PARAMS, request)
    assert response.media_type == "audio/pcm"
    assert response.body == _PCM


def test_mp3_requested_and_available(monkeypatch):
    gateway = _gateway()
    monkeypatch.setattr("wyoming_openai_gateway.gateway.encode_mp3", lambda *a, **k: b"\xff\xfbabc")
    response = gateway._audio_response(_PCM, _AUDIO_PARAMS, SpeechRequest(input="hi", response_format="mp3"))
    assert response.media_type == "audio/mpeg"
    assert response.body == b"\xff\xfbabc"


def test_mp3_requested_but_encoder_missing(monkeypatch):
    """Open Notebook always asks for mp3 — a build without lameenc must still answer with audio."""
    gateway = _gateway()
    monkeypatch.setattr("wyoming_openai_gateway.gateway.encode_mp3", lambda *a, **k: None)
    response = gateway._audio_response(_PCM, _AUDIO_PARAMS, SpeechRequest(input="hi", response_format="mp3"))
    assert response.media_type == "audio/wav"
    assert response.body[:4] == b"RIFF"
    assert response.headers.get("X-Response-Audio-Format") == "wav"


def test_unknown_format_falls_back_to_wav():
    gateway = _gateway()
    response = gateway._audio_response(_PCM, _AUDIO_PARAMS, SpeechRequest(input="hi", response_format="opus"))
    assert response.media_type == "audio/wav"
    assert response.body[:4] == b"RIFF"


# ── voice resolution ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_voice_is_replaced_by_an_installed_voice():
    """Open Notebook sends 'alloy'; wyoming-piper would try to download that."""
    gateway = _gateway()
    voice = await gateway._resolve_voice(_FakeClient(_info(["en_US-lessac-medium"])), "alloy")
    assert voice == "en_US-lessac-medium"


@pytest.mark.asyncio
async def test_known_voice_is_passed_through():
    gateway = _gateway()
    voice = await gateway._resolve_voice(_FakeClient(_info(["en_US-lessac-medium"])), "en_US-lessac-medium")
    assert voice == "en_US-lessac-medium"


@pytest.mark.asyncio
async def test_no_voice_means_server_default():
    gateway = _gateway()
    assert await gateway._resolve_voice(_FakeClient(_info(["en_US-lessac-medium"])), None) is None


@pytest.mark.asyncio
async def test_voice_fallback_can_be_disabled():
    gateway = _gateway(voice_fallback=False)
    assert await gateway._resolve_voice(_FakeClient(_info(["en_US-lessac-medium"])), "alloy") == "alloy"


@pytest.mark.asyncio
async def test_uninstalled_voices_are_not_offered():
    gateway = _gateway()
    voice = await gateway._resolve_voice(_FakeClient(_info(["alloy"], installed=False)), "alloy")
    assert voice == "alloy"  # nothing installed, so the request is passed through


@pytest.mark.asyncio
async def test_voice_resolution_survives_describe_failure():
    """A server that will not answer Describe must not break synthesis."""
    gateway = _gateway()
    client = _FakeClient(None, error=WyomingConnectionError("boom"))
    assert await gateway._resolve_voice(client, "alloy") == "alloy"


# ── model listing ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_models_lists_both_services():
    models = await _gateway().list_models()
    assert [m.id for m in models.data] == [TTS_MODEL_ID, STT_MODEL_ID]
    assert models.object == "list"


@pytest.mark.asyncio
async def test_models_lists_tts_only():
    gateway = _gateway(stt_host=None, stt_port=None)
    assert [m.id for m in (await gateway.list_models()).data] == [TTS_MODEL_ID]


@pytest.mark.asyncio
async def test_models_lists_stt_only():
    gateway = _gateway(tts_host=None, tts_port=None)
    assert [m.id for m in (await gateway.list_models()).data] == [STT_MODEL_ID]


# ── STT input handling ────────────────────────────────────────────────────────


class _FakeUpload:
    def __init__(self, data: bytes, filename: str | None = "audio.wav") -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


@pytest.mark.asyncio
async def test_undecodable_upload_is_415():
    """Not an ffmpeg/mp3 problem on our side — the client sent junk."""
    gateway = _gateway(ffmpeg_fallback=False)
    with pytest.raises(HTTPException) as exc:
        await gateway.transcribe_audio(_FakeUpload(b"not audio at all"), TranscriptionRequest(model="whisper-1"))
    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_empty_upload_is_400():
    gateway = _gateway()
    with pytest.raises(HTTPException) as exc:
        await gateway.transcribe_audio(_FakeUpload(b""), TranscriptionRequest(model="whisper-1"))
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    ("filename", "suffix"),
    [
        ("speech.m4a", ".m4a"),
        ("/tmp/weird/name.opus", ".opus"),
        ("no-extension", ""),
        ("trailing.", ""),
        (None, ""),
        ("too_long_extension.webmabcdeff", ""),
    ],
)
def test_filename_suffix(filename, suffix):
    assert _filename_suffix(filename) == suffix
