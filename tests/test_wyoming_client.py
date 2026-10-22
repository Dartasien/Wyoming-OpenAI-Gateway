"""Tests for the Wyoming event layer (what actually goes over the socket)."""

from __future__ import annotations

import dataclasses

from wyoming.audio import AudioStop
from wyoming.tts import Synthesize

from wyoming_openai_gateway.wyoming_client import WyomingStreamClient

_AUDIO_STOP = AudioStop().event()


class _FakeTcp:
    """Minimal stand-in for wyoming's AsyncTcpClient."""

    def __init__(self) -> None:
        self.written = []

    async def write_event(self, event) -> None:
        self.written.append(event)

    async def read_event(self):
        return _AUDIO_STOP


def _client_with_fake_transport() -> tuple[WyomingStreamClient, _FakeTcp]:
    client = WyomingStreamClient("127.0.0.1", 10200)
    fake = _FakeTcp()
    client._client = fake
    return client, fake


async def _drain(client: WyomingStreamClient, text: str, voice, speed: float) -> None:
    async for _ in client.synthesize_stream(text=text, voice=voice, speed=speed):
        pass


async def test_voice_is_sent_when_given():
    client, fake = _client_with_fake_transport()
    await _drain(client, "hello", "en_US-lessac-medium", 1.0)
    assert fake.written[0].data == {"text": "hello", "voice": {"name": "en_US-lessac-medium"}}


async def test_voice_is_omitted_when_not_given():
    """No voice key means the Wyoming server applies its own default."""
    client, fake = _client_with_fake_transport()
    await _drain(client, "hello", None, 1.0)
    assert "voice" not in fake.written[0].data

    client, fake = _client_with_fake_transport()
    await _drain(client, "hello", "   ", 1.0)
    assert "voice" not in fake.written[0].data


async def test_speed_does_not_break_synthesize():
    """wyoming's Synthesize has no speech-rate field — asking for one must not crash.

    A test so that the day wyoming gains the field, the forwarding branch here is
    already covered rather than silently dead.
    """
    client, fake = _client_with_fake_transport()
    await _drain(client, "hello", None, 1.5)
    event_data = fake.written[0].data
    assert event_data["text"] == "hello"
    if "speech_rate" in {f.name for f in dataclasses.fields(Synthesize)}:
        assert event_data.get("speech_rate") == 1.5
    else:
        assert "speech_rate" not in event_data
