"""Tests for audio utility functions."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile

import pytest

from wyoming_openai_gateway.audio_utils import (
    assemble_wav,
    create_wav_header,
    decode_to_pcm,
    encode_mp3,
    mp3_encoder,
)
from wyoming_openai_gateway.errors import AudioGenerationError


def test_create_wav_header():
    """Test WAV header creation."""
    header = create_wav_header(sample_rate=22050, bits_per_sample=16, channels=1)
    assert len(header) == 44  # Standard WAV header is 44 bytes
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    # Parse format fields
    audio_format, num_channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
        "<HHLLHH", header[20:36]
    )
    assert audio_format == 1  # PCM
    assert num_channels == 1
    assert sample_rate == 22050
    assert bits == 16


def test_create_wav_header_stereo():
    """Test WAV header for stereo 48kHz."""
    header = create_wav_header(sample_rate=48000, bits_per_sample=16, channels=2)
    assert header[:4] == b"RIFF"
    _, _, _, _, _, audio_format, num_channels, sample_rate, _, _, bits = struct.unpack(
        "<4sL4s4sLHHLLHH", header[:36]
    )
    assert audio_format == 1
    assert num_channels == 2
    assert sample_rate == 48000
    assert bits == 16


def test_assemble_wav():
    """Test assembling a WAV file from audio chunks."""
    # Create some dummy PCM audio data
    audio_chunks = [b"\x00\x00" * 100]  # 100 samples of 16-bit silence
    wav_bytes = assemble_wav(audio_chunks, rate=16000, width=2, channels=1)
    assert len(wav_bytes) > 44  # Header + data
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    # Data chunk ID is at byte 36
    assert wav_bytes[36:40] == b"data"
    # Data size at byte 40 should be the PCM data length
    data_size = int.from_bytes(wav_bytes[40:44], "little")
    assert data_size == 200  # 100 samples * 2 bytes
    # Total file size should be header (44) + data
    assert len(wav_bytes) == 44 + data_size


# ── decoding uploaded audio (STT input) ──────────────────────────────────────


def _wav_bytes(rate: int = 22050, channels: int = 1, frames: int = 220) -> bytes:
    return assemble_wav([b"\x00\x02" * frames], rate=rate, width=2, channels=channels)


def test_decode_to_pcm_reads_wav():
    """Whatever the client sends, the Wyoming server is handed 16 kHz mono 16-bit."""
    pcm, rate, width, channels = decode_to_pcm(_wav_bytes(rate=22050, channels=2))
    assert rate == 16000
    assert width == 2
    assert channels == 1
    assert pcm  # decoded samples


def test_decode_to_pcm_empty_input():
    with pytest.raises(AudioGenerationError):
        decode_to_pcm(b"")


def test_decode_to_pcm_junk_without_ffmpeg():
    with pytest.raises(AudioGenerationError) as exc:
        decode_to_pcm(b"\x00\x01\x02totally not audio", ffmpeg_fallback=False)
    assert "Unsupported audio format" in str(exc.value)
    assert "FFMPEG_FALLBACK" in str(exc.value)


def test_mp3_round_trip():
    """What Open Notebook sends (mp3) must decode, when an encoder is available to make one."""
    mp3 = encode_mp3(b"\x00\x02" * 1600, rate=16000, channels=1)
    if mp3 is None:
        pytest.skip("lameenc not installed")

    pcm, rate, width, channels = decode_to_pcm(mp3, ffmpeg_fallback=False)
    assert rate == 16000
    assert width == 2
    assert channels == 1
    assert pcm


def test_encode_mp3_matches_encoder_availability():
    """Missing lameenc must mean None (WAV fallback), never an exception."""
    encoded = encode_mp3(b"\x00\x02" * 1600, rate=16000, channels=1)
    if mp3_encoder() is None:
        assert encoded is None
    else:
        assert encoded
        # MPEG frame sync, or an ID3 tag in front of the first frame.
        assert encoded[:3] == b"ID3" or encoded[0] == 0xFF


def test_encode_mp3_rejects_bad_parameters():
    """lameenc blowing up must degrade to WAV, not a 500."""
    assert encode_mp3(b"\x00\x02" * 100, rate=1234, channels=7) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_fallback_decodes_containers_miniaudio_cannot():
    """m4a/AAC is what Open Notebook posts for video-derived sources."""
    temp_path = os.path.join(tempfile.mkdtemp(), "speech.m4a")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2:sample_rate=44100",
            "-ac", "2", "-c:a", "aac", "-y", temp_path,
        ],
        capture_output=True,
        check=True,
    )
    with open(temp_path, "rb") as handle:
        m4a_bytes = handle.read()
    assert m4a_bytes

    pcm, rate, width, channels = decode_to_pcm(m4a_bytes, source_hint=".m4a")
    assert rate == 16000  # resampled for the ASR server
    assert width == 2
    assert channels == 1
    assert len(pcm) > 1600 * 2 * 0.1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_fallback_reports_garbage_as_unsupported():
    with pytest.raises(AudioGenerationError):
        decode_to_pcm(b"\x00\x01\x02totally not audio", source_hint=".m4a")
