"""Audio processing utilities for Wyoming-OpenAI-Gateway."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave

import miniaudio

from .errors import AudioGenerationError

log = logging.getLogger(__name__)

# Formats a client may ask for that this gateway can produce natively from the
# PCM Wyoming always sends.
PCM_FORMATS = frozenset({"pcm", "pcm16", "s16le", "raw"})
WAV_FORMATS = frozenset({"wav", "wave"})
MP3_FORMATS = frozenset({"mp3", "mpeg"})

# PCM is normalised to this rate on the way in and out of the decoders — it is
# what Wyoming ASR servers transcribe at.
ASR_TARGET_RATE = 16000

_MP3_ENCODER: object | None = None
_MP3_ENCODER_RESOLVED = False


def create_wav_header(sample_rate: int, bits_per_sample: int, channels: int) -> bytes:
    """Create a WAV header with placeholder sizes for streaming.

    Uses 0xFFFFFFFF for chunk sizes to indicate unknown total size,
    which is acceptable for streaming scenarios.
    """
    chunk_size = 0xFFFFFFFF
    final_data_size = 0xFFFFFFFF
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sL4s4sLHHLLHH4sL",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # subchunk1 size (PCM)
        1,  # audio format (1 = PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        final_data_size,
    )


def assemble_wav(audio_chunks: list[bytes], rate: int, width: int, channels: int) -> bytes:
    """Assemble a complete WAV file from audio chunks using the wave module."""
    complete_audio_data = b"".join(audio_chunks)
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_writer:
            wav_writer.setframerate(rate)
            wav_writer.setsampwidth(width)
            wav_writer.setnchannels(channels)
            wav_writer.writeframes(complete_audio_data)
        return wav_io.getvalue()


def mp3_encoder() -> type | None:
    """Return the optional lameenc encoder class, or None when it is not installed.

    Open Notebook asks for ``response_format=mp3``; WAV is what Wyoming gives us,
    so mp3 support depends on the optional ``lameenc`` dependency
    (``pip install "wyoming-openai-gateway[mp3]"``).
    """
    global _MP3_ENCODER, _MP3_ENCODER_RESOLVED
    if not _MP3_ENCODER_RESOLVED:
        try:
            import lameenc

            _MP3_ENCODER = lameenc.Encoder
        except ImportError:
            _MP3_ENCODER = None
            log.debug("lameenc not installed — mp3 responses fall back to WAV")
        _MP3_ENCODER_RESOLVED = True
    return _MP3_ENCODER


def encode_mp3(pcm: bytes, rate: int, channels: int) -> bytes | None:
    """Encode 16-bit PCM to MP3, or None if that is not possible here.

    Returns None when lameenc is missing or refuses the parameters, so callers
    can fall back to WAV instead of failing a request Open Notebook already made.
    """
    encoder_cls = mp3_encoder()
    if encoder_cls is None:
        return None

    try:
        encoder = encoder_cls()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(rate)
        encoder.set_channels(channels)
        encoder.set_quality(2)  # 0 (fastest) .. 9 (slowest)
        mp3 = encoder.encode(pcm)
        mp3 += encoder.flush()
        return bytes(mp3)
    except Exception:
        log.warning("Could not encode MP3 at %d Hz / %d ch — falling back to WAV", rate, channels, exc_info=True)
        return None


def decode_to_pcm(
    raw: bytes, ffmpeg_fallback: bool = True, source_hint: str = ""
) -> tuple[bytes, int, int, int]:
    """Decode container/codec audio to raw PCM.

    Returns ``(pcm_bytes, rate, width, channels)`` — always 16 kHz / 16-bit / mono,
    the format Wyoming ASR servers expect. Supports everything miniaudio decodes
    (WAV, MP3, FLAC, OGG/Vorbis); when miniaudio cannot handle the container —
    typically m4a/AAC or webm/Opus from video sources — an ffmpeg subprocess is
    used if ``ffmpeg_fallback`` is enabled and ffmpeg is on PATH.

    ``source_hint`` is the upload's filename extension, which only helps ffmpeg
    pick a demuxer for containers it cannot sniff on its own.

    Raises:
        AudioGenerationError: no decoder could handle the audio.
    """
    if not raw:
        raise AudioGenerationError("No audio data received")

    try:
        # Normalised to what Wyoming ASR servers want anyway: 16 kHz / 16-bit / mono.
        # miniaudio resamples and downmixes for us, which also keeps the Wyoming
        # upload small for long recordings. (Passing the native rate instead would
        # be harmless — the ASR server converts too — but costs 3x the bandwidth.)
        result = miniaudio.decode(
            raw,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=ASR_TARGET_RATE,
        )
        log.debug(
            "Decoded with miniaudio: %d Hz, 16-bit, %d ch, %d PCM bytes",
            result.sample_rate,
            result.nchannels,
            len(result.samples) * 2,
        )
        return result.samples.tobytes(), result.sample_rate, 2, result.nchannels
    except Exception as miniaudio_error:
        log.debug("miniaudio could not decode audio: %s", miniaudio_error)

    if ffmpeg_fallback and shutil.which("ffmpeg"):
        decoded = _decode_with_ffmpeg(raw, source_hint)
        if decoded is not None:
            pcm_bytes, rate, width, channels = decoded
            log.debug(
                "Decoded with ffmpeg: %d Hz, %d-bit, %d ch, %d PCM bytes",
                rate,
                width * 8,
                channels,
                len(pcm_bytes),
            )
            return pcm_bytes, rate, width, channels
    elif not ffmpeg_fallback:
        log.debug("ffmpeg fallback disabled (FFMPEG_FALLBACK=false)")

    supported = "WAV, MP3, FLAC, OGG"
    hint = ""
    if not (ffmpeg_fallback and shutil.which("ffmpeg")):
        hint = (
            " Install ffmpeg in this container (or set FFMPEG_FALLBACK=false) to accept "
            "m4a/AAC and webm/Opus audio."
        )
    raise AudioGenerationError(
        f"Unsupported audio format or corrupt file. Decodable here: {supported}.{hint}"
    )


def _decode_with_ffmpeg(raw: bytes, source_hint: str = "") -> tuple[bytes, int, int, int] | None:
    """Decode arbitrary audio to 16-bit PCM with ffmpeg. None when ffmpeg fails.

    STT servers want narrowband mono audio anyway, so ffmpeg resamples while
    decoding — that also keeps the Wyoming upload small for long recordings.
    The WAV container carries the rate/channels back to us.

    The input is spooled to a temp file rather than piped: containers such as
    M4A/AAC keep their index at the end of the file and cannot be demuxed from a
    non-seekable pipe.

    Returns ``(pcm_bytes, rate, width, channels)``.
    """
    suffix = source_hint if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", source_hint) else ""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(raw)

        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                temp_path,
                "-map",
                "a:0",
                "-ac",
                "1",
                "-ar",
                str(ASR_TARGET_RATE),
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        log.warning("ffmpeg fallback failed to start", exc_info=True)
        return None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                log.debug("Could not remove temp file %s", temp_path, exc_info=True)

    if completed.returncode != 0 or not completed.stdout:
        log.warning(
            "ffmpeg could not decode audio (exit %s): %s",
            completed.returncode,
            completed.stderr.decode(errors="replace").strip()[:300],
        )
        return None

    # ffmpeg wrote a PCM WAV, which carries the rate/channels back to us.
    try:
        with wave.open(io.BytesIO(completed.stdout), "rb") as wav_reader:
            return (
                wav_reader.readframes(wav_reader.getnframes()),
                wav_reader.getframerate(),
                wav_reader.getsampwidth(),
                wav_reader.getnchannels(),
            )
    except (EOFError, wave.Error):
        log.warning("ffmpeg produced an unreadable WAV stream", exc_info=True)
        return None
