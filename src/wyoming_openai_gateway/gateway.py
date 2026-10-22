"""Core translation layer — converts OpenAI API requests to Wyoming protocol."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator

from fastapi import HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from wyoming.audio import AudioChunk, AudioStart, AudioStop

from .audio_utils import (
    MP3_FORMATS,
    PCM_FORMATS,
    assemble_wav,
    create_wav_header,
    decode_to_pcm,
    encode_mp3,
)
from .config import Settings
from .errors import AudioGenerationError, WyomingConnectionError, WyomingProtocolError
from .openai_models import (
    ModelInfo,
    ModelsResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationResponse,
    Voice,
    VoicesResponse,
)
from .wyoming_client import WyomingStreamClient

log = logging.getLogger(__name__)


# Model ids advertised by GET /models. Open Notebook (and esperanto's discovery)
# classify a model's modality from its name: "tts" -> text_to_speech and
# "whisper" -> speech_to_text, so these ids are chosen to be sorted correctly.
TTS_MODEL_ID = "wyoming-tts"
STT_MODEL_ID = "whisper-1"


def _read_upload(file: UploadFile) -> bytes:
    """Read an uploaded file into memory (size-capped by the caller's proxy)."""
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No audio data received")
    return raw


def _filename_suffix(filename: str | None) -> str:
    """Upload extension, e.g. '.m4a'. Only used to help ffmpeg pick a demuxer."""
    match = re.search(r"\.([A-Za-z0-9]{1,8})$", filename or "")
    return match.group(0) if match else ""


class OpenAIGateway:
    """Translates OpenAI-compatible API requests into Wyoming protocol calls."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def list_voices(self) -> VoicesResponse:
        """List all available TTS voices from the Wyoming TTS server."""
        if self._settings.tts_host is None or self._settings.tts_port is None:
            raise HTTPException(status_code=503, detail="TTS not configured")

        try:
            async with WyomingStreamClient(
                self._settings.tts_host, self._settings.tts_port
            ) as client:
                info = await client.describe()
        except WyomingConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))

        voices: list[Voice] = []
        for tts_program in info.tts:
            if tts_program.installed and tts_program.voices:
                for voice in tts_program.voices:
                    voices.append(
                        Voice(
                            id=voice.name,
                            name=voice.description or voice.name,
                            description=(
                                voice.attribution.name if voice.attribution else None
                            ),
                            languages=voice.languages,
                        )
                    )

        if not voices:
            return VoicesResponse(voices=[])
        return VoicesResponse(voices=voices)

    async def list_models(self) -> ModelsResponse:
        """List available models in the shape OpenAI clients expect from GET /models.

        Open Notebook's connection test and "Discover Models" both call this and
        sort models into the text-to-speech / speech-to-text slots by id, so one
        id per configured service is exposed instead of one per voice (voices are
        selected separately, through the client's Voice ID field).
        """
        data: list[ModelInfo] = []
        if self._settings.tts_host and self._settings.tts_port:
            data.append(
                ModelInfo(
                    id=TTS_MODEL_ID,
                    owned_by=f"wyoming:{self._settings.tts_host}:{self._settings.tts_port}",
                    capability="tts",
                )
            )
        if self._settings.stt_host and self._settings.stt_port:
            data.append(
                ModelInfo(
                    id=STT_MODEL_ID,
                    owned_by=f"wyoming:{self._settings.stt_host}:{self._settings.stt_port}",
                    capability="stt",
                )
            )
        return ModelsResponse(object="list", data=data)

    async def _resolve_voice(self, client: WyomingStreamClient, requested: str | None) -> str | None:
        """Map a requested voice onto one the Wyoming TTS server actually has.

        OpenAI clients send OpenAI voice names ("alloy", "nova") while a Wyoming
        server knows its own ids ("en_US-lessac-medium"). Forwarding an unknown
        name is not harmless — wyoming-piper treats it as a voice to download and
        the synthesis fails — so an unknown name is replaced by the first
        installed voice. No voice at all means "use the server's own default".
        """
        if not requested:
            return None
        if not self._settings.voice_fallback:
            return requested

        try:
            info = await client.describe()
        except (WyomingConnectionError, WyomingProtocolError) as e:
            log.debug("Could not describe TTS server to resolve voice: %s", e)
            return requested

        installed = [
            voice.name
            for program in info.tts
            if program.installed
            for voice in (program.voices or [])
            if voice.name and voice.installed
        ]
        if not installed:
            log.debug("TTS server reported no voices — passing '%s' through", requested)
            return requested

        if requested in installed:
            return requested

        if requested:
            log.warning(
                "Voice '%s' is not installed on the Wyoming TTS server — using '%s' (available: %s)",
                requested,
                installed[0],
                ", ".join(installed),
            )
        return installed[0]

    def _audio_response(self, pcm: bytes, audio_params: dict, request: SpeechRequest) -> Response:
        """Wrap Wyoming's PCM in the container the client asked for.

        WAV is native and always available. MP3 needs the optional ``[mp3]``
        dependency; when it is missing the WAV bytes are returned anyway, which
        keeps clients like Open Notebook working (they sniff the audio). PCM is
        handed back raw.
        """
        fmt = (request.response_format or "wav").strip().lower()
        rate, width, channels = audio_params["rate"], audio_params["width"], audio_params["channels"]

        if fmt in PCM_FORMATS:
            log.info("Generated %.1f KB of raw PCM audio for voice '%s'", len(pcm) / 1024, request.voice)
            return Response(content=pcm, media_type="audio/pcm")

        if fmt in MP3_FORMATS and width == 2:
            mp3 = encode_mp3(pcm, rate=rate, channels=channels)
            if mp3:
                log.info("Generated %.1f KB of MP3 audio for voice '%s'", len(mp3) / 1024, request.voice)
                return Response(content=mp3, media_type="audio/mpeg")
            return Response(
                content=assemble_wav([pcm], rate=rate, width=width, channels=channels),
                media_type="audio/wav",
                headers={"X-Response-Audio-Format": "wav"},
            )

        if fmt not in ("wav", "wave"):
            log.warning("response_format '%s' cannot be produced — returning WAV", fmt)

        wav_bytes = assemble_wav([pcm], rate=rate, width=width, channels=channels)
        log.info("Generated %.1f KB of WAV audio for voice '%s'", len(wav_bytes) / 1024, request.voice)
        return Response(content=wav_bytes, media_type="audio/wav")

    async def _complete_speech(self, request: SpeechRequest) -> Response:
        """Generate complete (non-streaming) speech response."""
        try:
            async with WyomingStreamClient(
                self._settings.tts_host, self._settings.tts_port
            ) as client:
                voice = await self._resolve_voice(client, request.voice)
                audio_params, audio_data = await client.synthesize_complete(
                    text=request.input,
                    voice=voice,
                    speed=request.speed,
                )
        except WyomingConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except WyomingProtocolError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return self._audio_response(audio_data, audio_params, request)

    async def text_to_speech(self, request: SpeechRequest) -> Response:
        """Convert text to speech via Wyoming protocol."""
        if self._settings.tts_host is None or self._settings.tts_port is None:
            raise HTTPException(status_code=503, detail="TTS not configured")

        if request.stream:
            return await self._stream_speech(request)
        return await self._complete_speech(request)

    async def _stream_speech(self, request: SpeechRequest) -> StreamingResponse:
        """Generate streaming speech response.

        Streaming responses are always WAV: the container is written while the
        audio is still arriving, so response_format cannot be honoured.
        """
        fmt = (request.response_format or "wav").strip().lower()
        if fmt not in ("wav", "wave"):
            log.warning("stream=true ignores response_format '%s' — streaming WAV", fmt)
        return StreamingResponse(
            self._generate_audio_stream(request),
            media_type="audio/wav",
            headers={"Connection": "close"},
        )

    async def _generate_audio_stream(
        self, request: SpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Async generator that yields WAV header then audio chunks."""
        try:
            async with WyomingStreamClient(
                self._settings.tts_host, self._settings.tts_port
            ) as client:
                wav_header_sent = False
                voice = await self._resolve_voice(client, request.voice)
                async for event in client.synthesize_stream(
                    text=request.input,
                    voice=voice,
                    speed=request.speed,
                ):
                    if not wav_header_sent and AudioStart.is_type(event.type):
                        audio_start = AudioStart.from_event(event)
                        bits_per_sample = audio_start.width * 8
                        header = create_wav_header(
                            sample_rate=audio_start.rate,
                            bits_per_sample=bits_per_sample,
                            channels=audio_start.channels,
                        )
                        yield header
                        wav_header_sent = True
                        log.debug("Streaming: WAV header sent")
                    elif wav_header_sent and AudioChunk.is_type(event.type):
                        yield AudioChunk.from_event(event).audio
                    elif AudioStop.is_type(event.type):
                        log.debug("Streaming: AudioStop received, finishing")
                        break
        except WyomingConnectionError as e:
            log.error("Streaming failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e))
        except WyomingProtocolError as e:
            log.error("Streaming failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    async def transcribe_audio(
        self,
        file: UploadFile,
        request: TranscriptionRequest,
        is_translation: bool = False,
    ) -> TranscriptionResponse | TranslationResponse:
        """Transcribe or translate audio using Wyoming ASR protocol."""
        if self._settings.stt_host is None or self._settings.stt_port is None:
            raise HTTPException(status_code=503, detail="STT not configured")

        log.info(
            "Processing STT request: model=%s, language=%s, format=%s, translate=%s",
            request.model,
            request.language,
            request.response_format,
            is_translation,
        )

        raw = _read_upload(file)

        try:
            pcm_bytes, rate, width, channels = decode_to_pcm(
                raw,
                ffmpeg_fallback=self._settings.ffmpeg_fallback,
                source_hint=_filename_suffix(file.filename),
            )
        except AudioGenerationError as e:
            raise HTTPException(status_code=415, detail=str(e)) from e

        log.debug(
            "Audio input: %d bytes, %d Hz, %d-bit, %d channels",
            len(pcm_bytes), rate, width * 8, channels,
        )

        try:
            async with WyomingStreamClient(
                self._settings.stt_host, self._settings.stt_port
            ) as client:
                text = await client.transcribe(
                    audio_bytes=pcm_bytes,
                    rate=rate,
                    width=width,
                    channels=channels,
                    language=request.language,
                    model_name=request.model,
                )
        except WyomingConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except WyomingProtocolError as e:
            raise HTTPException(status_code=500, detail=str(e))

        log.info("Transcription result (%d chars): %s", len(text), text[:100])

        if is_translation:
            return TranslationResponse(text=text)

        return TranscriptionResponse(text=text)
