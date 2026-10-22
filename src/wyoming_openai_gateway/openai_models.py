"""Pydantic models for OpenAI-compatible request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request body."""

    input: str = Field(..., description="The text to synthesize")
    model: str = Field("wyoming", description="Model identifier (any value; see /models)")
    voice: str | None = Field(
        None,
        description="Voice ID from /audio/voices. Unknown or missing names use the server's default voice.",
    )
    response_format: str = Field(
        "wav",
        alias="response_format",
        description="Audio format: wav (default), mp3 (needs the [mp3] extra) or pcm. Unsupported names return WAV.",
    )
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech rate multiplier")
    stream: bool = Field(False, description="Stream the response (always WAV)")

    model_config = ConfigDict(populate_by_name=True)


class ModelInfo(BaseModel):
    """OpenAI-compatible model entry. Open Notebook reads `id` and classifies the
    model type from its name, so the ids exposed here deliberately contain
    'tts' (text-to-speech) and 'whisper' (speech-to-text)."""

    id: str = Field(..., description="Model identifier to send back in requests")
    object: str = Field("model", description="Object type, always 'model'")
    created: int = Field(0, description="Creation timestamp (unused, always 0)")
    owned_by: str = Field("wyoming", description="Who provides the model")
    capability: str | None = Field(None, description="'tts' or 'stt' — non-standard hint")


class ModelsResponse(BaseModel):
    """OpenAI-compatible model list, the shape GET /models must return."""

    object: str = Field("list", description="Object type, always 'list'")
    data: list[ModelInfo] = Field(default_factory=list, description="Available models")


class Voice(BaseModel):
    """OpenAI-compatible voice representation."""

    id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Human-readable voice name")
    description: str | None = Field(None, description="Voice description")
    languages: list[str] | None = Field(None, description="Supported language codes")


class VoicesResponse(BaseModel):
    """OpenAI-compatible voices list response."""

    voices: list[Voice] = Field(..., description="List of available voices")


class TranscriptionRequest(BaseModel):
    """OpenAI-compatible transcription/translation request body (non-file fields)."""

    model: str = Field("whisper-1", description="Model identifier")
    language: str | None = Field(None, description="Language code (e.g. 'en')")
    prompt: str | None = Field(None, description="Optional context prompt")
    response_format: str = Field("json", alias="response_format", description="Response format (json/text)")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Sampling temperature")
    timestamp_granularities: list[str] | None = Field(
        None, alias="timestamp_granularities",
        description="Timestamp granularities (segment, word)",
    )

    model_config = ConfigDict(populate_by_name=True)


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str = Field(..., description="Transcribed text")


class TranslationResponse(BaseModel):
    """OpenAI-compatible translation response."""

    text: str = Field(..., description="Translated text")
