"""ASGI app factory for Wyoming-OpenAI-Gateway."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from . import __version__
from .config import Settings, normalize_prefix, openai_base_urls, route_prefixes
from .errors import WyomingConnectionError
from .gateway import OpenAIGateway
from .openai_models import (
    SpeechRequest,
    TranscriptionRequest,
)
from .wyoming_client import WyomingStreamClient

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings._parse()

    gateway = OpenAIGateway(settings)
    prefix = normalize_prefix(settings.prefix)
    prefixes = route_prefixes(prefix)

    def configured_services() -> dict[str, tuple[str, int]]:
        services: dict[str, tuple[str, int]] = {}
        if settings.tts_host and settings.tts_port:
            services["tts"] = (settings.tts_host, settings.tts_port)
        if settings.stt_host and settings.stt_port:
            services["stt"] = (settings.stt_host, settings.stt_port)
        return services

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = configured_services()
        base_urls = openai_base_urls(settings.host, settings.port, prefixes)

        log.info(
            "Starting Wyoming-OpenAI-Gateway v%s — Services: %s",
            __version__,
            ", ".join(f"{name} at {host}:{port}" for name, (host, port) in services.items())
            or "none configured",
        )
        log.info("OpenAI-compatible base URLs: %s", ", ".join(base_urls))

        # Validate connectivity on startup (non-fatal)
        for name, (host, port) in services.items():
            try:
                async with WyomingStreamClient(host, port) as client:
                    await client.describe()
                log.info("%s server at %s:%s is reachable", name.upper(), host, port)
            except WyomingConnectionError:
                log.warning(
                    "%s server at %s:%s is unreachable — will retry on each request",
                    name.upper(),
                    host,
                    port,
                )
            except Exception:
                log.warning("Unexpected error during %s connectivity check", name.upper(), exc_info=True)

        if "tts" not in services:
            log.info("TTS is disabled")
        if "stt" not in services:
            log.info("STT is disabled")

        yield
        log.info("Shutting down Wyoming-OpenAI-Gateway")

    app = FastAPI(
        title="Wyoming-OpenAI-Gateway",
        description="A gateway that exposes local Wyoming protocol services via OpenAI-compatible API endpoints.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz():
        services = configured_services()
        if not services:
            raise HTTPException(status_code=503, detail="No Wyoming services configured")

        for name, (host, port) in services.items():
            try:
                async with WyomingStreamClient(host, port) as client:
                    await client.describe()
            except WyomingConnectionError as e:
                raise HTTPException(status_code=503, detail=f"{name.upper()} not ready: {e}") from e

        return {"status": "ready", "services": sorted(services)}

    # --- OpenAI-compatible routes ---

    api = APIRouter()

    @api.get("/models")
    async def list_models():
        """Model listing every OpenAI-compatible client probes before use.

        Open Notebook's "Test connection" and "Discover Models" both call
        GET {base_url}/models, so without this endpoint those requests 404.
        """
        return await gateway.list_models()

    @api.get("/voices")
    @api.get("/audio/voices")
    async def list_voices():
        return await gateway.list_voices()

    @api.post("/audio/speech")
    async def text_to_speech(request: SpeechRequest):
        return await gateway.text_to_speech(request)

    @api.post("/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(..., description="Audio file to transcribe"),
        model: str = Form("whisper-1", description="Model identifier"),
        language: str | None = Form(None, description="Language code (e.g. 'en')"),
        prompt: str | None = Form(None, description="Optional context prompt"),
        response_format: str = Form("json", description="Response format (json/text/verbose_json)"),
        temperature: float = Form(0.0, description="Sampling temperature"),
    ):
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        result = await gateway.transcribe_audio(file, request, is_translation=False)
        if response_format.strip().lower() == "text":
            return PlainTextResponse(result.text, media_type="text/plain")
        return result

    @api.post("/audio/translations")
    async def create_translation(
        file: UploadFile = File(..., description="Audio file to translate"),
        model: str = Form("whisper-1", description="Model identifier"),
        language: str | None = Form(None, description="Language code (e.g. 'en')"),
        prompt: str | None = Form(None, description="Optional context prompt"),
        response_format: str = Form("json", description="Response format (json/text)"),
        temperature: float = Form(0.0, description="Sampling temperature"),
    ):
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        result = await gateway.transcribe_audio(file, request, is_translation=True)
        if response_format.strip().lower() == "text":
            return PlainTextResponse(result.text, media_type="text/plain")
        return result

    # The same routes are served under the configured prefix, /v1 and the root:
    # clients append /models, /audio/speech and /audio/transcriptions to whatever
    # base URL they were given and never add /v1 themselves, so http://host:PORT,
    # http://host:PORT/ and http://host:PORT/v1 must all work as a base URL.
    for candidate in prefixes:
        app.include_router(api, prefix=candidate, include_in_schema=(candidate == prefix))

    @app.get("/")
    async def index():
        """Pointer at the OpenAI-compatible base URLs, so a 404 in the browser
        is never the only thing a user sees."""
        return {
            "name": "Wyoming-OpenAI-Gateway",
            "version": __version__,
            "openai_base_urls": openai_base_urls(settings.host, settings.port, prefixes),
            "docs": "/docs",
            "health": "/healthz",
        }

    return app
