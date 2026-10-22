# Wyoming-OpenAI-Gateway

A gateway that exposes local [Wyoming](https://github.com/rhasspy/wyoming) protocol services via OpenAI-compatible API endpoints. This allows any OpenAI-compatible client to use local Wyoming speech services (TTS **and** STT) without modification.

## Features

- **OpenAI-Compatible API** — Drop-in replacement for OpenAI's `/v1/models`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`, and `/v1/audio/voices` endpoints
- **Wyoming Protocol** — Connects to any Wyoming-compatible TTS service (Piper, Sherpa-ONNX, etc.) **and** any Wyoming-compatible STT service (faster-whisper, Sherpa-ONNX, etc.)
- **Model Discovery** — `GET /v1/models` lets clients such as [Open Notebook](https://github.com/lfnovo/open-notebook) find the gateway's TTS/STT models automatically
- **Streaming Support** — Real-time audio streaming for both TTS and STT protocols
- **Few System Dependencies** — Audio transcoding via `miniaudio` (bundled wheels, ~30x faster than subprocess-based ffmpeg); `ffmpeg` is used only for containers `miniaudio` cannot open, and is optional
- **Health Checks** — Kubernetes/Docker-ready `/healthz` and `/readyz` endpoints

## Quick Start

### Docker

```bash
docker run -d \
  --name wyoming-openai-gateway \
  -p 8555:8555 \
  -e TTS_HOST=your-tts-server \
  -e TTS_PORT=10200 \
  -e STT_HOST=your-stt-server \
  -e STT_PORT=10300 \
  ghcr.io/archmonger/wyoming-openai-gateway:latest
```

### Using pip

```bash
pip install wyoming-openai-gateway

TTS_HOST=127.0.0.1 TTS_PORT=10200 \
STT_HOST=127.0.0.1 STT_PORT=10300 \
wyoming-openai-gateway
```

### Using Docker Compose

```bash
wget https://raw.githubusercontent.com/archmonger/Wyoming-OpenAI-Gateway/main/compose.yml
TTS_HOST=127.0.0.1 TTS_PORT=10200 \
STT_HOST=127.0.0.1 STT_PORT=10300 \
docker compose up -d
```

### Building and sharing the image

```bash
docker build -t wyoming-openai-gateway:0.3.0 .                       # ~240 MB (ffmpeg included)
docker build -t wyoming-openai-gateway:0.3.0 --build-arg WITH_FFMPEG=false .   # ~110 MB, no m4a/webm uploads
```

To run it on another machine, push it to a registry (GHCR shown) and pull there:

```bash
docker tag wyoming-openai-gateway:0.3.0 ghcr.io/YOUR_GH_USERNAME/wyoming-openai-gateway:0.3.0
docker login ghcr.io -u YOUR_GH_USERNAME     # PAT with write:packages, or: gh auth token | docker login ghcr.io -u USER --password-stdin
docker push ghcr.io/YOUR_GH_USERNAME/wyoming-openai-gateway:0.3.0

# on the other machine (no login needed if the package is public)
docker pull ghcr.io/YOUR_GH_USERNAME/wyoming-openai-gateway:0.3.0
```

Pushing `ghcr.io/YOUR_GH_USERNAME/…` only works for a namespace you own — the repo's own workflow derives it from the repository owner. Without a registry:

```bash
docker save wyoming-openai-gateway:0.3.0 | gzip > wog.tar.gz
scp wog.tar.gz otherbox:~/
ssh otherbox 'gunzip -c wog.tar.gz | docker load'
```

## API Reference

### `GET /v1/models`

Lists the models this gateway exposes. Every OpenAI-compatible client probes this endpoint before it will use a server, so it is what connection tests hit first.

**Response:**
```json
{
  "object": "list",
  "data": [
    { "id": "wyoming-tts", "object": "model", "created": 0, "owned_by": "wyoming:127.0.0.1:10200", "capability": "tts" },
    { "id": "whisper-1", "object": "model", "created": 0, "owned_by": "wyoming:127.0.0.1:10300", "capability": "stt" }
  ]
}
```

Only configured services are listed (a TTS-only gateway returns just `wyoming-tts`). Pass the `id` back as the `model` field of a request — the value is advisory, Wyoming servers are selected by host/port.

### `GET /v1/voices` (also `GET /v1/audio/voices`)

Lists available TTS voices from the Wyoming server. `/v1/audio/voices` is the OpenAI spelling and is accepted as an alias.

**Response:**
```json
{
  "voices": [
    {
      "id": "en_US-lessac-medium",
      "name": "Lessac (Medium)",
      "languages": ["en_US"]
    }
  ]
}
```

### `POST /v1/audio/speech`

Generates speech audio from text.

**Request Body:**
```json
{
  "input": "Hello, world!",
  "model": "wyoming",
  "voice": "en_US-lessac-medium",
  "response_format": "wav",
  "speed": 1.0,
  "stream": false
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | (required) | The text to synthesize |
| `model` | string | `"wyoming"` | Model identifier (see `GET /v1/models`; any value works) |
| `voice` | string | *(server default)* | Voice ID from `/v1/voices`. An unknown name — e.g. an OpenAI voice like `alloy` — is mapped onto the first installed voice (see `VOICE_FALLBACK`) |
| `response_format` | string | `"wav"` | `wav`, `mp3` (needs the `[mp3]` extra) or `pcm`. Anything else returns WAV |
| `speed` | float | `1.0` | Speech rate multiplier |
| `stream` | bool | `false` | Enable streaming response (always WAV) |

**Response:** audio bytes — `audio/wav`, `audio/mpeg` or `audio/pcm` depending on `response_format`. If `mp3` is requested but this build has no MP3 encoder, WAV is returned instead (with an `X-Response-Audio-Format: wav` header) rather than an error.

### `POST /v1/audio/transcriptions`

Transcribes audio to text using a Wyoming STT server.

**Request:** `multipart/form-data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | (required) | Audio file (WAV, MP3, FLAC, OGG; plus anything `ffmpeg` can open, see `FFMPEG_FALLBACK`) |
| `model` | string | `"whisper-1"` | Model identifier (passed to Wyoming STT) |
| `language` | string | optional | Language code (e.g. `"en"`) |
| `response_format` | string | `"json"` | Response format (`json`, `verbose_json` or `text`) |
| `temperature` | float | `0.0` | Sampling temperature |

**Response:**
```json
{
  "text": "Your hands lay open in the long fresh grass."
}
```

### `POST /v1/audio/translations`

Same as transcriptions, but translates the audio to English text.

## Configuration

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_HOST` | *(none)* | Wyoming TTS server hostname (omit to disable TTS) |
| `TTS_PORT` | *(none)* | Wyoming TTS server port (required if TTS_HOST is set) |
| `STT_HOST` | *(none)* | Wyoming STT server hostname (omit to disable STT) |
| `STT_PORT` | *(none)* | Wyoming STT server port (required if STT_HOST is set) |
| `HOST` | `0.0.0.0` | Gateway HTTP listen address |
| `PORT` | `8555` | Gateway HTTP listen port |
| `PREFIX` | `/v1` | API route prefix — the routes are also served under `/v1` and the root, so any of `http://host:8555`, `http://host:8555/` or `http://host:8555/v1` works as a client base URL |
| `VOICE_FALLBACK` | `true` | Replace a voice the Wyoming TTS server does not have with one it does (OpenAI voice names such as `alloy` are otherwise treated as voices to download, and synthesis fails) |
| `FFMPEG_FALLBACK` | `true` | Use `ffmpeg` on PATH to decode containers `miniaudio` cannot open (m4a/AAC, webm/Opus — typical for audio extracted from video) |
| `DEBUG` | `false` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Logging level |

> **Validation:** At least one of `TTS_HOST` or `STT_HOST` must be defined. It is an error to set only a host without its corresponding port, or vice versa. If neither host is defined, the application will exit with an error.

MP3 output is an optional dependency:

```bash
pip install "wyoming-openai-gateway[mp3]"   # adds lameenc
```

Without it the gateway still answers `response_format=mp3` requests, but with WAV bytes.

## Open Notebook Integration

[Open Notebook](https://github.com/lfnovo/open-notebook) reaches speech models through [esperanto](https://github.com/lfnovo/esperanto)'s `openai_compatible` provider, which appends `/models`, `/audio/speech` and `/audio/transcriptions` to whatever base URL it is given — it never adds `/v1` itself. That is why every prefix is served here.

In Open Notebook: **Settings → AI Models → Add Model**, provider **OpenAI (compatible)**:

| Field | Value |
|-------|-------|
| Base URL | `http://host:8555/v1` (port/`/v1` both work; `…/v1` is what the connection test expects) |
| API key | anything — the gateway does not authenticate |
| TTS model | `wyoming-tts` |
| STT model | `whisper-1` |

Then **Test connection** should report the two discovered models, and each modality can be tested individually.

Notes for this pairing:

- **Voice ID** — Open Notebook sends OpenAI voice names (`alloy` by default). `VOICE_FALLBACK=true` maps unknown names onto a voice your Wyoming server actually has; set it to `false` to forward names verbatim, and put a real voice id (from `GET /v1/audio/voices`) in each speaker profile when you want a specific voice.
- **Formats** — Open Notebook requests `response_format=mp3`. Install the `[mp3]` extra for true MP3; without it WAV bytes are returned, which still play (they are concatenated and re-encoded by `ffmpeg` in Open Notebook's podcast pipeline).
- **Transcription** — audio extracted from video sources arrives as m4a/AAC or webm/Opus, which `miniaudio` cannot open. The Docker image ships `ffmpeg` for those; if you run a slimmer build, either add `ffmpeg` or set `FFMPEG_FALLBACK=false` to get a clear 415 instead.
- Uploads are normalised to 16 kHz mono 16-bit before they reach the Wyoming STT server, whatever the client sent.

## Architecture

```
┌─────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│                     │       │                      │       │                      │
│  OpenAI-compatible  │ HTTP  │  Wyoming-OpenAI-     │ TCP   │  Wyoming Protocol    │
│  Client             │──────>│  Gateway             │──────>│  Server (TTS/STT)    │
│  (curl, ST, Home    │       │                      │       │                      │
│  Assistant, etc.)   │       │  FastAPI → Wyoming   │       │  Piper / Whisper     │
│                     │<──────│  Protocol Translator │<──────│                      │
└─────────────────────┘       └──────────────────────┘       └──────────────────────┘
```

The gateway acts as a **protocol translator** between OpenAI's HTTP API and the Wyoming TCP protocol. It maps REST endpoints to Wyoming events (Describe, Synthesize, Transcribe, etc.) and translates audio data formats transparently using `miniaudio`.

## Development

```bash
git clone https://github.com/archmonger/Wyoming-OpenAI-Gateway.git
cd Wyoming-OpenAI-Gateway

# Install with dev dependencies (and the optional MP3 encoder)
pip install -e ".[test,mp3]"

# Run tests
pytest --cov=wyoming_openai_gateway

# Run linting
ruff check src/ tests/

# Start the gateway
TTS_HOST=127.0.0.1 TTS_PORT=10200 \
STT_HOST=127.0.0.1 STT_PORT=10300 \
python -m wyoming_openai_gateway
```

## License

MIT
