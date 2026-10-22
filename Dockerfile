# Stage 1: Build wheels (incl. the optional MP3 encoder, so response_format=mp3
# works for clients like Open Notebook that always ask for MP3)
FROM python:3.13-slim AS builder

WORKDIR /build
COPY . .
RUN pip install hatchling && pip wheel --no-cache-dir --wheel-dir /wheels ".[mp3]"

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# ffmpeg decodes the containers miniaudio cannot open (m4a/AAC, webm/Opus) — the
# audio Open Notebook extracts from video sources. Set WITH_FFMPEG=false for a
# ~130 MB smaller image; the gateway then returns a clear 415 for those uploads.
ARG WITH_FFMPEG=true
RUN if [ "$WITH_FFMPEG" = "true" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends ffmpeg && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

EXPOSE 8555

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8555/healthz')" || exit 1

ENTRYPOINT ["wyoming-openai-gateway"]
