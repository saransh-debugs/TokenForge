# syntax=docker/dockerfile:1.7
# One image, two entrypoints: tokenforge-gateway and tokenforge-mock-engine.
# Base pinned by digest (versions.lock); Python deps pinned by requirements.serving.txt,
# which is exported from uv.lock by `make lock`.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY serving/docker/requirements.serving.txt /app/requirements.txt
# Optional CA bundle for TLS-intercepting proxies: `--secret id=ca,src=<bundle>`.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-deps -r /app/requirements.txt

COPY pyproject.toml README.md /app/
COPY tokenforge /app/tokenforge
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-deps /app

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["tokenforge-gateway"]
