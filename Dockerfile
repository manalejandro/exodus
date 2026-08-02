# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_DATA_HOME=/data

WORKDIR /app

# Install the exodus package with the API extra (fastapi + uvicorn).
# Layer cached by copying only the build inputs first.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Non-root runtime user.  /data holds the node identity + SQLite ledger and is
# meant to be mounted as a volume.
RUN useradd --create-home --uid 1000 exodus \
    && mkdir -p /data \
    && chown -R exodus:exodus /data

USER exodus

VOLUME /data
EXPOSE 52515

# `exodus run` starts the node loop; override with e.g. `exodus api` or
# `exodus status` (also used by the compose healthcheck).
ENTRYPOINT ["exodus"]
CMD ["run"]
