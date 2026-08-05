# syntax=docker/dockerfile:1

# ---- builder -------------------------------------------------------------
FROM rust:1.97-bookworm AS builder
WORKDIR /build

# rusqlite ("bundled") compiles SQLite from source -> needs a C toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

# ---- runtime -------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini wget \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" --home /home/exodus exodus

COPY --from=builder /build/target/release/exodus /usr/local/bin/exodus

# The node is CPU-only today; NVIDIA_* vars are provided so the container is
# ready for GPU inference once the runtime uses the device (the driver libs are
# injected automatically by the NVIDIA Container Toolkit / `gpus:` in compose).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

ENV EXODUS_DATA_DIR=/data \
    EXODUS_MODEL_DIR=/models \
    EXODUS_NODE_NAME=exodus-node \
    EXODUS_NODE_HOST=0.0.0.0 \
    EXODUS_NODE_PORT=52514 \
    EXODUS_API_HOST=0.0.0.0 \
    EXODUS_API_PORT=52515

RUN mkdir -p /data /models && chown -R exodus:exodus /data /models

USER exodus
WORKDIR /home/exodus

# UDP multicast discovery / TCP gossip / REST API
EXPOSE 52513/udp 52514/tcp 52515/tcp

VOLUME ["/data", "/models"]

ENTRYPOINT ["tini", "--"]
CMD ["exodus", "run", "--api"]
