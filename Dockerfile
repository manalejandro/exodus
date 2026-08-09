# syntax=docker/dockerfile:1

# ---- llama.cpp (prebuilt Vulkan release) --------------------------------
# llama.cpp publishes prebuilt Linux binaries for the Vulkan backend (CUDA
# prebuilds are Windows-only).  Vulkan runs on NVIDIA GPUs, so no source
# build is needed.  Swap LLAMA_ASSET for the CPU variant if no GPU is used.
ARG LLAMA_VERSION=b10276
ARG LLAMA_ASSET=llama-${LLAMA_VERSION}-bin-ubuntu-vulkan-x64.tar.gz
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM debian:bookworm-slim AS llamacpp
ARG LLAMA_VERSION
ARG LLAMA_ASSET
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q -O /llama.tar.gz \
       "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/${LLAMA_ASSET}" \
    && mkdir -p /opt/llama.cpp \
    && tar xzf /llama.tar.gz -C /opt/llama.cpp --strip-components=1 \
    && rm /llama.tar.gz

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

# ---- runtime (nvidia runtime base + Vulkan loader) -----------------------
# The nvidia/cuda base ships the CUDA runtime libs; the host GPU driver and
# Vulkan ICD are mounted automatically by the NVIDIA Container Toolkit at
# runtime (`gpus:` / `deploy.resources` in compose).
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates tini wget \
       libgomp1 libstdc++6 libssl3t64 \
       libvulkan1 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" --home /home/exodus exodus

COPY --from=builder /build/target/release/exodus /usr/local/bin/exodus
COPY --from=llamacpp /opt/llama.cpp /opt/llama.cpp
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# graphics capability is required for the NVIDIA Vulkan ICD.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# llama-cli/llama-server find the ggml Vulkan/CUDA backend .so via LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH=/opt/llama.cpp

ENV EXODUS_DATA_DIR=/data \
    EXODUS_MODEL_DIR=/models \
    EXODUS_NODE_NAME=exodus-node \
    EXODUS_NODE_HOST=0.0.0.0 \
    EXODUS_NODE_PORT=52514 \
    EXODUS_API_HOST=0.0.0.0 \
    EXODUS_API_PORT=52515 \
    EXODUS_LLAMA_BIN=/opt/llama.cpp/llama-cli \
    EXODUS_LLAMA_SERVER_BIN=/opt/llama.cpp/llama-server

RUN mkdir -p /data /models && chown -R exodus:exodus /data /models

# Run as root so the entrypoint can chown the bind-mounted volumes; it then
# drops to the unprivileged `exodus` user before starting the node.
WORKDIR /home/exodus

# UDP multicast discovery / TCP gossip / REST API
EXPOSE 52513/udp 52514/tcp 52515/tcp

VOLUME ["/data", "/models"]

ENTRYPOINT ["tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["run", "--api"]