# syntax=docker/dockerfile:1

# ---- llama.cpp (built from source with CUDA) -----------------------------
# llama.cpp releases only publish CUDA binaries for Windows; Linux CUDA builds
# must be compiled from source.  Pin the release and build with the CUDA
# backend so inference layers can be offloaded onto the VRAM.  Tune
# LLAMA_VERSION/CUDA_VERSION/UBUNTU_VERSION/LLAMA_CUDA_ARCH via build args.
ARG LLAMA_VERSION=b10276
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG LLAMA_CUDA_ARCH=all-major

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llamacpp
ARG LLAMA_VERSION
ARG LLAMA_CUDA_ARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git cmake build-essential ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --branch ${LLAMA_VERSION} --depth 1 \
       https://github.com/ggml-org/llama.cpp /opt/llama-src
WORKDIR /opt/llama-src
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCH} \
    && cmake --build build --config Release -j"$(nproc)" \
       --target llama-cli llama-server
# ggml looks for `libggml-cuda.so` next to the executable (and honours
# LD_LIBRARY_PATH), so drop every shared artifact beside the binaries.
RUN mkdir -p /opt/llama.cpp \
    && cp build/bin/llama-cli build/bin/llama-server /opt/llama.cpp/ \
    && find build -name "*.so*" -exec cp -P {} /opt/llama.cpp/ \; \
    && rm -rf /opt/llama-src

# ---- builder -------------------------------------------------------------
FROM rust:1.97-bookworm AS builder
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

# ---- runtime (CUDA runtime libraries + host GPU wired by the toolkit) ----
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# The nvidia/cuda base already ships libcudart/libcublas/etc.; the host GPU
# driver libs and /dev/nvidia* are mounted automatically by the NVIDIA
# Container Toolkit at runtime (`gpus:` / `deploy.resources` in compose).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates tini wget \
       libgomp1 libstdc++6 libssl3t64 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" --home /home/exodus exodus

COPY --from=builder /build/target/release/exodus /usr/local/bin/exodus
COPY --from=llamacpp /opt/llama.cpp /opt/llama.cpp

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# llama-cli/llama-server find the ggml CUDA backend .so via LD_LIBRARY_PATH.
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

USER exodus
WORKDIR /home/exodus

# UDP multicast discovery / TCP gossip / REST API
EXPOSE 52513/udp 52514/tcp 52515/tcp

VOLUME ["/data", "/models"]

ENTRYPOINT ["tini", "--"]
CMD ["exodus", "run", "--api"]