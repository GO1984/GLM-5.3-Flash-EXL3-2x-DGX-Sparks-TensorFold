# The Zig engine image (ENGINE=zig): TensorFold's tensorfold-native, built from TensorFold at TF_REF with zig/patches/*.patch
# applied (git apply in the checkout's root, filename order), on a CUDA runtime image with only what the binary loads at
# run time: libcuda (the host driver, through the NVIDIA runtime), cuBLAS/cuBLASLt (the CUDA runtime image), NCCL (the pip
# wheel of the version the Python image carries) and libibverbs (the RoCE all-gathers and NCCL's IB transport).
# Built by scripts/prepare.sh (never by hand: it passes the build arguments and the build context, which holds this file,
# zig/patches and zig/aot only). No GPU is needed to build: nvcc compiles the sm_121 fatbins, and the captured Triton
# kernel set (zig/aot/glm-aot-sm121.tar.xz) is data.
ARG BUILD_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG RUN_IMAGE=nvidia/cuda:13.0.2-runtime-ubuntu24.04

FROM ${BUILD_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ca-certificates curl git xz-utils python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY zig/patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
ARG SM=121
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tensorfold
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
# libtfgrammar (structured output: response_format, guided_*): TensorFold's grammar library over xgrammar, which the
# server dlopens at the first structured request from <exe>/../lib. xgrammar's tag is pinned by commit
ARG XGRAMMAR_REF=v0.2.8
ARG XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa
RUN git clone -q --depth 1 --branch ${XGRAMMAR_REF} --recurse-submodules --shallow-submodules https://github.com/mlc-ai/xgrammar /xgrammar && \
    test "$(git -C /xgrammar rev-parse HEAD)" = "${XGRAMMAR_COMMIT}" && \
    cd /tensorfold && /opt/zig/zig build grammar -Dxgrammar=/xgrammar -Doptimize=fast --prefix /opt/tensorfold -j"$(nproc)" && \
    mkdir -p /opt/tensorfold/native/lib && cp /opt/tensorfold/lib/libtfgrammar.so /opt/tensorfold/native/lib/ && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES && \
    cp /xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-Apache-2.0.txt && \
    cp /xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-NOTICE.txt
# the captured Triton kernel set (aot.json + cubins) the GLM engine replays, checked against its pinned sha256
ARG AOT_SHA256
COPY zig/aot/glm-aot-sm121.tar.xz /tmp/glm-aot-sm121.tar.xz
RUN echo "${AOT_SHA256}  /tmp/glm-aot-sm121.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/tensorfold/share/tensorfold/cuda/sm121 && \
    tar -xJf /tmp/glm-aot-sm121.tar.xz -C /opt/tensorfold/share/tensorfold/cuda/sm121 && \
    test -f /opt/tensorfold/share/tensorfold/cuda/sm121/aot.json
# NCCL: the wheel's library, the version the Python image runs (the RoCE links are tuned against it)
ARG NCCL_VERSION
RUN python3 -m venv /tmp/nv && /tmp/nv/bin/pip install -q --no-cache-dir "nvidia-nccl-cu13==${NCCL_VERSION}" && \
    mkdir -p /opt/nccl && cp -a "$(find /tmp/nv -name 'libnccl.so.2' | head -1)" /opt/nccl/ && ls -l /opt/nccl

FROM ${RUN_IMAGE}
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      libibverbs1 ibverbs-providers rdma-core && rm -rf /var/lib/apt/lists/*
ARG NCCL_VERSION
# (the CUDA runtime image carries an older NCCL at this name: replaced, so the one that loads is the wheel's)
COPY --from=build /opt/nccl/libnccl.so.2 /tmp/libnccl.so.new
COPY --from=build /opt/tensorfold /opt/tensorfold
RUN cd /lib/aarch64-linux-gnu && rm -f libnccl.so.2 libnccl.so.2.[0-9]*.[0-9]* && mv /tmp/libnccl.so.new libnccl.so.${NCCL_VERSION} && \
    ln -s libnccl.so.${NCCL_VERSION} libnccl.so.2 && \
    ldconfig && ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native && \
    ln -s /opt/tensorfold/bin/tensorfold /usr/local/bin/tensorfold && tensorfold-native --version
ARG PATCHES_HASH
ARG TF_REF
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF}
ENV HF_HOME=/root/.cache/huggingface TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /workspace
