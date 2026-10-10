# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:26.04@sha256:da6fc2be547864451aa253836dd926da33623312df4a9a243e35dc877c378a78
FROM ${BASE_IMAGE} AS build
ARG BASE_IMAGE
ARG TARGETARCH
ARG BUILD_JOBS=6
ENV BASE_IMAGE=${BASE_IMAGE} BUILD_JOBS=${BUILD_JOBS}
RUN test "$TARGETARCH" = arm64 && apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
       git pkg-config libdrm-dev libssl-dev ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
COPY kernel-uapi/ /kernel-uapi/
COPY config/sources.env /src/sources.env
COPY scripts/build-stack.sh /src/build-stack.sh
RUN bash /src/build-stack.sh

FROM build AS package
RUN mkdir -p /out/opt/npu-demo/source /out/opt/npu-demo/licenses \
    && for project in llama.cpp ggml-rocket rocket-userspace; do \
         tar -C /src --exclude='*/.git' --exclude='*/build' --exclude='*/build-*' \
           -czf "/out/opt/npu-demo/source/${project}.tar.gz" "$project"; \
         cp "/src/${project}/LICENSE" "/out/opt/npu-demo/licenses/${project}.txt"; \
       done \
    && cp /src/build-stack.sh /out/opt/npu-demo/source/ \
    && cp -a /kernel-uapi /out/opt/npu-demo/source/

# Keep validation separate: upstream CTest needs its source scripts and complete build trees.
FROM build AS validation
RUN cmake -S /src/rocket-userspace -B /opt/validation/rocket -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DROCKETNPU_BUILD_TESTS=ON -DROCKETNPU_PROVIDER=builtin \
    && cmake --build /opt/validation/rocket --parallel "$BUILD_JOBS" \
    && cmake -S /src/ggml-rocket -B /opt/validation/ggml -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_DIR=/src/llama.cpp/ggml \
      -DROCKETNPU_DIR=/src/rocket-userspace -DGGML_ROCKET_STRICT_ABI=ON \
      -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
    && cmake --build /opt/validation/ggml --parallel "$BUILD_JOBS"
COPY scripts/qualify.sh /usr/local/bin/npu-qualify
RUN chmod 0755 /usr/local/bin/npu-qualify
ENTRYPOINT ["/usr/local/bin/npu-qualify"]

FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ARG TARGETARCH
LABEL org.opencontainers.image.title="workload-npu-demo" \
    org.opencontainers.image.description="llama.cpp with mainline Rocket NPU offload for Admiral RK3588" \
    org.opencontainers.image.source="https://github.com/admrlos/workload-npu-demo" \
    org.opencontainers.image.base.name="${BASE_IMAGE}" co.admrl.workload="npu-demo"
RUN test "$TARGETARCH" = arm64 && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 tini \
       libstdc++6 libgcc-s1 libgomp1 libdrm2 libssl3t64 ffmpeg v4l-utils \
    && rm -rf /var/lib/apt/lists/*
COPY --from=package /out/ /
COPY scripts/npu-demo.py /usr/local/bin/npu-demo
COPY scripts/vision-client.py /usr/local/bin/npu-vision
COPY scripts/fetch-model.py /usr/local/bin/npu-fetch-model
COPY config/model*.json /opt/npu-demo/
ENV PATH=/opt/llama/bin:$PATH LD_LIBRARY_PATH=/opt/llama/bin \
    GGML_BACKEND_PATH=/opt/rocket/libggml-rocket.so ROCKET_DEV=/dev/accel/accel0 \
    ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1 \
    NPU_MODE=rocket NPU_MODEL_MANIFEST=/models/model.json NPU_DATA_DIR=/data \
    NPU_HOST=0.0.0.0 NPU_PORT=8080 LANG=C.UTF-8
RUN chmod 0755 /usr/local/bin/npu-demo /usr/local/bin/npu-vision /usr/local/bin/npu-fetch-model \
    && mkdir -p /models /data /run/npu-demo \
    && printf '/opt/llama/bin\n' > /etc/ld.so.conf.d/llama.conf && ldconfig \
    && dpkg-query -W > /opt/npu-demo/provenance/runtime-packages.txt \
    && python3 -c 'import glob,subprocess; files=glob.glob("/opt/llama/bin/*.so*")+["/opt/rocket/libggml-rocket.so","/opt/llama/bin/llama-server"]; results=[subprocess.run(["ldd",f],capture_output=True,text=True) for f in files]; assert all(r.returncode==0 and "not found" not in r.stdout for r in results), "unresolved runtime library"'
EXPOSE 8080/tcp
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/npu-demo"]
CMD ["serve"]
