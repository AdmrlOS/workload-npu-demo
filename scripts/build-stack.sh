#!/usr/bin/env bash
set -euo pipefail
source /src/sources.env
jobs=${BUILD_JOBS:-6}
cd /kernel-uapi
sha256sum --strict -c SHA256SUMS > /src/uapi-verification.txt
test -s include/drm/rocket_accel.h
test -s include/drm/drm.h
# Keep the exported DRM directory together; no vendor or downstream header substitution.
cp -a include/drm /usr/local/include/
cd /src
checkout() {
    local name=$1 url=$2 commit=$3
    git init -q "$name"
    git -C "$name" fetch -q --depth 1 "$url" "$commit"
    git -C "$name" checkout -q --detach FETCH_HEAD
    test "$(git -C "$name" rev-parse HEAD)" = "$commit"
}
checkout llama.cpp https://github.com/ggml-org/llama.cpp.git "$LLAMA_COMMIT"
checkout ggml-rocket https://github.com/gregordinary/ggml-rocket.git "$GGML_ROCKET_COMMIT"
checkout rocket-userspace https://github.com/gregordinary/rocket-userspace.git "$ROCKET_USERSPACE_COMMIT"

cmake -S llama.cpp -B llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_VULKAN=OFF \
    -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp/build --parallel "$jobs" \
    --target llama-server llama-cli llama-bench llama-mtmd-cli ggml-cpu
cmake -S ggml-rocket -B ggml-rocket/build-dl -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_ROCKET_DL=ON -DGGML_ROCKET_STRICT_ABI=ON \
    -DHOST_DIR=/src/llama.cpp -DGGML_LIB_DIR=/src/llama.cpp/build/bin \
    -DROCKETNPU_DIR=/src/rocket-userspace -DROCKETNPU_PROVIDER=builtin
cmake --build ggml-rocket/build-dl --parallel "$jobs"
mkdir -p /out/opt/{llama/bin,rocket,npu-demo/provenance}
cp -a llama.cpp/build/bin/. /out/opt/llama/bin/
cp ggml-rocket/build-dl/libggml-rocket.so /out/opt/rocket/
cp /kernel-uapi/{identity.json,SHA256SUMS} /out/opt/npu-demo/provenance/
cp /src/sources.env /out/opt/npu-demo/provenance/
cp llama.cpp/build/CMakeCache.txt /out/opt/npu-demo/provenance/llama-cmake.txt
cp ggml-rocket/build-dl/CMakeCache.txt /out/opt/npu-demo/provenance/rocket-cmake.txt
gcc --version > /out/opt/npu-demo/provenance/compiler.txt
dpkg-query -W > /out/opt/npu-demo/provenance/build-packages.txt
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
src = dict(line.split('=', 1) for line in Path('/src/sources.env').read_text().splitlines()
           if line and not line.startswith('#'))
info = {'sources': src, 'base_image': os.environ['BASE_IMAGE'],
        'architecture': 'linux/arm64', 'cpu_arch': 'armv8.2-a+dotprod+fp16',
        'rocket_provider': 'builtin', 'uapi_identity': json.loads(Path('/kernel-uapi/identity.json').read_text()),
        'rocket_uapi_sha256': hashlib.sha256(Path('/kernel-uapi/include/drm/rocket_accel.h').read_bytes()).hexdigest()}
Path('/out/opt/npu-demo/build-info.json').write_text(json.dumps(info, indent=2) + '\n')
PY
