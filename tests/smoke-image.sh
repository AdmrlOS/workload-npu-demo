#!/usr/bin/env bash
# Packaging and fail-closed checks that do not claim NPU computation.
set -euo pipefail
image=${1:-workload-npu-demo:local}
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
docker run --rm --entrypoint /bin/bash "$image" -ec '
    test "$(uname -m)" = aarch64
    test -s /opt/npu-demo/build-info.json
    test -s /opt/rocket/libggml-rocket.so
    for program in llama-cli llama-server llama-mtmd-cli llama-bench; do
        "$program" --help >/dev/null 2>&1
    done
    # Registration exists even without hardware. Never count this as NPU acceptance.
    llama-cli --list-devices 2>/dev/null | python3 -c '\''import sys; assert "ROCKET:" in sys.stdin.read()'\''
    env -u GGML_BACKEND_PATH llama-cli --list-devices 2>/dev/null | python3 -c '\''import sys; assert "ROCKET:" not in sys.stdin.read()'\''
    npu-vision --help >/dev/null
    npu-fetch-model --help >/dev/null
'
# A valid file/hash fixture isolates missing-hardware failure from missing-model failure.
python3 - "$temporary" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); data=b'GGUFsmoke-only'
(root/'model.gguf').write_bytes(data)
(root/'model.json').write_text(json.dumps({'name':'packaging fixture','model':{
    'path':'/models/model.gguf','sha256':hashlib.sha256(data).hexdigest(),
    'source':'fixture:smoke-only'}}))
PY
set +e
docker run --rm -v "$temporary:/models:ro" "$image" > "$temporary/missing-npu.log" 2>&1
status=$?
set -e
[[ $status == 1 ]] || { cat "$temporary/missing-npu.log"; echo "Expected missing-NPU exit 1, got $status" >&2; exit 1; }
python3 - "$temporary/missing-npu.log" <<'PY'
import pathlib,re,sys
text=pathlib.Path(sys.argv[1]).read_text()
assert re.search(r'Required Rocket device .* unavailable',text), text
PY
echo 'ARM64 packaging, plugin separation and missing-NPU failure passed; hardware execution remains untested.'
