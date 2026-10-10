#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e kernel-uapi ]]; then
    echo 'kernel-uapi already exists; keep it or move it aside before exporting a different host contract.' >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    docker buildx build --platform linux/arm64 -f Dockerfile.uapi \
        --output type=local,dest=kernel-uapi .
elif [[ $# -eq 2 ]]; then
    # The second argument is the BSP team's recorded kernel/build/patch identity JSON.
    test -s "$1/include/drm/rocket_accel.h"
    test -s "$1/include/drm/drm.h"
    test -s "$2"
    mkdir -p kernel-uapi/include
    # Only DRM is required. Other Linux headers have case-colliding filenames on macOS.
    cp -a "$1/include/drm" kernel-uapi/include/
    cp "$2" kernel-uapi/identity.json
    python3 - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('kernel-uapi')
json.loads((root / 'identity.json').read_text())
with (root / 'SHA256SUMS').open('w') as output:
    for file in sorted((root / 'include').rglob('*')):
        if file.is_file():
            output.write(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(root)}\n')
PY
else
    echo 'Usage: scripts/prepare-uapi.sh [EXPORTED_HEADERS BSP_IDENTITY.json]' >&2
    exit 2
fi
echo 'Export staged. Build this image only for a host with the matching UAPI and patch identity.'
