# Mainline RK3588 NPU workload for Admiral

An ARM64 container running **llama.cpp + ggml-rocket + rocket-userspace** through
the mainline **Rocket** driver. It serves an OpenAI-compatible text/image API,
supports saved images and opt-in USB UVC capture, and contains no RKNN/RKLLM
runtime or vendor NPU kernel driver.

Targets the kernel team's **original Orange Pi 5 / RK3588S** Admiral image with
stock Linux 7.2.9 Rocket. Other RK3588 boards need their own boot-tested BSP.
The signed kernel/module/device-tree/clock policy remains a host responsibility.

Published manually to the private [AdmrlOS repository](https://github.com/AdmrlOS/workload-npu-demo)
and [GHCR package](https://github.com/orgs/AdmrlOS/packages/container/package/workload-npu-demo)
on 10 October 2026. Tags `2026-10-10` and `latest` refer to the initial ARM64 runtime.
Use the immutable reference already pinned in the Admiral configuration:

```text
ghcr.io/admrlos/workload-npu-demo@sha256:2a562d5590266e7c276a5b9c7d6757506da67d3cf703f5e24621a64492abb575
```

Private GHCR access requires Admiral registry credentials with permission to pull
this package. The CI workflow validates source builds; publication remains manual.

This is flexible **NPU-assisted inference**: llama.cpp determines supported model
architectures and ggml-rocket determines which operations can offload. Unsupported
operations run on CPU; token decode is primarily CPU in the documented RK3588
strategy. A new vision architecture needs a frontend and operator integration.
There is no promise that arbitrary models, all operators, or every GGUF
quantization execute on the NPU. Vulkan/PanVK and Mesa/Teflon are separate stacks
and are not included in this first qualification path.

## Build

Use Docker Buildx on an ARM64 Linux builder (Docker Desktop on Apple Silicon also
builds the container). From this repository:

```sh
# Stock handover only: export pinned, checksum-verified Linux 7.2.9 headers.
scripts/prepare-uapi.sh

# Alternatively, use the BSP team's exact exported headers and identity JSON:
# scripts/prepare-uapi.sh /path/to/kernel-uapi-export /path/to/bsp-identity.json

docker buildx build --platform linux/arm64 --load -t workload-npu-demo:local .
python3 -m unittest discover -s tests -v

# Separate image with the same pinned sources and full hardware test builds:
docker buildx build --platform linux/arm64 --target validation --load \
  -t workload-npu-demo:validation .
```

`kernel-uapi/` is generated and ignored by Git. Its complete exported DRM headers
and SHA256 manifest are verified in the build. For a patched host, import its
matching export; never substitute a downstream header for a stock driver. Remove
or move an old export before preparing a different one. The stock exporter does
not modify or build a kernel for deployment.

The source revisions are locked in [`config/sources.env`](config/sources.env).
The plugin uses llama.cpp's bundled ggml and shared `libggml-base.so`, with strict
ABI checking. Both stages use the same digest-pinned Ubuntu 26.04 baseline,
explicit ARMv8.2 FP16/dot-product CPU flags, and Vulkan disabled. Apt packages are
resolved at build time: this is a pinned source build, not a claim of byte-for-byte
reproducibility. Compiler, packages, flags, UAPI hash/identity and source revisions
are recorded in `/opt/npu-demo/build-info.json` and `provenance/`.
The image also retains the three upstream license files and pinned application
source archives in `/opt/npu-demo/licenses/` and `source/`.

## Models

Models are downloaded once into a durable volume, with HTTPS sources pinned to
repository revisions and SHA256. They are not downloaded at service startup.

```sh
mkdir -p models data

# Small CPU vision smoke candidate: ~365 MB of model + FP16 projector files.
python3 scripts/fetch-model.py --manifest config/model.example.json --destination models

# Kernel-team initial NPU candidate: ~1.99 GB; use a fresh directory/volume.
# python3 scripts/fetch-model.py --manifest config/model.smolvlm2.json --destination models
```

Copy the resulting `model.gguf`, `mmproj.gguf` and `model.json` into Admiral's
`npu-models` Custom Volume at `/models`. The downloader refuses to overwrite a
different existing model; use a separate model directory/volume when switching
pairs. To stage directly inside a populated Admiral volume, use
`npu-fetch-model --manifest /opt/npu-demo/model.smolvlm2.json --destination /models`.
Then restore the empty Command field and start the normal service.

Use the same manifest format for other compatible GGUFs. A projector is optional
for text-only models. The runtime verifies GGUF magic and every hash before
starting. Recorded model architecture/license fields are provenance, not an
automatic architecture or accuracy certification. Hashes cannot establish that
two independently sourced files form a compatible model pair.

## Run and deploy

On the Orange Pi host, a local bring-up invocation is:

```sh
docker run --rm --name npu-demo --device /dev/accel/accel0 \
  -v /sys:/sys:ro -v "$PWD/models:/models:ro" -v "$PWD/data:/data" \
  -p 8080:8080 workload-npu-demo:local
```

For Admiral, use [`config/admiral-dashboard.yaml`](config/admiral-dashboard.yaml)
and the [deployment guide](docs/admiral-mapping.md). Leave Command empty, turn
device isolation off, mount Custom Volumes at `/models` and `/data`, and publish
TCP 8080. The service uses absolute paths and supports Admiral's root launch from
`/`. It exits on startup failure, server exit or persistent watchdog failure;
Admiral ignores Docker `HEALTHCHECK`. Publish a tested image and deploy its
immutable registry digest:

```sh
docker buildx build --platform linux/arm64 --push \
  -t ghcr.io/admrlos/workload-npu-demo:YOUR_VERSION .
```

`NPU_MODE=rocket` is the default. Missing plugin, device access, wrong DRM driver,
model mismatch or backend registration failure causes a visible failure. There
is no automatic switch to CPU service. `NPU_MODE=cpu` explicitly selects the
comparison path and retains CPU weight repacking.

```sh
# Inside the deployed workload:
npu-demo inspect
npu-demo check
npu-demo validate-models

# Saved image; logs and response metadata can be kept in the data volume:
npu-vision --image /data/frame.jpg --prompt 'Describe the objects in this image.' \
  --output /data/result.json

# Discover the actual camera's modes before choosing capture parameters:
v4l2-ctl --device /dev/video0 --list-formats-ext
npu-vision --camera /dev/video0 --format mjpeg --size 640x480 \
  --frames 10 --interval 5 --output /data/camera-result.json
```

The camera client acquires one frame and completes one request at a time. It
retains a bounded latest-frame file, drops queuing entirely, and records capture
and request times. It does not start a camera on boot. CSI cameras, ISP/RGA/MPP
and hardware video decode require separate qualification.

## Runtime settings

| Environment | Default | Meaning |
| --- | --- | --- |
| `NPU_MODE` | `rocket` | `rocket` or explicit `cpu` comparison. |
| `NPU_MODEL_MANIFEST` | `/models/model.json` | Hash-pinned model/projector pair. |
| `ROCKET_DEV` | `/dev/accel/accel0` | Actual Rocket node; one node may represent three cores. |
| `NPU_HOST`, `NPU_PORT` | `0.0.0.0`, `8080` | API bind address and port. |
| `NPU_CONTEXT` | `4096` | Prompt, image tokens and output context. |
| `NPU_BATCH`, `NPU_UBATCH` | `2048`, `2048` | Initial prefill batching; reduce if memory requires. |
| `NPU_THREADS`, `NPU_PARALLEL` | `4`, `1` | CPU workers and simultaneous server slots. |
| `NPU_STARTUP_TIMEOUT` | `120` | Whole startup deadline, including hashes and readiness. |
| `NPU_HEALTH_INTERVAL`, `NPU_HEALTH_FAILURES` | `15`, `3` | Watchdog interval and consecutive failure threshold. |
| `NPU_DATA_DIR` | `/data` | Timestamped logs and run metadata. |
| `ROCKET_MM_PROFILE`, `ROCKET_LOG_STDERR` | `1`, `1` | Request-time offload profiling and library logs. |

Rocket selects the vision encoder explicitly (`--mmproj-device ROCKET`) and uses
`-ngl 0 -nr` for eligible accelerator prefill. CPU mode disables operation and
projector offload and removes the external plugin. Batch/context values are
starting points, not RAM guarantees. FP16 expansion of quantized weights, NPU
buffers, KV cache, image tiling and camera buffers all consume memory. Inspect
the actual board RAM and effective cgroup limit before qualifying a model.

Admiral's rollout probation measures three minutes of process uptime, rather than
this service's HTTP readiness. The default startup deadline leaves time to exit
on a load failure within that interval. Raising it for a slow model can let a
rollout be considered proven before the model becomes ready; qualify startup
time and inspect readiness separately when using a longer deadline.

## Acceptance

See [hardware and model acceptance](docs/acceptance.md) and the
[local validation record](docs/validation.md). Health establishes that the
configured server/device is available; `npu_execution_verified: false` remains
explicit until separate request-time offload evidence has been collected.
Backend registration and a successful API response do not establish NPU work.
Measure image encode, language prefill, first-token latency, decode and end-to-end
latency separately against the same CPU input, then qualify sustained memory,
thermal conditions and recovery on the physical board.

The supplied [kernel-team handover](docs/kernel-handover.txt) begins partway
through a table, as pasted. It reports configuration/object compilation checks,
but no completed image build, board boot or workload NPU performance result.

Primary references: [Admiral workload contract](https://docs.admrl.co/core-concepts/configurations/),
[Linux Rocket](https://docs.kernel.org/accel/rocket/index.html),
[pinned ggml-rocket](https://github.com/gregordinary/ggml-rocket/tree/550986d2212eb3a6f213a98c6a0dec0a8bf2f43b),
[pinned rocket-userspace](https://github.com/gregordinary/rocket-userspace/tree/f86cf52c666b4eddadc17d80d6c558b4067c0d6b),
[small model pair](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/tree/b9e4379657e1450d04d02eec8e345667265b0a00),
[handover model pair](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/tree/1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91).
