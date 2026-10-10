# Board acceptance

This repository packages an open Rocket userspace path for llama.cpp. A container build, a listed device, or a healthy HTTP service does not establish NPU inference correctness, performance, or support for every architecture. Acceptance applies to one recorded host/container/model tuple.

## 1. Record the host and build identity

The BSP team supplies a boot-tested original Orange Pi 5 / RK3588S image, the matching exported UAPI headers, and its kernel/module/DTB identities. Record the image revision, `uname -a`, board model and compatible strings, Rocket binding/core discovery, module signature/loading result, RAM, container memory limit, and actual clock/thermal conditions. Record the container digest, base digest, three source commits, compiler/build flags and exported header hash. Keep the generated build provenance with the result.

The handover's Linux 7.2.9 stock Rocket driver and the optional downstream patch series are different qualification targets. Start with the stock host. Capability probing must remain enabled; absence of downstream parameters is expected. The workload does not install kernel modules, write clocks, or supply Mali GPU firmware. If the BSP team adopts patches, rebuild and sign through its pipeline, export the resulting headers, and repeat acceptance. Published 600 MHz project measurements cannot establish results on the stock Admiral board.

The pinned sources were retrieved and inspected, including confirmation that llama.cpp `b11401` resolves to the selected commit. They have not thereby been proven to run together on this board.

## 2. Primitive correctness gates

Build the Dockerfile's `validation` target for Linux ARM64 using the same prepared UAPI build context as the application image. The validation image retains upstream source fixtures, both CTest build trees, and its test programs. Run its qualification command on the physical board with `/dev/accel`, readable sysfs, and an output data volume exposed using Admiral's unrestricted-hardware workload mode. The Docker equivalent is:

```sh
docker build --platform linux/arm64 --target validation -t npu-demo:validation .
mkdir -p qualification
docker run --rm --device /dev/accel/accel0 -v /sys:/sys:ro \
  -v "$PWD/qualification:/data/qualification" \
  npu-demo:validation /data/qualification/run-001
```

The Docker command is a local testing equivalent, not an Admiral deployment instruction. On Admiral, use the workload's persistent data volume for `/data`. Run one qualification process at a time. Set `ROCKET_DEV` if the node is not `/dev/accel/accel0`; inspect its actual driver identity.

The script runs three library hardware gates (`matmul_tiled_rocket`, `matmul_mt_rocket`, `conv2d_fp16_rocket`), the ggml graph gate (`test-rocket-matmul`), and the host metadata placement gate (`test-rocket-placement`). It writes raw logs, JUnit XML, exit codes, metadata, and a JSON result. Every required gate must exist, execute, and pass. Required hardware skips fail qualification; a collection of skipped tests is never a successful NPU result. The pinned library uses exit code 2 for absent hardware; the pinned ggml gates use 77. CTest translates each into a skipped test.

Use `QUALIFY_FULL_SUITE=1` for the broader upstream suites. Additional RK3576-only or artifact-dependent tests may skip; retain their reasons. All failures still fail the run, and required RK3588 gates still must pass. A full run can take substantially longer and may need additional SigLIP/SAM/ModernBERT fixtures. Keep these network tests separate from this workload's saved-image VLM acceptance.

Capture host kernel logs across the entire test interval. Admiral uses s6, so upstream's `journalctl` fixture can skip even when computation tests ran. The script reports this gap instead of assuming a clean kernel. A Rocket job timeout can signal its fence like completed work, making independent host log review necessary. Check for NPU timeouts, IOMMU errors, resets, and Rocket/DRM warnings. Save the interval log alongside the qualification output; setting `HOST_KERNEL_LOG` to a mounted interval log also enables automated screening. Preserve before/after host snapshots and confirm that the capture did not truncate or roll over.

`llama-cli --list-devices` separately checks plugin loading and registration. The pinned plugin always advertises one `ROCKET` accelerator even before opening hardware; listing alone cannot prove device access or successful computation.

## 3. Saved-image correctness and offload

Use one compatible GGUF/projector pair and one fixed saved image before adding camera capture. Record exact model/projector URLs or artifact IDs and SHA-256 hashes, quantizations, model architectures, prompt, image preprocessing/tiling, context, batch/microbatch, threads, and output budget. Ensure model, projector, KV cache, temporary tensors, NPU residency, and image buffers fit the effective container limit. `/proc/meminfo` can describe host memory; set explicit residency budgets when its automatic sizing would exceed the workload budget.

Run the CPU and Rocket services serially, stopping the first before starting the second, on the same board and with identical files and request settings. Keep all other workload/NPU activity quiet. These upstream invocations make placement explicit:

```sh
# Optimized CPU baseline; leave CPU repacking enabled.
env -u GGML_BACKEND_PATH -u MTMD_BACKEND_DEVICE \
  llama-server -m /models/model.gguf --mmproj /models/mmproj.gguf \
  --no-mmproj-offload --no-op-offload -ngl 0 \
  -b 2048 -ub 2048 -c 4096 --host 0.0.0.0 --port 8080

# Stop the CPU server before running this arm.
GGML_BACKEND_PATH=/opt/rocket/libggml-rocket.so \
ROCKET_DEV=/dev/accel/accel0 ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1 \
  llama-server -m /models/model.gguf --mmproj /models/mmproj.gguf \
  --mmproj-device ROCKET -ngl 0 -nr \
  -b 2048 -ub 2048 -c 4096 --host 0.0.0.0 --port 8080
```

The batch/context values above follow the handover's starting recipe; reduce them for the selected model, image tokens, RAM, and container budget and use the same values in both arms. `-ngl 0` alone does not disable Rocket. `-nr` makes quantized host weights eligible for Rocket; retain CPU repacking in the optimized CPU arm and optionally add a separate CPU no-repack arm to isolate its cost. Do not set `--no-op-offload` in the Rocket arm. `GGML_BACKEND_PATH` names one shared-library file. Place the plugin outside llama.cpp's ordinary backend directory so an unset variable actually removes it from the baseline.

The vision encoder's automatic selection checks GPU/IGPU backends, while Rocket is ACCEL, so use `--mmproj-device ROCKET` (or `MTMD_BACKEND_DEVICE=ROCKET`). The equivalent saved-image CLI is `llama-mtmd-cli -m /models/model.gguf --mmproj /models/mmproj.gguf --image /data/frame.jpg --mmproj-device ROCKET -ngl 0 -nr -p 'Describe the objects and their positions.'`; use the same plugin/runtime environment as the server. For the CPU CLI arm, unset the plugin/device environment and supply `--no-mmproj-offload --no-op-offload`.

Save complete server stdout/stderr and the exact JSON request/response for each arm. Use the server's `/v1/chat/completions` image input with a JPEG/PNG data URL, deterministic sampling (`temperature: 0`), and the same output-token budget. Keep the full response, including timings when available, and record client wall time. Warm up each arm, record cold startup separately, and compare at least three subsequent runs. State clocks/temperature at the time of each sample.

Confirm `CLIP using ROCKET backend` for vision selection and actual Rocket operation/route/profile counters for eligible vision and language prefill work. Selection alone does not prove every operation ran on the NPU; default counters may be emitted when the server exits, so preserve shutdown logs. Unsupported operations and token decode can remain on CPU. A workload can answer correctly while executing zero useful NPU operations; that does not pass the NPU acceptance milestone.

Compare projected embeddings where practical (`MTMD_DEBUG_EMBEDDINGS` is an upstream debugging hook), fixed-image answers, and a task-relevant labeled image set. Text token equality alone is not a complete accuracy metric. Accept only when model outputs meet the application's criteria and actual offload evidence exists for both the supported encoder and prefill paths. Missing model support, unaligned shapes, small batches, or incompatible projector files require a new model/graph decision.

## 4. Performance and sustained operation

Report camera/preprocessing time, image-encoder time, language prefill rate, time to first token, CPU decode rate, client end-to-end latency, peak memory/cgroup pressure, temperatures/clocks, and errors/restarts separately. Quote measured encoder and whole-application speedups separately. Rocket documentation's modest SmolVLM image-encoder improvement and larger language-prefill results are project reports; neither is an Admiral result.

After the saved-image milestone passes, qualify USB UVC capture format/resolution and add bounded frame sampling, a resize/crop policy, capture timestamps, and stale-frame dropping. Test repeated requests, service restart, memory pressure, intended concurrency, and sustained thermal load. Device renumbering or host driver rebind requires discovery and process restart because userspace capability state is cached. Do not claim every-frame realtime operation from a single successful image response.

The first release gate is one physical Orange Pi 5 answering a saved-image request with a validated GGUF/projector pair, required primitive gates passed, independently reviewed host logs, and evidence that supported encoder and prefill operations ran through Rocket. Broader model coverage, camera throughput, and production reliability remain separately measured gates.

## Source contract

- [Pinned llama.cpp backend loader](https://github.com/ggml-org/llama.cpp/blob/a7fb71fab83b474a0892b9a05aaa3a8ddca2729b/ggml/src/ggml-backend-reg.cpp), [argument definitions](https://github.com/ggml-org/llama.cpp/blob/a7fb71fab83b474a0892b9a05aaa3a8ddca2729b/common/arg.cpp), and [multimodal backend selection](https://github.com/ggml-org/llama.cpp/blob/a7fb71fab83b474a0892b9a05aaa3a8ddca2729b/tools/mtmd/clip.cpp).
- [Pinned ggml-rocket build modes and gates](https://github.com/gregordinary/ggml-rocket/blob/550986d2212eb3a6f213a98c6a0dec0a8bf2f43b/CMakeLists.txt), [registration/ABI guard](https://github.com/gregordinary/ggml-rocket/blob/550986d2212eb3a6f213a98c6a0dec0a8bf2f43b/ggml-rocket.cpp), and [model/runtime contract](https://github.com/gregordinary/ggml-rocket/blob/550986d2212eb3a6f213a98c6a0dec0a8bf2f43b/API.md).
- [Pinned Rocket library test contract](https://github.com/gregordinary/rocket-userspace/blob/f86cf52c666b4eddadc17d80d6c558b4067c0d6b/CMakeLists.txt), [device and capability probing](https://github.com/gregordinary/rocket-userspace/blob/f86cf52c666b4eddadc17d80d6c558b4067c0d6b/src/rocket_npu.c), and [kernel-log fixture](https://github.com/gregordinary/rocket-userspace/blob/f86cf52c666b4eddadc17d80d6c558b4067c0d6b/tests/npu_klog_fixture.sh).
