# Local validation record

Validation on 10 October 2026 used Apple Silicon Docker Desktop's native Linux
ARM64 builder, with LinuxKit kernel `7.0.14-linuxkit`. This machine has no RK3588
Rocket NPU. These results qualify packaging and software behavior; they are not
Orange Pi boot, NPU correctness, or NPU performance results.

Completed:

- Exported Linux 7.2.9 DRM UAPI from the kernel archive whose SHA256 matches the
  supplied handover. Preserved its DRM directory and verified every exported
  header hash during the userspace build. Exporting only DRM avoids unrelated
  Linux header filename collisions on macOS filesystems.
- Built the pinned llama.cpp tools, shared ggml/llama/multimodal libraries,
  strict-ABI Rocket plugin, and runtime image for ARM64. All runtime ELF
  dependencies resolved. The plugin registers when explicitly loaded and is
  absent from the CPU baseline when its environment setting is removed.
- Built the separate Rocket library and standalone ggml hardware-test trees.
  The portable ggml placement gate passed.
- Ran the hardware qualification runner without a device. All four required
  computation gates skipped, and qualification correctly exited 1 with
  `primitive_gates_passed: false`. The passing placement and fixture tests did
  not turn this into an NPU pass. Host kernel review remained pending.
- Passed 44 runtime, model staging, client and qualification-report regression
  tests covering hash failures, wrong/missing device identity, backend
  registration, CPU flags, readiness/watchdog/shutdown and skipped-test handling.
- Loaded the SHA256-pinned SmolVLM-256M Q8_0 model and matching FP16 projector
  in CPU mode under a 2 GB container limit. Sent the bundled synthetic red-circle
  and blue-square image through `/v1/chat/completions`; the service returned
  `Red and Blue`. The client retained image hash, response, server timing fields
  and elapsed time. This is a smoke test of the vision path, not a model accuracy
  evaluation.
- `npu-demo check` identified the live CPU service; `inspect` correctly reported
  the 2 GB effective memory limit. A normal stop exited 0 and left the recorded
  run state stopped. Required Rocket startup without hardware exited 1 visibly.

The raw local results are under the ignored `artifacts/` and `data/` directories
when present. The runtime image is available locally as
`workload-npu-demo:local`; the test image is `workload-npu-demo:validation`.
The initial runtime was manually published on 10 October 2026 to private GHCR
as `ghcr.io/admrlos/workload-npu-demo:2026-10-10` and `:latest`, with digest
`sha256:2a562d5590266e7c276a5b9c7d6757506da67d3cf703f5e24621a64492abb575`.
The source is in [AdmrlOS/workload-npu-demo](https://github.com/AdmrlOS/workload-npu-demo).
No Admiral configuration was deployed. Registry publication does not change the
pending physical-board acceptance status.

Before deployment acceptance, use [the board procedure](acceptance.md) with the
boot-tested Admiral image and matching UAPI. Verify primitive results, independent
host kernel logs, model/projector accuracy, actual vision and language-prefill
Rocket execution, memory, startup/rollback, camera behavior and sustained
temperature/recovery on the physical Orange Pi 5.
