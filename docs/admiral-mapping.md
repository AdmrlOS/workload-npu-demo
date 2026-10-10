# Deploying the Rocket workload on Admiral

Build and publish a `linux/arm64` OCI image, then select its immutable digest in
**Configs > New Configuration**. Use
[`config/admiral-dashboard.yaml`](../config/admiral-dashboard.yaml) as an
**illustrative mapping of dashboard fields**. It is not an API request or an
import format. Configure [registry credentials](https://docs.admrl.co/core-concepts/registry-credentials/)
when the image is private.

| Dashboard field | Value | Purpose |
| --- | --- | --- |
| Command | Empty | Keep the image's entrypoint and supervision. |
| Interactive Mode | Off | Run a headless service. |
| Device Isolation | Off | Expose the Rocket accelerator and camera devices. |
| Host Network | Off | Use the normal workload network. |
| Read-only Root | Off | Allow startup state and temporary files. |
| Environment | `NPU_MODEL_MANIFEST=/models/model.json`, `NPU_MODE=rocket` | Load the selected model with the Rocket mode. |
| Custom Volumes | `npu-models` at `/models`, `npu-data` at `/data` | Preserve models, manifests, inputs, and results. |
| Port | TCP 8080 to TCP 8080 | Reach the API through the device's published port. |

Keep credentials in Admiral Secret Files where supported by the application;
restrict access to the published inference API through the deployment's network
policy. Do not replace installed code or libraries with a data-volume mount.

## Launch and health behavior

Admiral starts the process at `/`, with root privileges. The image uses absolute
paths. Setting Command replaces both image ENTRYPOINT and CMD. Docker
`HEALTHCHECK` is not executed: the workload must supervise its server and exit
its primary process when its watchdog detects a persistent failure. Admiral
then applies its restart and rollout policy. Rootfs writes are lost on image
replacement; Custom Volumes persist. See the
[launch contract](https://docs.admrl.co/core-concepts/configurations/).
Admiral's three-minute process-uptime probation is separate from application
readiness. The default 120-second whole-startup deadline leaves time to stop
within that window; a longer override requires separate rollout/readiness
qualification.

The watchdog establishes that the server remains responsive. `npu-demo inspect`
reports diagnostics and stored supervisor state; `npu-demo check` verifies the
tracked server's readiness and current plugin/device availability. Startup
validates model hashes; hardware acceptance separately checks discovery paths.
These results do not prove that a submitted model used the NPU.
Record actual Rocket execution/profile counters during a request, compare the
same input with CPU mode, and retain correctness and timing results before
accepting an NPU-assisted model. A visible node or registered backend alone is
insufficient evidence of offload.

## Required host interfaces

The kernel-team handoff requires the following visibility inside the workload:

| Interface | Requirement |
| --- | --- |
| `/dev/accel/accelN` | Read/write Rocket device access; select the actual node, not an assumed number. |
| `/sys/bus/platform/drivers/rocket/` and resolved device links | Readable bound-core discovery. |
| `/sys/module/rocket/parameters/`, where present | Readable capability discovery for the deployed driver. |
| `/sys/devices/system/cpu/` | Readable CPU topology and frequency discovery. |
| `/proc/meminfo` | Readable memory information used by userspace heuristics. |
| `/proc/device-tree/compatible`, or bound-device `of_node` data | Readable RK3588/RK3588S identification. |
| Effective cgroup memory-limit files | Discover the workload budget rather than treating host RAM as its budget. |
| `/dev/video*`, and `/dev/media*` where required | Camera capture for camera-based use. |

Read-only sysfs and procfs visibility is sufficient for the normal Rocket
discovery described in the handoff. Inspect the **actual deployed workload**:
unrestricted device access alone does not establish that every sysfs link,
device-tree path, or cgroup file resolves inside its namespace. Confirm that the
accelerator's DRM identity reports `rocket`. Missing optional driver parameters
must not be interpreted as patched-driver capabilities. One accelerator node
can represent all three NPU cores.

Admiral documents access to `/sys/class`, `/sys/devices`, `/sys/bus`, and
`/sys/module`; it does not expose debugfs or tracefs. If a required discovery
path is missing, resolve that runtime/BSP contract with the platform team.
Ordinary host directory mounts are not accepted as System Mounts; do not add
ad-hoc mounts of `/sys`, `/proc`, or host libraries. Collect module identity,
signature/probe failures, and kernel errors from the host separately. See the
[platform boundary](https://docs.admrl.co/core-concepts/platform-boundary/).

The BSP supplies the signed Rocket driver, matching kernel UAPI, device tree,
and validated clock/power policy. The workload supplies llama.cpp,
`librocketnpu`, its matching ggml plugin, and model/projector files. Do not install
a kernel driver or proprietary RKNN runtime into this workload, bind the host's
musl libraries over container libraries, or change accelerator clocks from an
application startup script.

## API automation caveat

The current [Swagger schema](https://api.admrl.co/v1/swagger/doc.json) differs from
the dashboard examples: its specification uses `environment`, a volume map with
`destination`/`source`/`options`, ports with `port` and lowercase `protocol`, and
options such as `all_devices_allowed`, `read_only_root`, `tty`, and
`host_namespaces`. Configuration creation wraps the specification in
`{"name": "...", "spec": {...}}`; unknown fields are rejected. Verify the
dashboard-to-API translation against the deployed Admiral version before
automating deployment. Do not post the accompanying illustrative YAML to the
API.
