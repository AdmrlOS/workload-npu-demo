#!/usr/bin/env bash
# Run inside the validation image on the physical RK3588 board.
set -euo pipefail

if [[ $# -gt 1 || ${1:-} == --help ]]; then
    printf 'Usage: qualify.sh [output-directory]\n'
    printf 'Set QUALIFY_FULL_SUITE=1 to run all upstream tests.\n'
    printf 'ROCKET_TEST_DIR and GGML_TEST_DIR override the validation build directories.\n'
    printf 'HOST_KERNEL_LOG optionally names a separately captured host log for this run.\n'
    [[ ${1:-} == --help ]] && exit 0
    exit 2
fi

output=${1:-/data/qualification/$(date -u +%Y%m%dT%H%M%SZ)}
rocket_dir=${ROCKET_TEST_DIR:-/opt/validation/rocket}
ggml_dir=${GGML_TEST_DIR:-/opt/validation/ggml}
full=${QUALIFY_FULL_SUITE:-0}
for command in ctest python3 tee; do
    command -v "$command" >/dev/null || { printf 'Required program missing: %s\n' "$command" >&2; exit 1; }
done
for build in "$rocket_dir" "$ggml_dir"; do
    [[ -f $build/CTestTestfile.cmake ]] || { printf 'Validation build not found: %s\n' "$build" >&2; exit 1; }
done
mkdir -p "$output"
output=$(cd "$output" && pwd)
# A rerun must not accidentally consume a previous successful XML result.
for report in rocket.xml ggml.xml summary.json metadata.json; do
    [[ ! -e $output/$report ]] || { printf 'Use a fresh output directory; %s already exists.\n' "$report" >&2; exit 1; }
done

python3 - "$output/metadata.json" "$full" <<'PY'
import datetime
import json
import os
import platform
import pathlib
import sys

def read(path):
    try:
        return pathlib.Path(path).read_text(errors="replace").replace("\0", " ").strip()
    except OSError:
        return None

metadata = {
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "upstream primitive and ggml correctness gates; model acceptance remains separate",
    "uname": tuple(platform.uname()),
    "rocket_device": os.environ.get("ROCKET_DEV", "/dev/accel/accel0"),
    "board_model": read("/proc/device-tree/model"),
    "board_compatible": read("/proc/device-tree/compatible"),
    "meminfo": read("/proc/meminfo"),
    "cgroup": read("/proc/self/cgroup"),
    "memory_max": read("/sys/fs/cgroup/memory.max"),
    "memory_limit_v1": read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    "full_suite": sys.argv[2] == "1",
    "rocket_environment": {k: v for k, v in os.environ.items() if k.startswith("ROCKET_")},
    "source_manifest": read("/out/opt/npu-demo/provenance/sources.env"),
    "build_provenance": read("/out/opt/npu-demo/build-info.json"),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + "\n")
PY

run_suite() {
    local suite=$1 build=$2 required=$3
    local -a select=()
    [[ $full == 1 ]] || select=(-R "$required")
    # One test at a time prevents independent processes competing for the NPU.
    set +e
    ctest --test-dir "$build" --parallel 1 --output-on-failure \
        --no-tests=error --output-junit "$output/$suite.xml" "${select[@]}" \
        2>&1 | tee "$output/$suite.log"
    local statuses=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${statuses[0]}" > "$output/$suite.exit-code"
    if [[ ${statuses[1]} != 0 ]]; then
        printf 'Failed to save %s test log.\n' "$suite" >&2
        return 1
    fi
}

run_suite rocket "$rocket_dir" '^(matmul_tiled_rocket|matmul_mt_rocket|conv2d_fp16_rocket)$'
run_suite ggml "$ggml_dir" '^(test-rocket-matmul|test-rocket-placement)$'

python3 - "$output" "${HOST_KERNEL_LOG:-}" <<'PY'
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

directory = pathlib.Path(sys.argv[1])
required = {
    "rocket": {"matmul_tiled_rocket", "matmul_mt_rocket", "conv2d_fp16_rocket"},
    "ggml": {"test-rocket-matmul", "test-rocket-placement"},
}
hardware = required["rocket"] | {"test-rocket-matmul"}
summary = {"suites": {}, "errors": [], "hardware_passes": [], "host_kernel_review": "pending"}
for suite, gates in required.items():
    tests = {}
    try:
        root = ET.parse(directory / f"{suite}.xml").getroot()
        for test in root.iter("testcase"):
            name = test.attrib.get("name", "")
            status = test.attrib.get("status", "run").lower()
            skipped = test.find("skipped") is not None or status in {"notrun", "disabled", "skipped"}
            failed = test.find("failure") is not None or test.find("error") is not None or status == "fail"
            verdict = "skip" if skipped else "fail" if failed else "pass"
            tests[name] = {"verdict": verdict, "seconds": test.attrib.get("time")}
            if failed:
                summary["errors"].append(f"{suite}: {name} failed")
            if verdict == "pass" and name in hardware:
                summary["hardware_passes"].append(f"{suite}:{name}")
        if not tests:
            summary["errors"].append(f"{suite}: no tests were recorded")
        for name in sorted(gates):
            if name not in tests:
                summary["errors"].append(f"{suite}: required gate {name} is missing")
            elif tests[name]["verdict"] != "pass":
                summary["errors"].append(f"{suite}: required gate {name} did not run and pass ({tests[name]['verdict']})")
        exit_code = (directory / f"{suite}.exit-code").read_text().strip()
        if exit_code != "0":
            summary["errors"].append(f"{suite}: ctest exited {exit_code}")
    except (OSError, ET.ParseError) as exc:
        summary["errors"].append(f"{suite}: cannot read test results: {exc}")
    summary["suites"][suite] = {
        "counts": {v: sum(t["verdict"] == v for t in tests.values()) for v in ("pass", "fail", "skip")},
        "tests": tests,
    }

if not summary["hardware_passes"]:
    summary["errors"].append("No required hardware gate ran and passed; skipped tests never qualify the NPU.")
if sys.argv[2]:
    log = pathlib.Path(sys.argv[2])
    try:
        text = log.read_text(errors="replace")
        (directory / "host-kernel.log").write_text(text)
        pattern = re.compile(r"NPU job timed out|rk_iommu|WARNING:.*(?:rocket|drivers/accel|drivers/gpu/drm|drivers/iommu)", re.I)
        findings = [line for line in text.splitlines() if pattern.search(line)]
        summary["host_kernel_review"] = "fail" if findings else "captured; manually verify the capture covers this test interval"
        summary["kernel_findings"] = findings
        if findings:
            summary["errors"].append("Host kernel log contains a timeout, IOMMU report, or NPU warning; review host-kernel.log.")
    except OSError as exc:
        summary["errors"].append(f"Cannot read supplied host kernel log: {exc}")

summary["primitive_gates_passed"] = not summary["errors"]
summary["model_acceptance"] = "pending saved-image CPU/Rocket comparison and actual-offload evidence"
(directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if summary["host_kernel_review"] == "pending":
    print("Host kernel review is pending. Capture host logs independently; the journal fixture can skip on Admiral.", file=sys.stderr)
if summary["errors"]:
    sys.exit(1)
print(f"Primitive gates passed. Reports: {directory}. Model and sustained workload acceptance remain pending.")
PY
