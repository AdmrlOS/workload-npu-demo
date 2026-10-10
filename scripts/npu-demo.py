#!/usr/bin/env python3
"""Fail-closed llama.cpp service supervision for an Admiral Rocket workload.

Availability checks intentionally do not claim that an inference operation ran
on the NPU. Keep the emitted profile logs and run the hardware qualification.
"""

import argparse
import ctypes
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid


LLAMA_BIN = Path("/opt/llama/bin")
ROCKET_PLUGIN = Path("/opt/rocket/libggml-rocket.so")
STATE_PATH = Path("/run/npu-demo/state.json")
BUILD_INFO_PATH = Path("/opt/npu-demo/build-info.json")
SHUTDOWN_TIMEOUT = 20
HASH_CHUNK = 1024 * 1024


class DemoError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Config:
    mode: str = "rocket"
    manifest: str = "/models/model.json"
    host: str = "0.0.0.0"
    port: int = 8080
    context: int = 4096
    batch: int = 2048
    ubatch: int = 2048
    threads: int = 4
    parallel: int = 1
    startup_timeout: float = 120
    health_interval: float = 15
    health_failures: int = 3
    data_dir: str = "/data"
    rocket_dev: str = "/dev/accel/accel0"

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        defaults = cls()
        fields = {
            "mode": ("NPU_MODE", str),
            "manifest": ("NPU_MODEL_MANIFEST", str),
            "host": ("NPU_HOST", str),
            "port": ("NPU_PORT", int),
            "context": ("NPU_CONTEXT", int),
            "batch": ("NPU_BATCH", int),
            "ubatch": ("NPU_UBATCH", int),
            "threads": ("NPU_THREADS", int),
            "parallel": ("NPU_PARALLEL", int),
            "startup_timeout": ("NPU_STARTUP_TIMEOUT", float),
            "health_interval": ("NPU_HEALTH_INTERVAL", float),
            "health_failures": ("NPU_HEALTH_FAILURES", int),
            "data_dir": ("NPU_DATA_DIR", str),
            "rocket_dev": ("ROCKET_DEV", str),
        }
        values = {}
        for field, (key, convert) in fields.items():
            try:
                values[field] = convert(env.get(key, getattr(defaults, field)))
            except (TypeError, ValueError) as exc:
                raise DemoError(f"{key} has an invalid value") from exc
        config = cls(**values)
        if config.mode not in ("rocket", "cpu"):
            raise DemoError("NPU_MODE must be rocket or cpu")
        if not 1 <= config.port <= 65535:
            raise DemoError("NPU_PORT must be between 1 and 65535")
        for name in ("context", "batch", "ubatch", "threads", "parallel", "health_failures"):
            if getattr(config, name) < 1:
                raise DemoError(f"{fields[name][0]} must be positive")
        for name in ("startup_timeout", "health_interval"):
            value = getattr(config, name)
            if not 0 < value < float("inf"):
                raise DemoError(f"{fields[name][0]} must be finite and positive")
        if config.ubatch > config.batch:
            raise DemoError("NPU_UBATCH must not exceed NPU_BATCH")
        for name in ("manifest", "data_dir", "rocket_dev"):
            if not Path(getattr(config, name)).is_absolute():
                raise DemoError(f"{fields[name][0]} must be an absolute path")
        if not config.host or any(ch.isspace() for ch in config.host):
            raise DemoError("NPU_HOST must be a host name or IP address")
        return config

    @property
    def backend(self):
        return "ROCKET" if self.mode == "rocket" else "CPU"

    @property
    def health_url(self):
        host = self.host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1" if host == "0.0.0.0" else "::1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}/health"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path, max_bytes=1024 * 1024):
    try:
        with Path(path).open("rb") as source:
            data = source.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise DemoError(f"JSON file exceeds {max_bytes} bytes: {path}")
        return json.loads(data)
    except (OSError, ValueError) as exc:
        raise DemoError(f"Cannot read JSON {path}: {exc}") from exc


def startup_remaining(deadline, phase):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DemoError(f"Startup deadline exceeded during {phase}")
    return remaining


def validate_models(manifest_path, cancel=None, deadline=None):
    startup_remaining(deadline, "model validation")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise DemoError("Model manifest requires a non-empty name")
    if "model" not in manifest:
        raise DemoError("Model manifest requires a model entry")
    checked = {"name": manifest["name"]}
    for role in ("model", "projector"):
        if role == "projector" and role not in manifest:
            continue
        item = manifest[role]
        if not isinstance(item, dict):
            raise DemoError(f"{role} must be an object")
        path = item.get("path")
        checksum = item.get("sha256")
        source = item.get("source")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise DemoError(f"{role}.path must be absolute")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
            raise DemoError(f"{role}.sha256 must contain exactly 64 hexadecimal characters")
        if not isinstance(source, str) or not source.strip():
            raise DemoError(f"{role}.source must identify the pinned source")
        try:
            if not stat.S_ISREG(os.stat(path).st_mode):
                raise DemoError(f"{role} must be a regular file: {path}")
            with open(path, "rb") as model:
                before = os.fstat(model.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise DemoError(f"{role} must be a regular file: {path}")
                if model.read(4) != b"GGUF":
                    raise DemoError(f"{role} does not have GGUF magic: {path}")
                model.seek(0)
                digest = hashlib.sha256()
                while True:
                    if cancel is not None and cancel.is_set():
                        raise DemoError("Model validation interrupted")
                    startup_remaining(deadline, "model checksum validation")
                    chunk = model.read(HASH_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(model.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise DemoError(f"{role} changed during checksum validation: {path}")
        except OSError as exc:
            raise DemoError(f"Cannot open {role} {path}: {exc}") from exc
        actual = digest.hexdigest()
        if actual != checksum.lower():
            raise DemoError(f"SHA-256 mismatch for {role} {path}: expected {checksum.lower()}, got {actual}")
        checked[role] = {"path": path, "sha256": actual, "source": source, "bytes": after.st_size}
    return checked


def runtime_env(config, has_projector=False, environ=None):
    env = dict(os.environ if environ is None else environ)
    # All application decisions come from the validated NPU_* configuration.
    # Inherited llama flags could otherwise add downloads or change the CPU arm.
    env = {key: value for key, value in env.items() if not key.startswith("LLAMA_ARG_")}
    env.pop("GGML_BACKEND_PATH", None)
    env.pop("MTMD_BACKEND_DEVICE", None)
    old_lib = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(LLAMA_BIN) + (":" + old_lib if old_lib else "")
    env["PATH"] = str(LLAMA_BIN) + ":" + env.get("PATH", "/usr/bin:/bin")
    if config.mode == "rocket":
        env["GGML_BACKEND_PATH"] = str(ROCKET_PLUGIN)
        env["ROCKET_DEV"] = config.rocket_dev
        env.setdefault("ROCKET_LOG_STDERR", "1")
        env.setdefault("ROCKET_MM_PROFILE", "1")
        if has_projector:
            env["MTMD_BACKEND_DEVICE"] = "ROCKET"
    return env


class DrmVersion(ctypes.Structure):
    # This is libdrm's drmVersion from xf86drm.h, not drm_version from drm.h.
    _fields_ = [
        ("version_major", ctypes.c_int), ("version_minor", ctypes.c_int),
        ("version_patchlevel", ctypes.c_int), ("name_len", ctypes.c_int),
        ("name", ctypes.c_char_p), ("date_len", ctypes.c_int),
        ("date", ctypes.c_char_p), ("desc_len", ctypes.c_int),
        ("desc", ctypes.c_char_p),
    ]


def rocket_driver_name(device):
    """Open the actual node and ask libdrm for its DRM_IOCTL_VERSION identity."""
    try:
        drm = ctypes.CDLL("libdrm.so.2", use_errno=True)
    except OSError as exc:
        raise DemoError(f"Cannot load libdrm.so.2 for the Rocket identity check: {exc}") from exc
    drm.drmGetVersion.argtypes = [ctypes.c_int]
    drm.drmGetVersion.restype = ctypes.POINTER(DrmVersion)
    drm.drmFreeVersion.argtypes = [ctypes.POINTER(DrmVersion)]
    drm.drmFreeVersion.restype = None
    try:
        fd = os.open(device, os.O_RDWR | os.O_CLOEXEC)
    except OSError as exc:
        raise DemoError(f"Cannot open Rocket device {device} read/write: {exc}") from exc
    version = None
    try:
        if not stat.S_ISCHR(os.fstat(fd).st_mode):
            raise DemoError(f"Rocket device is not a character device: {device}")
        version = drm.drmGetVersion(fd)
        if not version:
            error = ctypes.get_errno()
            raise DemoError(f"DRM_IOCTL_VERSION failed for {device}: {os.strerror(error)}")
        name = version.contents.name
        return name.decode("ascii", errors="replace") if name else ""
    finally:
        if version:
            drm.drmFreeVersion(version)
        os.close(fd)


def check_rocket_device(config):
    try:
        mode = os.stat(config.rocket_dev).st_mode
    except OSError as exc:
        raise DemoError(f"Required Rocket device {config.rocket_dev} is unavailable: {exc}") from exc
    if not stat.S_ISCHR(mode):
        raise DemoError(f"Required Rocket node is not a character device: {config.rocket_dev}")
    name = rocket_driver_name(config.rocket_dev)
    if name != "rocket":
        raise DemoError(f"{config.rocket_dev} reports DRM driver {name!r}; expected 'rocket'")
    return name


def preflight(config, deadline=None):
    startup_remaining(deadline, "backend preflight")
    driver = None
    if config.mode == "rocket":
        check_rocket_device(config)
        driver = "rocket"
        if not ROCKET_PLUGIN.is_file():
            raise DemoError(f"Required ggml Rocket plugin is missing: {ROCKET_PLUGIN}")
    cli = LLAMA_BIN / "llama-cli"
    if not os.access(cli, os.X_OK):
        raise DemoError(f"llama-cli is missing or not executable: {cli}")
    remaining = startup_remaining(deadline, "backend enumeration")
    timeout = min(30, remaining) if remaining is not None else 30
    try:
        result = subprocess.run([str(cli), "--list-devices"], env=runtime_env(config),
                                capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DemoError(f"Backend enumeration failed: {exc}") from exc
    # Match only the actual device list, never a log mentioning a failed plugin.
    output = result.stdout
    inventory = output.partition("Available devices:")[2]
    devices = re.findall(r"^\s+([A-Za-z0-9_.-]+):\s+[^\n]+$", inventory, re.MULTILINE)
    if result.returncode:
        raise DemoError(f"llama-cli --list-devices exited {result.returncode}: {result.stderr[-4096:]}")
    if config.mode == "rocket" and "ROCKET" not in devices:
        raise DemoError("The Rocket plugin did not register a ROCKET device; refusing CPU fallback. "
                        + (result.stdout + result.stderr)[-4096:])
    startup_remaining(deadline, "backend enumeration")
    return {"requested": config.backend, "selected": config.backend,
            "available": ["CPU"] + devices, "driver": driver,
            "device": config.rocket_dev if driver else None,
            "npu_execution_verified": False,
            "enumeration": (result.stdout + result.stderr)[-16384:]}


def server_command(config, models):
    command = [str(LLAMA_BIN / "llama-server"), "-m", models["model"]["path"],
               "-ngl", "0", "-b", str(config.batch), "-ub", str(config.ubatch),
               "-c", str(config.context), "--host", config.host, "--port", str(config.port),
               "--threads", str(config.threads), "--parallel", str(config.parallel)]
    if config.mode == "rocket":
        command += ["-nr"]
    else:
        command += ["--device", "none", "--no-op-offload"]
    if "projector" in models:
        command += ["--mmproj", models["projector"]["path"]]
        if config.mode == "rocket":
            command += ["--mmproj-device", "ROCKET"]
        else:
            command += ["--no-mmproj-offload"]
    return command


def health_probe(config, timeout=2):
    try:
        # A local health probe must not be redirected through an inherited proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(config.health_url, timeout=timeout) as response:
            body = response.read(4097)
            if response.status != 200 or len(body) > 4096:
                return False
            return json.loads(body).get("status") == "ok"
    except (OSError, ValueError, AttributeError, urllib.error.URLError):
        return False


def ensure_port_available(config):
    try:
        addresses = socket.getaddrinfo(config.host, config.port, type=socket.SOCK_STREAM,
                                      flags=socket.AI_PASSIVE)
        family, kind, protocol, _, address = addresses[0]
        with socket.socket(family, kind, protocol) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(address)
    except OSError as exc:
        raise DemoError(f"Cannot bind {config.host}:{config.port}; another service may own the port: {exc}") from exc


def atomic_state(state, path=None):
    path = STATE_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".state-", delete=False) as dest:
            temporary = Path(dest.name)
            json.dump(state, dest, indent=2, sort_keys=True)
            dest.write("\n")
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_identity(pid):
    try:
        # The command in /proc/PID/stat can contain spaces and parentheses.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if tail[0] == "Z":
            return None
        return tail[19]
    except (OSError, IndexError):
        return None


def tracked_processes_live(state):
    for pid_key, identity_key in (("supervisor_pid", "supervisor_identity"),
                                   ("server_pid", "server_identity")):
        pid = state.get(pid_key)
        identity = state.get(identity_key)
        if not isinstance(pid, int) or pid < 1 or not identity or process_identity(pid) != identity:
            return False
    try:
        actual = Path(f"/proc/{state['server_pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return [arg.decode(errors="replace") for arg in actual] == state.get("command")
    except OSError:
        return False


def read_optional(path, max_bytes=65536):
    try:
        return Path(path).read_bytes()[:max_bytes].decode(errors="replace").replace("\0", "\n").strip()
    except OSError:
        return None


def system_info():
    meminfo = read_optional("/proc/meminfo") or ""
    match = re.search(r"^MemTotal:\s+(\d+) kB$", meminfo, re.MULTILINE)
    candidates = [Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]
    for line in (read_optional("/proc/self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        relative = parts[2].lstrip("/")
        if not parts[1]:
            mount, filename = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in parts[1].split(","):
            mount, filename = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        # A parent cgroup can impose a tighter limit than the leaf. A container
        # cgroup namespace may instead expose only the mounted root limit.
        group = mount / relative
        while group.is_relative_to(mount):
            candidates.append(group / filename)
            if group == mount:
                break
            group = group.parent
    limits = []
    for path in candidates:
        value = read_optional(path)
        if value and value.isdecimal() and 0 < int(value) < (1 << 60):
            limits.append(int(value))
    host_ram = int(match.group(1)) * 1024 if match else None
    limit = min(limits) if limits else None
    return {"uname": platform.uname()._asdict(),
            "board": read_optional("/proc/device-tree/model"),
            "compatible": (read_optional("/proc/device-tree/compatible") or "").splitlines(),
            "host_ram_bytes": host_ram, "cgroup_memory_limit_bytes": limit,
            "effective_memory_budget_bytes": min(v for v in (host_ram, limit) if v is not None)
            if host_ram is not None or limit is not None else None}


def inspect(config):
    try:
        backend = preflight(config)
    except DemoError as exc:
        backend = {"requested": config.backend, "selected": None, "available": [],
                   "error": str(exc), "npu_execution_verified": False}
    try:
        state = read_json(STATE_PATH)
    except DemoError:
        state = None
    try:
        build_info = read_json(BUILD_INFO_PATH)
    except DemoError:
        build_info = None
    return {"timestamp": utc_now(), "config": dataclasses.asdict(config), "system": system_info(),
            "backend": backend, "state": state, "build": build_info,
            "npu_execution_verified": False,
            "evidence_note": "Device identity and plugin registration establish availability only. "
            "NPU execution requires primitive tests and per-operation profile evidence."}


def check(config):
    backend = preflight(config)
    state = read_json(STATE_PATH)
    if not isinstance(state, dict) or not state.get("ready") or state.get("status") != "ready" or state.get("logging_failure"):
        raise DemoError("The tracked llama-server has not reached ready state")
    if state.get("mode") != config.mode or state.get("host") != config.host or state.get("port") != config.port:
        raise DemoError("Supervisor state does not match the requested backend/listener")
    if state.get("selected_backend") != backend["selected"]:
        raise DemoError("Supervisor backend selection does not match current availability")
    if not tracked_processes_live(state):
        raise DemoError("Tracked supervisor/server identity is stale or no longer running")
    if not health_probe(config):
        raise DemoError("The tracked llama-server /health endpoint is not ready")
    return {"ready": True, "availability_verified": True,
            "selected_backend": config.backend, "npu_execution_verified": False}


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.logging_failure_event = threading.Event()
        self.logging_failure = None
        self.stop_signal = signal.SIGTERM
        self.process = None
        self.log_file = None
        self.log_lock = threading.Lock()
        self.tee_thread = None
        self.run_state_path = None
        self.state = {"version": 1, "status": "preflight", "ready": False,
                      "mode": config.mode, "host": config.host, "port": config.port,
                      "requested_backend": config.backend, "selected_backend": None,
                      "npu_execution_verified": False, "supervisor_pid": os.getpid(),
                      "supervisor_identity": process_identity(os.getpid()), "started_at": utc_now(),
                      "system": system_info()}
        try:
            self.state["build"] = read_json(BUILD_INFO_PATH)
        except DemoError:
            self.state["build"] = None

    def update(self, **values):
        self.state.update(values)
        if self.logging_failure is not None:
            self.state["logging_failure"] = self.logging_failure
        self.state["updated_at"] = utc_now()
        atomic_state(self.state)
        if self.run_state_path is not None:
            try:
                atomic_state(self.state, self.run_state_path)
            except OSError as exc:
                self.mark_logging_failure(f"Persistent run metadata write failed: {exc}")
                self.state["logging_failure"] = self.logging_failure
                atomic_state(self.state)

    def mark_logging_failure(self, message):
        if self.logging_failure is None:
            self.logging_failure = message
        self.logging_failure_event.set()
        self.wake.set()

    def check_logging(self):
        if self.logging_failure_event.is_set():
            raise DemoError(self.logging_failure)

    def wait(self, duration):
        self.wake.wait(duration)
        self.check_logging()
        return self.stop.is_set()

    @staticmethod
    def stdout(data):
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            else:
                sys.stdout.write(data.decode(errors="replace"))
                sys.stdout.flush()
        except (OSError, ValueError):
            pass

    def output(self, data):
        with self.log_lock:
            if self.log_file is not None:
                try:
                    self.log_file.write(data)
                    self.log_file.flush()
                except (OSError, ValueError) as exc:
                    failed_file, self.log_file = self.log_file, None
                    self.mark_logging_failure(f"Persistent server log write failed: {exc}")
                    try:
                        failed_file.close()
                    except (OSError, ValueError):
                        pass
                    # Never call log() here: that would retry the failed file.
                    self.stdout(f"npu-demo: {self.logging_failure}\n".encode())
            self.stdout(data)

    def log(self, message):
        self.output(f"[{utc_now()}] npu-demo: {message}\n".encode())

    def tee(self):
        try:
            while True:
                chunk = self.process.stdout.read1(65536)
                if not chunk:
                    break
                self.output(chunk)
        except (OSError, ValueError) as exc:
            self.mark_logging_failure(f"Server log stream failed: {exc}")
            self.stdout(f"npu-demo: {self.logging_failure}\n".encode())

    def forward(self, signum):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signum)
            except ProcessLookupError:
                pass

    def on_signal(self, signum, _frame):
        self.stop_signal = signum
        self.stop.set()
        self.wake.set()
        self.forward(signum)

    def shutdown(self):
        if self.process is None:
            return
        self.forward(self.stop_signal)
        try:
            self.process.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.log("Graceful shutdown deadline exceeded; killing the server process group")
            self.forward(signal.SIGKILL)
            self.process.wait(timeout=5)
        if self.tee_thread is not None:
            self.tee_thread.join(timeout=5)
            if self.tee_thread.is_alive():
                # A surviving descendant can retain the pipe after the main
                # process exits. Close on a reader in another thread can block.
                self.forward(signal.SIGKILL)
                self.tee_thread.join(timeout=1)
            if self.tee_thread.is_alive():
                return
        self.process.stdout.close()

    def run(self):
        deadline = time.monotonic() + self.config.startup_timeout
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_handlers:
            signal.signal(sig, self.on_signal)
        result = 1
        try:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_dir = Path(self.config.data_dir) / "runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=False)
            self.log_file = (run_dir / "server.log").open("ab", buffering=0)
            self.run_state_path = run_dir / "state.json"
            self.update(log_path=str(run_dir / "server.log"), run_state_path=str(self.run_state_path))
            self.check_logging()
            models = validate_models(self.config.manifest, self.stop, deadline)
            if self.stop.is_set():
                raise DemoError("Startup interrupted")
            backend = preflight(self.config, deadline)
            self.update(models=models, backend=backend, selected_backend=backend["selected"])
            self.log("Backend availability verified; NPU operation execution remains unverified")
            self.log("Backend enumeration:\n" + backend["enumeration"])
            self.check_logging()
            ensure_port_available(self.config)
            startup_remaining(deadline, "server launch")
            command = server_command(self.config, models)
            if not os.access(command[0], os.X_OK):
                raise DemoError(f"llama-server is missing or not executable: {command[0]}")
            if self.stop.is_set():
                raise DemoError("Startup interrupted")
            self.log("Starting " + json.dumps(command))
            self.check_logging()
            self.process = subprocess.Popen(command, env=runtime_env(self.config, "projector" in models),
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            self.tee_thread = threading.Thread(target=self.tee, name="llama-log", daemon=True)
            self.tee_thread.start()
            self.update(status="loading", command=command, server_pid=self.process.pid,
                        server_identity=process_identity(self.process.pid))
            while not self.stop.is_set():
                self.check_logging()
                if self.process.poll() is not None:
                    raise DemoError(f"llama-server exited during startup with status {self.process.returncode}")
                remaining = startup_remaining(deadline, "llama-server readiness")
                healthy = health_probe(self.config, timeout=min(2, remaining))
                self.check_logging()
                if healthy and self.process.poll() is None and not self.stop.is_set():
                    startup_remaining(deadline, "llama-server readiness")
                    self.update(status="ready", ready=True, health_failures=0)
                    self.log("llama-server is ready; profile evidence is still required to prove NPU execution")
                    break
                self.wait(min(1, max(0, deadline - time.monotonic())))
            failures = 0
            while not self.wait(self.config.health_interval):
                if self.process.poll() is not None:
                    raise DemoError(f"llama-server exited with status {self.process.returncode}")
                reason = "llama-server /health failed"
                healthy = health_probe(self.config)
                if self.config.mode == "rocket":
                    try:
                        check_rocket_device(self.config)
                    except DemoError as exc:
                        healthy, reason = False, str(exc)
                self.check_logging()
                failures = 0 if healthy else failures + 1
                self.update(status="ready" if healthy else "degraded", ready=healthy,
                            health_failures=failures, last_health_error=None if healthy else reason)
                if failures >= self.config.health_failures:
                    raise DemoError(f"Watchdog failed {failures} consecutive checks: {reason}")
            result = 0
            self.update(status="stopping", ready=False)
        except (DemoError, OSError) as exc:
            if self.stop.is_set() and not self.logging_failure_event.is_set():
                result = 0
                self.update(status="stopping", ready=False)
            else:
                self.log(str(exc))
                self.update(status="failed", ready=False, error=str(exc))
        finally:
            self.shutdown()
            with self.log_lock:
                if self.log_file is not None:
                    try:
                        self.log_file.close()
                    except OSError as exc:
                        self.mark_logging_failure(f"Persistent server log close failed: {exc}")
                    self.log_file = None
            if self.logging_failure_event.is_set():
                result = 1
                self.update(status="failed", ready=False, error=self.logging_failure)
            elif result == 0:
                self.update(status="stopped", ready=False)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="serve",
                        choices=("serve", "inspect", "check", "validate-models"))
    args = parser.parse_args(argv)
    try:
        config = Config.from_env()
        if args.command == "serve":
            return Supervisor(config).run()
        if args.command == "inspect":
            result = inspect(config)
        elif args.command == "check":
            result = check(config)
        else:
            result = validate_models(config.manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (DemoError, OSError) as exc:
        print(json.dumps({"ready": False, "error": str(exc), "npu_execution_verified": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
