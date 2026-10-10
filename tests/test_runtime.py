"""Host-independent contract tests; these do not qualify actual NPU execution."""

import ctypes
import dataclasses
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "npu-demo.py"
SPEC = importlib.util.spec_from_file_location("npu_demo", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = runtime.Config(mode="cpu", manifest=str(self.root / "model.json"),
                                     data_dir=str(self.root / "data"), startup_timeout=0.02,
                                     health_interval=0.001, health_failures=2)
        self.models = {"name": "test", "model": self.artifact("model.gguf")}
        (self.root / "model.json").write_text(json.dumps(self.models))
        self.addCleanup(self.temp.cleanup)

    def artifact(self, name, data=b"GGUF" + b"test fixture" * 4096):
        path = self.root / name
        path.write_bytes(data)
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                "source": "https://example.test/models/pinned-revision/" + name}


class RuntimeTests(RuntimeFixture):
    def test_valid_model_and_projector_are_verified(self):
        self.models["projector"] = self.artifact("mmproj.gguf")
        (self.root / "model.json").write_text(json.dumps(self.models))
        actual = runtime.validate_models(self.config.manifest)
        self.assertEqual(actual["model"]["sha256"], self.models["model"]["sha256"])
        self.assertGreater(actual["projector"]["bytes"], 4)

    def test_hash_mismatch_fails_before_inference(self):
        self.models["model"]["sha256"] = "0" * 64
        (self.root / "model.json").write_text(json.dumps(self.models))
        with self.assertRaisesRegex(runtime.DemoError, "SHA-256 mismatch"):
            runtime.validate_models(self.config.manifest)

    def test_non_gguf_fails_even_when_hash_matches(self):
        self.models["model"] = self.artifact("model.gguf", b"not a model")
        (self.root / "model.json").write_text(json.dumps(self.models))
        with self.assertRaisesRegex(runtime.DemoError, "GGUF magic"):
            runtime.validate_models(self.config.manifest)

    def test_relative_model_path_is_rejected(self):
        self.models["model"]["path"] = "model.gguf"
        (self.root / "model.json").write_text(json.dumps(self.models))
        with self.assertRaisesRegex(runtime.DemoError, "must be absolute"):
            runtime.validate_models(self.config.manifest)

    def test_invalid_config_is_rejected(self):
        for env in ({"NPU_MODE": "vulkan"}, {"NPU_PORT": "0"}, {"NPU_STARTUP_TIMEOUT": "nan"},
                    {"NPU_BATCH": "10", "NPU_UBATCH": "20"}, {"NPU_THREADS": "0"}, {"NPU_PARALLEL": "0"}):
            with self.subTest(env=env), self.assertRaises(runtime.DemoError):
                runtime.Config.from_env(env)

    def test_default_is_one_slot_and_120_second_entire_startup_budget(self):
        config = runtime.Config.from_env({})
        self.assertEqual(config.startup_timeout, 120)
        self.assertEqual(config.parallel, 1)
        command = runtime.server_command(config, self.models)
        self.assertEqual(command[command.index("--parallel") + 1], "1")

    def test_hashing_checks_whole_startup_deadline(self):
        with mock.patch.object(runtime.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(runtime.DemoError, "deadline exceeded during model checksum"):
                runtime.validate_models(self.config.manifest, deadline=1)

    def test_cpu_arm_preserves_repacking_and_disables_all_offload(self):
        models = dict(self.models, projector=self.artifact("mmproj.gguf"))
        command = runtime.server_command(self.config, models)
        self.assertNotIn("-nr", command)
        self.assertNotIn("--mmproj-device", command)
        self.assertIn("--no-mmproj-offload", command)
        self.assertIn("--no-op-offload", command)
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertEqual(command[command.index("-ngl") + 1], "0")
        env = runtime.runtime_env(self.config, True, {"GGML_BACKEND_PATH": "bad.so",
                 "MTMD_BACKEND_DEVICE": "ROCKET", "LLAMA_ARG_NO_REPACK": "1",
                 "LD_LIBRARY_PATH": "/existing"})
        self.assertNotIn("GGML_BACKEND_PATH", env)
        self.assertNotIn("MTMD_BACKEND_DEVICE", env)
        self.assertNotIn("LLAMA_ARG_NO_REPACK", env)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/llama/bin:/existing")

    def test_rocket_selects_projector_and_eligible_host_operations(self):
        config = dataclasses.replace(self.config, mode="rocket")
        models = dict(self.models, projector=self.artifact("mmproj.gguf"))
        command = runtime.server_command(config, models)
        self.assertIn("-nr", command)
        self.assertEqual(command[command.index("--mmproj-device") + 1], "ROCKET")
        self.assertNotIn("--no-op-offload", command)
        self.assertNotIn("--no-mmproj-offload", command)
        self.assertEqual(command[command.index("-b") + 1], "2048")
        self.assertEqual(command[command.index("-ub") + 1], "2048")
        env = runtime.runtime_env(config, True, {})
        self.assertEqual(env["GGML_BACKEND_PATH"], "/opt/rocket/libggml-rocket.so")
        self.assertEqual(env["MTMD_BACKEND_DEVICE"], "ROCKET")

    def test_absent_rocket_device_never_falls_back_to_cpu(self):
        config = dataclasses.replace(self.config, mode="rocket", rocket_dev=str(self.root / "absent"))
        with self.assertRaisesRegex(runtime.DemoError, "Required Rocket device"):
            runtime.preflight(config)

    def test_regular_file_cannot_impersonate_rocket_device(self):
        config = dataclasses.replace(self.config, mode="rocket", rocket_dev=self.models["model"]["path"])
        with self.assertRaisesRegex(runtime.DemoError, "not a character device"):
            runtime.preflight(config)

    def test_wrong_drm_driver_is_rejected(self):
        config = dataclasses.replace(self.config, mode="rocket", rocket_dev="/dev/null")
        with mock.patch.object(runtime, "rocket_driver_name", return_value="panthor"):
            with self.assertRaisesRegex(runtime.DemoError, "expected 'rocket'"):
                runtime.check_rocket_device(config)

    def test_libdrm_identity_uses_correct_abi_and_frees_result(self):
        self.assertEqual(runtime.DrmVersion.name_len.size, ctypes.sizeof(ctypes.c_int))
        version = runtime.DrmVersion(version_major=1, name_len=6, name=b"rocket")
        drm = mock.Mock()
        drm.drmGetVersion.return_value = ctypes.pointer(version)
        with mock.patch.object(runtime.ctypes, "CDLL", return_value=drm):
            self.assertEqual(runtime.rocket_driver_name("/dev/null"), "rocket")
        drm.drmFreeVersion.assert_called_once()
        fd = drm.drmGetVersion.call_args.args[0]
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_libdrm_failed_ioctl_is_not_availability(self):
        drm = mock.Mock()
        drm.drmGetVersion.return_value = ctypes.POINTER(runtime.DrmVersion)()
        with mock.patch.object(runtime.ctypes, "CDLL", return_value=drm):
            with self.assertRaisesRegex(runtime.DemoError, "DRM_IOCTL_VERSION failed"):
                runtime.rocket_driver_name("/dev/null")

    def enumeration_context(self, output):
        binaries = self.root / "bin"
        binaries.mkdir(exist_ok=True)
        cli = binaries / "llama-cli"
        cli.write_text("unused mock")
        cli.chmod(0o755)
        plugin = self.root / "libggml-rocket.so"
        plugin.touch()
        return mock.patch.multiple(runtime, LLAMA_BIN=binaries, ROCKET_PLUGIN=plugin), \
            mock.patch.object(runtime, "check_rocket_device", return_value="rocket"), \
            mock.patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess(
                [str(cli), "--list-devices"], 0, output, ""))

    def test_plugin_log_without_enumerated_device_is_rejected(self):
        config = dataclasses.replace(self.config, mode="rocket")
        a, b, c = self.enumeration_context("Failed to initialize ROCKET: missing NPU\nAvailable devices:\n  (none)\n")
        with a, b, c, self.assertRaisesRegex(runtime.DemoError, "refusing CPU fallback"):
            runtime.preflight(config)

    def test_real_device_listing_still_does_not_claim_operations(self):
        config = dataclasses.replace(self.config, mode="rocket")
        a, b, c = self.enumeration_context("Available devices:\n  ROCKET: Rockchip NPU (0 MiB, 0 MiB free)\n")
        with a, b, c:
            backend = runtime.preflight(config)
        self.assertEqual(backend["selected"], "ROCKET")
        self.assertEqual(backend["driver"], "rocket")
        self.assertFalse(backend["npu_execution_verified"])

    def test_enumeration_timeout_is_capped_to_remaining_startup_budget(self):
        config = dataclasses.replace(self.config, mode="rocket")
        a, b, c = self.enumeration_context("Available devices:\n  ROCKET: Rockchip NPU (0 MiB, 0 MiB free)\n")
        with a, b, c as enumerate_process, mock.patch.object(runtime.time, "monotonic", return_value=100):
            runtime.preflight(config, deadline=103)
            self.assertEqual(enumerate_process.call_args.kwargs["timeout"], 3)

    def test_healthy_http_does_not_override_missing_acceleration(self):
        config = dataclasses.replace(self.config, mode="rocket")
        with mock.patch.object(runtime, "preflight", side_effect=runtime.DemoError("Rocket missing")), \
                mock.patch.object(runtime, "health_probe", return_value=True) as health:
            with self.assertRaisesRegex(runtime.DemoError, "Rocket missing"):
                runtime.check(config)
            health.assert_not_called()

    def test_healthy_unrelated_server_does_not_override_stale_process(self):
        state = {"ready": True, "status": "ready", "mode": "cpu", "host": self.config.host,
                 "port": self.config.port, "selected_backend": "CPU"}
        with mock.patch.object(runtime, "preflight", return_value={"selected": "CPU"}), \
                mock.patch.object(runtime, "read_json", return_value=state), \
                mock.patch.object(runtime, "tracked_processes_live", return_value=False), \
                mock.patch.object(runtime, "health_probe", return_value=True) as health:
            with self.assertRaisesRegex(runtime.DemoError, "identity is stale"):
                runtime.check(self.config)
            health.assert_not_called()

    def test_existing_listener_is_rejected_before_startup(self):
        listener = mock.Mock()
        listener.bind.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
        socket_context = mock.MagicMock()
        socket_context.__enter__.return_value = listener
        config = dataclasses.replace(self.config, host="127.0.0.1", port=8080)
        with mock.patch.object(runtime.socket, "socket", return_value=socket_context):
            with self.assertRaisesRegex(runtime.DemoError, "another service"):
                runtime.ensure_port_available(config)
        listener.bind.assert_called_once_with(("127.0.0.1", 8080))

    def test_atomic_state_is_readable_without_leftover_temporary(self):
        path = self.root / "run" / "state.json"
        runtime.atomic_state({"status": "ready", "ready": True}, path)
        self.assertEqual(json.loads(path.read_text())["status"], "ready")
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_memory_budget_accounts_for_tighter_parent_cgroup(self):
        values = {"/proc/meminfo": "MemTotal: 8192 kB\n", "/proc/self/cgroup": "0::/tenant/job\n",
                  "/sys/fs/cgroup/memory.max": "max", "/sys/fs/cgroup/tenant/job/memory.max": "max",
                  "/sys/fs/cgroup/tenant/memory.max": "4096"}
        with mock.patch.object(runtime, "read_optional", side_effect=lambda path: values.get(str(path))):
            info = runtime.system_info()
        self.assertEqual(info["host_ram_bytes"], 8192 * 1024)
        self.assertEqual(info["cgroup_memory_limit_bytes"], 4096)
        self.assertEqual(info["effective_memory_budget_bytes"], 4096)


class SupervisorTests(RuntimeFixture):
    def make_supervisor(self, health):
        supervisor = runtime.Supervisor(self.config)
        process = mock.Mock(pid=20000, returncode=None)
        process.poll.return_value = None
        process.stdout = io.BytesIO(b"test server log\n")
        self.patches = [mock.patch.object(runtime, "STATE_PATH", self.root / "state.json"),
            mock.patch.object(runtime, "preflight", return_value={"requested": "CPU", "selected": "CPU",
                             "available": ["CPU"], "enumeration": "Available devices:\n  (none)"}),
            mock.patch.object(runtime, "ensure_port_available"),
            mock.patch.object(runtime.os, "access", return_value=True),
            mock.patch.object(runtime.subprocess, "Popen", return_value=process),
            mock.patch.object(runtime, "health_probe", side_effect=health),
            mock.patch.object(runtime.os, "killpg"),
            mock.patch.object(runtime.sys, "stdout", new=io.StringIO())]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        return supervisor, process

    def test_startup_timeout_stops_child_and_records_failure(self):
        supervisor, process = self.make_supervisor(lambda *args, **kwargs: False)
        self.assertEqual(supervisor.run(), 1)
        self.assertEqual(supervisor.state["status"], "failed")
        self.assertFalse(supervisor.state["ready"])
        self.assertIn("deadline exceeded during llama-server readiness", supervisor.state["error"])
        runtime.os.killpg.assert_called_with(process.pid, signal.SIGTERM)
        self.assertEqual(Path(supervisor.state["log_path"]).read_text().count("test server log"), 1)
        saved = json.loads(Path(supervisor.state["run_state_path"]).read_text())
        self.assertEqual(saved["status"], "failed")

    def test_expired_hash_budget_prevents_server_launch(self):
        supervisor, _ = self.make_supervisor(lambda *args, **kwargs: True)
        with mock.patch.object(runtime.time, "monotonic", side_effect=[0, 100]):
            self.assertEqual(supervisor.run(), 1)
        runtime.subprocess.Popen.assert_not_called()
        self.assertIn("deadline exceeded during model validation", supervisor.state["error"])

    def test_readiness_does_not_get_a_new_budget_after_hashing(self):
        clock = [0]
        supervisor, _ = self.make_supervisor(lambda *args, **kwargs: False)
        supervisor.config = dataclasses.replace(self.config, startup_timeout=10)
        validate = runtime.validate_models

        def validate_slowly(*args, **kwargs):
            result = validate(*args, **kwargs)
            clock[0] = 9.99
            return result

        def health(*args, **kwargs):
            self.assertLessEqual(kwargs["timeout"], 0.02)
            clock[0] = 10.01
            return False

        with mock.patch.object(runtime.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(runtime, "validate_models", side_effect=validate_slowly), \
                mock.patch.object(runtime, "health_probe", side_effect=health):
            self.assertEqual(supervisor.run(), 1)
        self.assertIn("deadline exceeded during llama-server readiness", supervisor.state["error"])

    def test_disk_full_fails_service_and_log_reader_keeps_draining(self):
        supervisor, process = self.make_supervisor(lambda *args, **kwargs: True)
        process.stdout = mock.Mock()
        process.stdout.read1.side_effect = [b"test server log\n", b"further child output\n", b""]
        log_file = mock.Mock()

        def write(data):
            if b"test server log" in data:
                raise OSError(errno.ENOSPC, "No space left on device")

        log_file.write.side_effect = write
        original_open = Path.open

        def open_log(path, *args, **kwargs):
            if path.name == "server.log":
                return log_file
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", new=open_log):
            self.assertEqual(supervisor.run(), 1)
        self.assertEqual(supervisor.state["status"], "failed")
        self.assertFalse(supervisor.state["ready"])
        self.assertIn("No space left on device", supervisor.state["logging_failure"])
        self.assertEqual(process.stdout.read1.call_count, 3)
        self.assertIn("further child output", runtime.sys.stdout.getvalue())
        self.assertEqual(log_file.write.call_count, 4)

    def test_process_exit_during_startup_is_failure(self):
        supervisor, process = self.make_supervisor(lambda *args, **kwargs: True)
        process.poll.return_value = 7
        process.returncode = 7
        self.assertEqual(supervisor.run(), 1)
        self.assertIn("exited during startup with status 7", supervisor.state["error"])

    def test_watchdog_exits_after_consecutive_health_failures(self):
        responses = iter([True, False, False])
        supervisor, process = self.make_supervisor(lambda *args, **kwargs: next(responses))
        self.assertEqual(supervisor.run(), 1)
        self.assertEqual(supervisor.state["health_failures"], 2)
        self.assertIn("Watchdog failed", supervisor.state["error"])
        process.wait.assert_called()

    def test_watchdog_exits_when_rocket_driver_disappears_despite_healthy_http(self):
        supervisor, _ = self.make_supervisor(lambda *args, **kwargs: True)
        supervisor.config = dataclasses.replace(self.config, mode="rocket")
        with mock.patch.object(runtime, "check_rocket_device", side_effect=runtime.DemoError("NPU removed")):
            self.assertEqual(supervisor.run(), 1)
        self.assertIn("NPU removed", supervisor.state["error"])
        self.assertFalse(supervisor.state["ready"])

    def test_signal_is_forwarded_and_normal_stop_is_success(self):
        supervisor = None

        def health(*args, **kwargs):
            supervisor.on_signal(signal.SIGTERM, None)
            return True

        supervisor, process = self.make_supervisor(health)
        self.assertEqual(supervisor.run(), 0)
        self.assertEqual(supervisor.state["status"], "stopped")
        self.assertFalse(supervisor.state["ready"])
        runtime.os.killpg.assert_any_call(process.pid, signal.SIGTERM)

    def test_shutdown_escalates_after_grace_period(self):
        supervisor, process = self.make_supervisor(lambda *args, **kwargs: False)
        supervisor.process = process
        process.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 20), 0]
        supervisor.shutdown()
        runtime.os.killpg.assert_any_call(process.pid, signal.SIGKILL)
        self.assertEqual(process.wait.call_count, 2)

    def test_surviving_pipe_reader_cannot_block_shutdown(self):
        supervisor, process = self.make_supervisor(lambda *args, **kwargs: False)
        supervisor.process = process
        supervisor.tee_thread = mock.Mock()
        supervisor.tee_thread.is_alive.return_value = True
        process.stdout = mock.Mock()
        supervisor.shutdown()
        runtime.os.killpg.assert_any_call(process.pid, signal.SIGKILL)
        process.stdout.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
