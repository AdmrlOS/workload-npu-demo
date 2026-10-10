import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("vision_client", Path(__file__).resolve().parents[1] / "scripts/vision-client.py")
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
JPEG = b"\xff\xd8\xff\xe0camera-test\xff\xd9"
RESPONSE = {"choices": [{"message": {"content": "A test image."}}], "timings": {"prompt_ms": 25.0}}


class VisionClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.image = self.directory / "input.png"
        self.image.write_bytes(PNG)

    def fake_response(self):
        return io.BytesIO(json.dumps(RESPONSE).encode())

    def test_saved_image_request_and_record(self):
        output = self.directory / "nested" / "result.json"
        args = client.parse_args(["--image", str(self.image), "--prompt", "What is here?", "--output", str(output)])
        with mock.patch.object(client, "urlopen", return_value=self.fake_response()) as opened:
            result = client.infer(args, self.image)
        request = opened.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"][0], {"type": "text", "text": "What is here?"})
        self.assertEqual(payload["messages"][0]["content"][1]["image_url"]["url"],
                         "data:image/png;base64," + base64.b64encode(PNG).decode())
        self.assertEqual(payload["max_tokens"], 128)
        self.assertFalse(payload["stream"])
        self.assertEqual(opened.call_args.kwargs["timeout"], 300)
        self.assertEqual(result["source_image"]["sha256"], hashlib.sha256(PNG).hexdigest())
        self.assertEqual(result["response"]["timings"], RESPONSE["timings"])
        self.assertTrue(result["timestamp"].endswith("Z"))
        client.atomic_write(output, json.dumps(result).encode())
        self.assertEqual(json.loads(output.read_text()), result)
        self.assertEqual(list(output.parent.glob(".vision-*")), [])

    def test_rejects_invalid_or_oversize_image_before_request(self):
        self.image.write_bytes(b"not an image")
        args = client.parse_args(["--image", str(self.image)])
        with mock.patch.object(client, "urlopen") as opened:
            with self.assertRaises(ValueError):
                client.infer(args, self.image)
            opened.assert_not_called()
        self.image.write_bytes(PNG)
        with mock.patch.object(client, "MAX_IMAGE_BYTES", 8):
            with self.assertRaises(ValueError):
                client.read_image(self.image)

    def test_invalid_options(self):
        cases = [["--url", "file:///tmp/server"], ["--url", "http://host:99999"],
                 ["--url", "http://user:secret@host"], ["--size", "0x480"],
                 ["--interval", "nan"], ["--interval", "0"], ["--frames", "0"],
                 ["--timeout", "-1"], ["--output", str(self.image)], ["--output", ""]]
        for options in cases:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    client.parse_args(["--image", str(self.image)] + options)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            client.parse_args(["--camera", "/dev/../tmp/video"])

    def test_capture_arguments_and_sequential_requests(self):
        capture_dir = self.directory / "captures"
        args = client.parse_args(["--camera", "/dev/video0", "--capture-dir", str(capture_dir),
                                  "--format", "yuyv422", "--size", "800x600"])

        def acquire(command, **kwargs):
            Path(command[-1]).write_bytes(JPEG)
            self.assertEqual(command[command.index("-input_format") + 1], "yuyv422")
            self.assertEqual(command[command.index("-video_size") + 1], "800x600")
            self.assertEqual(command[command.index("-frames:v") + 1], "1")
            self.assertEqual(kwargs["timeout"], 15)
            self.assertNotIn("shell", kwargs)

        with mock.patch.object(client.subprocess, "run", side_effect=acquire):
            image, acquisition = client.capture_frame(args)
        self.assertEqual(image, capture_dir / "latest.jpg")
        self.assertEqual(image.read_bytes(), JPEG)
        self.assertTrue(acquisition["frame_completed_at"].endswith("Z"))
        self.assertGreaterEqual(acquisition["elapsed_seconds"], 0)
        events = []
        with mock.patch.object(client, "capture_frame", side_effect=lambda _: (events.append("capture") or image, acquisition)), \
                mock.patch.object(client, "infer", side_effect=lambda *_: (events.append("request") or RESPONSE)), \
                mock.patch.object(client.time, "sleep", side_effect=lambda _: events.append("wait")), \
                contextlib.redirect_stdout(io.StringIO()):
            code = client.main(["--camera", "/dev/video0", "--capture-dir", str(capture_dir), "--frames", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(events, ["capture", "request", "wait", "capture", "request"])

    def test_capture_timeout_fails_without_retaining_partial_frame(self):
        args = client.parse_args(["--camera", "/dev/video0", "--capture-dir", str(self.directory / "captures")])
        with mock.patch.object(client.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 15)):
            with self.assertRaisesRegex(ValueError, "acquisition exceeded"):
                client.capture_frame(args)
        self.assertEqual(list(args.capture_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
