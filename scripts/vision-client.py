#!/usr/bin/env python3
"""Send a saved image, or explicitly sampled V4L2 frames, to llama-server."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def endpoint_url(value):
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or re.search(r"[\s\\]", value)):
        raise argparse.ArgumentTypeError("URL must be an http(s) server URL without credentials, query, or fragment")
    try:
        if parts.port is not None and parts.port < 1:
            raise ValueError("port must be positive")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    value = value.rstrip("/")
    return value if value.endswith("/v1/chat/completions") else value + "/v1/chat/completions"


def positive_number(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def image_size(value):
    if not re.fullmatch(r"[1-9]\d{0,4}x[1-9]\d{0,4}", value):
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    if any(int(part) > 16384 for part in value.split("x")):
        raise argparse.ArgumentTypeError("each dimension must be at most 16384")
    return value


def nonempty_path(value):
    if not value.strip() or "\x00" in value:
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value)


def read_image(path):
    if not path.is_file():
        raise ValueError(f"image is not a regular file: {path}")
    with path.open("rb") as source:
        data = source.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image must contain between 1 byte and 20 MiB")
    if data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33:
        mime = "image/png"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("image must be a JPEG, PNG, or WebP file (recognized from its contents)")
    return data, mime


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".vision-", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_frame(args):
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=args.capture_dir, prefix=".capture-") as temporary:
        frame = Path(temporary) / "frame.jpg"
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "v4l2", "-input_format", args.format, "-video_size", args.size,
                   "-i", args.camera, "-frames:v", "1", "-an", "-c:v", "mjpeg",
                   "-q:v", "2", "-fs", str(MAX_IMAGE_BYTES + 1),
                   "-f", "image2", "-update", "1", str(frame)]
        try:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=args.capture_timeout, check=True)
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"camera acquisition exceeded {args.capture_timeout:g} seconds") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", errors="replace")[-2000:]
            raise ValueError(f"camera acquisition failed: {detail.strip()}") from error
        completed = utc_now()
        elapsed = time.monotonic() - started
        data, _ = read_image(frame)
        latest = args.capture_dir / "latest.jpg"
        atomic_write(latest, data)
    return latest, {"frame_completed_at": completed, "elapsed_seconds": elapsed,
                    "device": args.camera, "format": args.format, "size": args.size}


def infer(args, image, acquisition=None):
    data, mime = read_image(image)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": args.prompt},
        {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}},
    ]}], "max_tokens": args.max_tokens, "temperature": 0, "stream": False}
    request = Request(args.url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    with urlopen(request, timeout=args.timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("server response exceeded 10 MiB")
    response = json.loads(raw)
    if not isinstance(response, dict) or "error" in response or not response.get("choices"):
        raise ValueError("server did not return a successful chat-completion response")
    return {"timestamp": utc_now(), "elapsed_seconds": time.monotonic() - started,
            "source_image": {"path": str(image), "sha256": hashlib.sha256(data).hexdigest(),
                             "bytes": len(data), "mime_type": mime},
            "acquisition": acquisition, "response": response}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=nonempty_path, help="saved JPEG, PNG, or WebP")
    source.add_argument("--camera", help="opt-in V4L2 device, for example /dev/video0")
    parser.add_argument("--url", type=endpoint_url, default="http://127.0.0.1:8080")
    parser.add_argument("--prompt", default="Describe the objects and their positions.")
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--timeout", type=positive_number, default=300)
    parser.add_argument("--output", type=nonempty_path, help="atomically replace this JSON file after each response")
    parser.add_argument("--format", default="mjpeg", help="camera input format already checked with v4l2-ctl")
    parser.add_argument("--size", type=image_size, default="640x480")
    parser.add_argument("--interval", type=positive_number, default=5, help="seconds to wait after each response")
    parser.add_argument("--frames", type=positive_int, default=1)
    parser.add_argument("--capture-timeout", type=positive_number, default=15)
    parser.add_argument("--capture-dir", type=nonempty_path, default=Path("/data/captures"))
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.format):
        parser.error("--format must be a V4L2 format name such as mjpeg or yuyv422")
    if args.camera and (not args.camera.startswith("/dev/")
                        or ".." in Path(args.camera).parts or re.search(r"[\x00-\x1f]", args.camera)):
        parser.error("--camera must be an absolute device path under /dev")
    if args.image and args.frames != 1:
        parser.error("--frames requires --camera")
    if args.capture_dir.exists() and not args.capture_dir.is_dir():
        parser.error("--capture-dir must be a directory")
    if args.output:
        image = args.image if args.image else args.capture_dir / "latest.jpg"
        if args.output.exists() and not args.output.is_file():
            parser.error("--output must be a regular file path")
        if args.output.resolve() == image.resolve():
            parser.error("--output must not overwrite the input image")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        for index in range(args.frames):
            image, acquisition = capture_frame(args) if args.camera else (args.image, None)
            result = infer(args, image, acquisition)
            encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            if args.output:
                atomic_write(args.output, encoded)
            print(encoded.decode("utf-8"), end="", flush=True)
            if index + 1 < args.frames:
                time.sleep(args.interval)
        return 0
    except HTTPError as error:
        detail = error.read(2000).decode("utf-8", errors="replace")
        print(f"vision-client: HTTP {error.code}: {detail.strip()}", file=sys.stderr)
    except (OSError, ValueError, URLError) as error:
        print(f"vision-client: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
