"""Runpod input and download tests. These do not import torch or runpod."""
from __future__ import annotations

import base64
import inspect
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from serverless_io import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_BASE64_BODY_BYTES,
    MAX_INPUT_BYTES,
    READ_TIMEOUT_SECONDS,
    WEIGHT_NAMES,
    BucketTarget,
    base64_body_length,
    choose_output_encoding,
    classify_video,
    clear_read_deadline,
    content_type_for_codec,
    download_timeout,
    download_url,
    materialize_video,
    parse_job_input,
    resolve_bucket,
)
from video_io import DEFAULT_CODEC, DEFAULT_CRF, run_checked

_BUCKET = {
    "endpointUrl": "https://s3.example.com",
    "accessId": "access",
    "accessSecret": "secret",
    "bucketName": "clips",
}
_ENV_BUCKET = {
    "BUCKET_ENDPOINT_URL": "https://env.example.com",
    "BUCKET_ACCESS_KEY_ID": "env-access",
    "BUCKET_SECRET_ACCESS_KEY": "env-secret",
    "BUCKET_NAME": "env-clips",
}


def _root(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", *parts)


def test_defaults_match_replicate_predictor():
    params = parse_job_input({"video": "https://example.com/clip.mp4"})
    assert params.noise_sigma == 25
    assert params.weights == "gaussian"
    assert params.codec == DEFAULT_CODEC == "libx265"
    assert params.crf == DEFAULT_CRF == 18
    assert content_type_for_codec(params.codec) == "video/mp4"


def test_weight_names_match_predictor_and_checkpoints():
    predict = open(_root("predict.py"), encoding="utf-8").read()
    infer = open(_root("infer_video.py"), encoding="utf-8").read()
    for name in WEIGHT_NAMES:
        assert f'"{name}"' in predict
        assert f'"{name}"' in infer
    assert '"model.pth"' in infer
    assert '"model_clipped_noise.pth"' in infer


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"video": "https://example.com/a.mp4", "noise_sigma": 4}, "noise_sigma"),
        ({"video": "https://example.com/a.mp4", "noise_sigma": 56}, "noise_sigma"),
        ({"video": "https://example.com/a.mp4", "noise_sigma": True}, "noise_sigma"),
        ({"video": "https://example.com/a.mp4", "noise_sigma": 25.0}, "noise_sigma"),
        ({"video": "https://example.com/a.mp4", "crf": -1}, "crf"),
        ({"video": "https://example.com/a.mp4", "crf": 52}, "crf"),
        ({"video": "https://example.com/a.mp4", "weights": "nope"}, "weights"),
        ({"video": "https://example.com/a.mp4", "codec": "libx264 -vf evil"}, "codec"),
        ({}, "video"),
        ("clip", "object"),
    ],
)
def test_invalid_input_is_rejected(payload, match):
    with pytest.raises(ValueError, match=match):
        parse_job_input(payload)


def test_explicit_options_are_kept():
    params = parse_job_input(
        {
            "video": " https://example.com/a.webm ",
            "noise_sigma": 55,
            "weights": "clipped",
            "codec": "libvpx-vp9",
            "crf": 0,
        }
    )
    assert params.video == "https://example.com/a.webm"
    assert params.noise_sigma == 55
    assert params.weights == "clipped"
    assert params.codec == "libvpx-vp9"
    assert params.crf == 0
    assert content_type_for_codec("ffv1") == "video/x-matroska"
    assert content_type_for_codec("prores_ks") == "video/quicktime"


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("https://example.com/a.mp4", "url"),
        ("http://example.com/a.mov", "url"),
        ("data:video/mp4;base64,AAAA", "data_uri"),
        ("/tmp/clip.mp4", "path"),
        ("clip.mkv", "path"),
        (base64.b64encode(b"a" * 16).decode("ascii"), "base64"),
    ],
)
def test_classify_video(value, kind):
    assert classify_video(value) == kind


def test_classify_rejects_empty_and_unknown():
    with pytest.raises(ValueError, match="video"):
        classify_video("   ")
    with pytest.raises(ValueError, match="https URL"):
        classify_video("not a video")


def test_materialize_copies_a_local_file(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"frame-bytes")
    dest_dir = tmp_path / "work"
    dest_dir.mkdir()
    copied = materialize_video(str(source), str(dest_dir))
    assert os.path.basename(copied) == "input.mov"
    assert open(copied, "rb").read() == b"frame-bytes"
    assert copied != str(source)


def test_materialize_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        materialize_video(str(tmp_path / "missing.mp4"), str(tmp_path))


def test_materialize_decodes_data_uri_and_base64(tmp_path):
    payload = b"denoised-bytes"
    encoded = base64.b64encode(payload).decode("ascii")
    nested = tmp_path / "nested"
    nested.mkdir()
    data_path = materialize_video(f"data:video/mp4;base64,{encoded}", str(tmp_path))
    raw_path = materialize_video(encoded, str(nested))
    assert open(data_path, "rb").read() == payload
    assert open(raw_path, "rb").read() == payload


def test_download_timeout_is_connect_only():
    assert download_timeout() == (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
    assert CONNECT_TIMEOUT_SECONDS == 60.0
    assert READ_TIMEOUT_SECONDS is None


def test_clear_read_deadline_removes_socket_timeout():
    class _Socket:
        def __init__(self):
            self.timeout = 60.0

        def settimeout(self, value):
            self.timeout = value

        def gettimeout(self):
            return self.timeout

    class _Connection:
        def __init__(self):
            self.sock = _Socket()

    connection = _Connection()
    clear_read_deadline(connection)
    assert connection.sock.gettimeout() is None


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_download_follows_http_and_clears_the_read_deadline(tmp_path, monkeypatch):
    body = b"video-body"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/redir"):
                self.send_response(302)
                self.send_header("Location", "/clip.mp4")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = _serve(Handler)
    host, port = server.server_address
    recorded = {}
    real_clear = clear_read_deadline

    def spy(connection):
        recorded["before"] = connection.sock.gettimeout() if connection.sock else "missing"
        real_clear(connection)
        recorded["after"] = connection.sock.gettimeout() if connection.sock else "missing"

    monkeypatch.setattr("serverless_io.clear_read_deadline", spy)
    opened = {}
    real_open = __import__("serverless_io", fromlist=["open_connection"]).open_connection

    def spy_open(hostname, port, scheme, timeout):
        opened["timeout"] = timeout
        opened["scheme"] = scheme
        return real_open(hostname, port, scheme, timeout)

    monkeypatch.setattr("serverless_io.open_connection", spy_open)
    dest = tmp_path / "clip.mp4"
    try:
        download_url(f"http://{host}:{port}/redir", str(dest))
    finally:
        server.shutdown()
    assert dest.read_bytes() == body
    assert opened["timeout"] == CONNECT_TIMEOUT_SECONDS
    assert opened["scheme"] == "http"
    assert recorded["before"] == CONNECT_TIMEOUT_SECONDS
    assert recorded["after"] is None


def test_download_rejects_non_http_redirect(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    server = _serve(Handler)
    host, port = server.server_address
    try:
        with pytest.raises(ValueError, match="http or https"):
            download_url(f"http://{host}:{port}/clip.mp4", str(tmp_path / "out.mp4"))
    finally:
        server.shutdown()


def test_download_rejects_oversize_content_length(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "50")
            self.end_headers()
            self.wfile.write(b"x" * 50)

        def log_message(self, fmt, *args):
            return

    server = _serve(Handler)
    host, port = server.server_address
    dest = tmp_path / "big.mp4"
    try:
        with pytest.raises(ValueError, match="byte limit"):
            download_url(f"http://{host}:{port}/big.mp4", str(dest), max_bytes=8)
    finally:
        server.shutdown()
    assert not dest.exists()


def test_bucket_from_environment_and_job_override():
    assert resolve_bucket({}, {}) is None
    env_bucket = resolve_bucket({}, _ENV_BUCKET)
    assert env_bucket is not None
    assert env_bucket.bucket_name == "env-clips"
    job_bucket = resolve_bucket({"s3Config": _BUCKET}, _ENV_BUCKET)
    assert job_bucket is not None
    assert job_bucket.endpoint_url == "https://s3.example.com"
    assert job_bucket.upload_creds()["accessId"] == "access"
    with pytest.raises(ValueError, match="BUCKET_NAME"):
        resolve_bucket({}, {"BUCKET_ENDPOINT_URL": "https://s3.example.com"})
    with pytest.raises(ValueError, match="bucketName"):
        resolve_bucket({"s3Config": {"endpointUrl": "https://s3.example.com"}}, {})


def test_output_encoding_prefers_a_url_and_limits_base64():
    bucket = BucketTarget("https://s3.example.com", "a", "b", "clips")
    assert choose_output_encoding(1, bucket) == "url"
    raw_limit = (MAX_BASE64_BODY_BYTES // 4) * 3
    assert base64_body_length(raw_limit) <= MAX_BASE64_BODY_BYTES
    assert choose_output_encoding(raw_limit, None) == "base64"
    with pytest.raises(ValueError, match="base64"):
        choose_output_encoding(raw_limit + 3, None)
    assert MAX_INPUT_BYTES >= 32 * 1024 * 1024 * 1024


def test_ffmpeg_and_handler_have_no_short_deadline():
    assert "timeout" not in inspect.getsource(run_checked)
    handler = open(_root("handler.py"), encoding="utf-8").read()
    infer = open(_root("infer_video.py"), encoding="utf-8").read()
    dockerfile = open(_root("Dockerfile"), encoding="utf-8").read()
    assert "refresh_worker" not in handler
    assert "on_chunk" in handler
    assert "on_chunk: Callable[[int, int], None] | None = None" in infer
    assert "-r replicate-requirements.txt" not in dockerfile
    assert "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime" in dockerfile
    assert "604800000" in open(_root("README.md"), encoding="utf-8").read()
