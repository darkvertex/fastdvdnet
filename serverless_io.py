"""Runpod job input, video fetch, and output-size decisions.

This module does not import torch, cog, or runpod. The Cog predictor does
not use it. Downloads have a connect timeout and no read deadline so a long
source file can transfer for as long as the platform allows the job to run.
"""
from __future__ import annotations

import base64
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from urllib.parse import urljoin, urlparse

from video_io import DEFAULT_CODEC, DEFAULT_CRF, get_codec

# Keep these aligned with predict.py Input choices. Cog cannot import them.
WEIGHT_NAMES = ("gaussian", "clipped")

CONNECT_TIMEOUT_SECONDS = 60.0
# None means the socket has no read deadline. A slow transfer is limited by
# Runpod's execution timeout, not by this process.
READ_TIMEOUT_SECONDS = None
MAX_REDIRECTS = 5
MAX_INPUT_BYTES = 32 * 1024 * 1024 * 1024
# Encoded base64 length, not the raw file. Stays under the 10 MB /run limit.
MAX_BASE64_BODY_BYTES = 7 * 1024 * 1024

_CONTENT_TYPES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
}
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mxf")
_READ_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class DenoiseParams:
    video: str
    noise_sigma: int
    weights: str
    codec: str
    crf: int


@dataclass(frozen=True)
class BucketTarget:
    endpoint_url: str
    access_id: str
    access_secret: str
    bucket_name: str

    def upload_creds(self) -> dict[str, str]:
        return {
            "endpointUrl": self.endpoint_url,
            "accessId": self.access_id,
            "accessSecret": self.access_secret,
        }


def download_timeout() -> tuple[float, None]:
    """Connect timeout, then no read deadline."""
    return (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)


def parse_job_input(job_input: object) -> DenoiseParams:
    if not isinstance(job_input, dict):
        raise ValueError("job input must be an object")
    video = job_input.get("video")
    if not isinstance(video, str) or not video.strip():
        raise ValueError("video is required")
    noise_sigma = _bounded_int(
        _field(job_input, "noise_sigma", 25),
        "noise_sigma",
        5,
        55,
    )
    weights = _field(job_input, "weights", "gaussian")
    if not isinstance(weights, str) or weights not in WEIGHT_NAMES:
        allowed = ", ".join(WEIGHT_NAMES)
        raise ValueError(f"Unknown weights {weights!r}. Choose one of: {allowed}")
    codec = _field(job_input, "codec", DEFAULT_CODEC)
    if not isinstance(codec, str):
        raise ValueError("codec must be a string")
    get_codec(codec)
    crf = _bounded_int(_field(job_input, "crf", DEFAULT_CRF), "crf", 0, 51)
    return DenoiseParams(
        video=video.strip(),
        noise_sigma=noise_sigma,
        weights=weights,
        codec=codec,
        crf=crf,
    )


def classify_video(value: str) -> str:
    """Return ``url``, ``data_uri``, ``path``, or ``base64``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("video is required")
    text = value.strip()
    lowered = text.lower()
    if lowered.startswith("https://") or lowered.startswith("http://"):
        return "url"
    if lowered.startswith("data:"):
        return "data_uri"
    if _is_path_like(text):
        return "path"
    compact = "".join(text.split())
    if len(compact) >= 16 and len(compact) % 4 == 0 and _BASE64_RE.fullmatch(compact):
        return "base64"
    raise ValueError(
        "video must be an https URL, a base64 string, a data URI, or a local file path"
    )


def materialize_video(source: str, directory: str, *, max_bytes: int = MAX_INPUT_BYTES) -> str:
    """Copy, decode, or download ``source`` into ``directory`` and return the path."""
    kind = classify_video(source)
    if kind == "url":
        suffix = _suffix_from_url(source)
        dest = os.path.join(directory, f"input{suffix}")
        download_url(source.strip(), dest, max_bytes=max_bytes)
        return dest
    if kind == "data_uri":
        payload = _decode_data_uri(source.strip(), max_bytes=max_bytes)
        dest = os.path.join(directory, "input.mp4")
        _write_bytes(dest, payload)
        return dest
    if kind == "base64":
        payload = _decode_base64("".join(source.split()), max_bytes=max_bytes)
        dest = os.path.join(directory, "input.mp4")
        _write_bytes(dest, payload)
        return dest
    path = os.path.expanduser(source.strip())
    if not os.path.isfile(path):
        raise ValueError(f"video file not found: {path}")
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(
            f"video is {size} bytes, which is above the {max_bytes} byte limit"
        )
    suffix = os.path.splitext(path)[1].lower()
    if not suffix or len(suffix) > 8:
        suffix = ".mp4"
    dest = os.path.join(directory, f"input{suffix}")
    shutil.copy(path, dest)
    return dest


def resolve_bucket(
    job: Mapping[str, object],
    env: Mapping[str, str] | None = None,
) -> BucketTarget | None:
    """Return bucket credentials from ``s3Config`` or the ``BUCKET_*`` environment.

    A per-job ``s3Config`` wins. An incomplete configuration raises instead of
    uploading to an unnamed bucket.
    """
    if env is None:
        env = os.environ
    if "s3Config" in job and job.get("s3Config") is not None:
        return _bucket_from_mapping(job.get("s3Config"), source="s3Config")
    values = {
        "endpointUrl": env.get("BUCKET_ENDPOINT_URL", ""),
        "accessId": env.get("BUCKET_ACCESS_KEY_ID", ""),
        "accessSecret": env.get("BUCKET_SECRET_ACCESS_KEY", ""),
        "bucketName": env.get("BUCKET_NAME", ""),
    }
    present = [bool(item) for item in values.values()]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "Bucket configuration is incomplete. Set BUCKET_ENDPOINT_URL, "
            "BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY, and BUCKET_NAME."
        )
    return _bucket_from_mapping(values, source="environment")


def base64_body_length(file_size: int) -> int:
    if file_size < 0:
        raise ValueError("file_size must be >= 0")
    if file_size == 0:
        return 0
    return 4 * ((file_size + 2) // 3)


def choose_output_encoding(file_size: int, bucket: BucketTarget | None) -> str:
    """Return ``url`` when a bucket is configured, otherwise ``base64`` if it fits."""
    if file_size < 0:
        raise ValueError("file_size must be >= 0")
    if bucket is not None:
        return "url"
    if base64_body_length(file_size) <= MAX_BASE64_BODY_BYTES:
        return "base64"
    raise ValueError(
        "Denoised video is too large to return as base64. "
        "Set BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY, "
        "and BUCKET_NAME, or pass s3Config on the job."
    )


def content_type_for_codec(codec: str) -> str:
    container = get_codec(codec).container
    try:
        return _CONTENT_TYPES[container]
    except KeyError as exc:
        raise ValueError(f"No content type for codec {codec!r}") from exc


def download_url(url: str, dest: str, *, max_bytes: int = MAX_INPUT_BYTES) -> None:
    """Download ``url`` to ``dest``.

    The connection attempt times out after ``CONNECT_TIMEOUT_SECONDS``. After
    the socket connects, the read deadline is cleared.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"video URL must be http or https, got {parsed.scheme or current!r}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        connection = open_connection(parsed.hostname, port, parsed.scheme, CONNECT_TIMEOUT_SECONDS)
        try:
            connection.request("GET", path, headers={"User-Agent": "fastdvdnet-runpod"})
            clear_read_deadline(connection)
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                _drain_redirect(response)
                if not location:
                    raise ValueError("video URL redirect is missing a Location header")
                current = urljoin(current, location)
                continue
            if response.status != 200:
                raise ValueError(f"video download failed with HTTP {response.status}")
            _stream_response(response, dest, max_bytes)
            return
        finally:
            connection.close()
    raise ValueError(f"video URL exceeded {MAX_REDIRECTS} redirects")


def open_connection(
    hostname: str,
    port: int,
    scheme: str,
    timeout: float,
) -> HTTPConnection:
    if scheme == "https":
        return HTTPSConnection(hostname, port, timeout=timeout)
    return HTTPConnection(hostname, port, timeout=timeout)


def clear_read_deadline(connection: HTTPConnection) -> None:
    sock = connection.sock
    if sock is not None:
        sock.settimeout(READ_TIMEOUT_SECONDS)


def _drain_redirect(response: HTTPResponse) -> None:
    """Read a short redirect body so a huge redirect cannot stall the job."""
    remaining = 64 * 1024
    while remaining > 0:
        chunk = response.read(min(_READ_CHUNK, remaining))
        if not chunk:
            return
        remaining -= len(chunk)


def _stream_response(response: HTTPResponse, dest: str, max_bytes: int) -> None:
    declared = response.getheader("Content-Length")
    if declared is not None:
        try:
            announced = int(declared)
        except ValueError as exc:
            raise ValueError("video download returned a bad Content-Length") from exc
        if announced > max_bytes:
            raise ValueError(
                f"video is {announced} bytes, which is above the {max_bytes} byte limit"
            )
    written = 0
    try:
        with open(dest, "wb") as handle:
            while True:
                chunk = response.read(_READ_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"video is above the {max_bytes} byte limit"
                    )
                handle.write(chunk)
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        raise
    if written == 0 or (declared is not None and written != announced):
        if os.path.exists(dest):
            os.remove(dest)
        if written == 0:
            raise ValueError("video download was empty")
        raise ValueError(
            f"video download ended after {written} bytes, expected {announced}"
        )


def _decode_data_uri(value: str, *, max_bytes: int) -> bytes:
    header, separator, payload = value.partition(",")
    if not separator or ";base64" not in header.lower():
        raise ValueError("video data URI must be base64 encoded")
    return _decode_base64("".join(payload.split()), max_bytes=max_bytes)


def _decode_base64(payload: str, *, max_bytes: int) -> bytes:
    if len(payload) > base64_body_length(max_bytes):
        raise ValueError(
            f"video is above the {max_bytes} byte limit"
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("video base64 could not be decoded") from exc
    if len(data) > max_bytes:
        raise ValueError(
            f"video is {len(data)} bytes, which is above the {max_bytes} byte limit"
        )
    if not data:
        raise ValueError("video base64 was empty")
    return data


def _write_bytes(path: str, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)


def _suffix_from_url(url: str) -> str:
    suffix = os.path.splitext(urlparse(url).path)[1].lower()
    if suffix in _VIDEO_SUFFIXES:
        return suffix
    return ".mp4"


def _is_path_like(text: str) -> bool:
    if text.startswith(("/", "./", "../", "~")):
        return True
    if "/" in text or "\\" in text:
        return True
    if os.path.exists(text):
        return True
    return text.lower().endswith(_VIDEO_SUFFIXES)


def _bucket_from_mapping(value: object, *, source: str) -> BucketTarget:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be an object")
    endpoint = value.get("endpointUrl")
    access = value.get("accessId")
    secret = value.get("accessSecret")
    name = value.get("bucketName")
    missing = [
        key
        for key, item in (
            ("endpointUrl", endpoint),
            ("accessId", access),
            ("accessSecret", secret),
            ("bucketName", name),
        )
        if not isinstance(item, str) or not item
    ]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"{source} is missing {fields}")
    return BucketTarget(
        endpoint_url=str(endpoint),
        access_id=str(access),
        access_secret=str(secret),
        bucket_name=str(name),
    )


def _field(payload: Mapping[str, object], name: str, default: object) -> object:
    if name not in payload:
        return default
    return payload[name]


def _bounded_int(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    if value < lower or value > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value
