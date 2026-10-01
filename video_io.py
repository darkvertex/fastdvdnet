"""ffmpeg decode and encode helpers for the Replicate video API.

The codec argument is a closed set. Encoder names, containers, and pixel
formats come from this table, never from caller-supplied strings.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass


DEFAULT_CODEC = "libx265"
DEFAULT_CRF = 18
TEMP_PATCH = 5
CHUNK_CONTEXT = (TEMP_PATCH - 1) // 2


@dataclass(frozen=True)
class CodecSpec:
    name: str
    encoder: str
    container: str
    pix_fmt: str
    audio_encoder: str
    use_crf: bool
    video_args: tuple[str, ...] = ()
    vf: str | None = None
    probe_codec: str = ""


_EVEN_FRAME = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
_EVEN_WIDTH = "pad=ceil(iw/2)*2:ih"

CODECS: dict[str, CodecSpec] = {
    "libx265": CodecSpec(
        name="libx265",
        encoder="libx265",
        container="mp4",
        pix_fmt="yuv420p",
        audio_encoder="aac",
        use_crf=True,
        video_args=("-tag:v", "hvc1", "-x265-params", "log-level=error"),
        vf=_EVEN_FRAME,
        probe_codec="hevc",
    ),
    "libx264": CodecSpec(
        name="libx264",
        encoder="libx264",
        container="mp4",
        pix_fmt="yuv420p",
        audio_encoder="aac",
        use_crf=True,
        vf=_EVEN_FRAME,
        probe_codec="h264",
    ),
    "libvpx-vp9": CodecSpec(
        name="libvpx-vp9",
        encoder="libvpx-vp9",
        container="webm",
        pix_fmt="yuv420p",
        audio_encoder="libopus",
        use_crf=True,
        vf=_EVEN_FRAME,
        probe_codec="vp9",
    ),
    "prores_ks": CodecSpec(
        name="prores_ks",
        encoder="prores_ks",
        container="mov",
        pix_fmt="yuv422p10le",
        audio_encoder="pcm_s16le",
        use_crf=False,
        vf=_EVEN_WIDTH,
        probe_codec="prores",
    ),
    "ffv1": CodecSpec(
        name="ffv1",
        encoder="ffv1",
        container="mkv",
        pix_fmt="yuv444p",
        audio_encoder="pcm_s16le",
        use_crf=False,
        probe_codec="ffv1",
    ),
}


def get_codec(name: str) -> CodecSpec:
    try:
        return CODECS[name]
    except KeyError as exc:
        allowed = ", ".join(CODECS)
        raise ValueError(f"Unknown codec {name!r}. Choose one of: {allowed}") from exc


def output_name(codec: str, stem: str = "output") -> str:
    return f"{stem}.{get_codec(codec).container}"


def iter_frame_chunks(n_frames: int, chunk_size: int, context: int = CHUNK_CONTEXT):
    """Yield (load_start, load_end, keep_start, keep_end) half-open ranges.

    ``keep_*`` are the frames to write. ``load_*`` includes temporal context
    so chunk borders are not reflected unless they are the real video edges.
    """
    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if context < 0:
        raise ValueError("context must be >= 0")
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        load_start = max(0, start - context)
        load_end = min(n_frames, end + context)
        yield load_start, load_end, start, end
        start = end


def build_encode_args(
    *,
    frame_pattern: str,
    fps: str,
    output_path: str,
    codec: str,
    crf: int,
    audio_path: str | None = None,
) -> list[str]:
    spec = get_codec(codec)
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-framerate",
        fps,
        "-start_number",
        "0",
        "-i",
        frame_pattern,
    ]
    if audio_path:
        args.extend(["-i", audio_path])
    args.extend(["-map", "0:v:0"])
    if audio_path:
        args.extend(["-map", "1:a:0"])
    if spec.vf:
        args.extend(["-vf", spec.vf])
    args.extend(["-c:v", spec.encoder, "-pix_fmt", spec.pix_fmt])
    if spec.use_crf:
        args.extend(["-crf", str(int(crf))])
    args.extend(spec.video_args)
    if audio_path:
        args.extend(["-c:a", spec.audio_encoder])
        if spec.audio_encoder == "aac":
            args.extend(["-b:a", "192k"])
        args.append("-shortest")
    if spec.container in {"mp4", "mov"}:
        args.extend(["-movflags", "+faststart"])
    args.append(output_path)
    return args


def run_checked(args: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n{detail}"
        )
    return completed


def _ffprobe(path: str, select_streams: str | None, entries: str) -> dict:
    args = ["ffprobe", "-v", "error", "-of", "json"]
    if select_streams:
        args.extend(["-select_streams", select_streams])
    args.extend(["-show_entries", entries, path])
    completed = run_checked(args)
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe returned unexpected JSON for {path}")
    return payload


def _positive_rate(rate: str | None) -> str | None:
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        try:
            if float(denominator) == 0 or float(numerator) <= 0:
                return None
        except ValueError:
            return None
        return rate
    try:
        if float(rate) <= 0:
            return None
    except ValueError:
        return None
    return rate


def probe_frame_rate(video_path: str) -> str:
    payload = _ffprobe(
        video_path,
        "v:0",
        "stream=avg_frame_rate,r_frame_rate",
    )
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = _positive_rate(stream.get(key))
        if rate:
            return rate
    return "24/1"


def probe_codec_name(video_path: str) -> str:
    payload = _ffprobe(video_path, "v:0", "stream=codec_name")
    streams = payload.get("streams") or []
    if not streams or "codec_name" not in streams[0]:
        raise RuntimeError(f"ffprobe did not report a video codec for {video_path}")
    return str(streams[0]["codec_name"])


def probe_audio_codec_name(video_path: str) -> str | None:
    payload = _ffprobe(video_path, "a:0", "stream=codec_name")
    streams = payload.get("streams") or []
    if not streams:
        return None
    name = streams[0].get("codec_name")
    return str(name) if name else None


def has_audio(video_path: str) -> bool:
    return probe_audio_codec_name(video_path) is not None


def decode_frames(video_path: str, frames_dir: str) -> int:
    os.makedirs(frames_dir, exist_ok=True)
    pattern = os.path.join(frames_dir, "%06d.png")
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            video_path,
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-start_number",
            "0",
            pattern,
        ]
    )
    frames = [
        name
        for name in os.listdir(frames_dir)
        if name.endswith(".png")
    ]
    if not frames:
        raise RuntimeError(f"ffmpeg decoded zero frames from {video_path}")
    return len(frames)


def extract_audio(video_path: str, audio_path: str) -> bool:
    if not has_audio(video_path):
        return False
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            video_path,
            "-vn",
            "-c:a",
            "pcm_s16le",
            audio_path,
        ]
    )
    return os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0


def encode_frames(
    *,
    frame_pattern: str,
    fps: str,
    output_path: str,
    codec: str,
    crf: int = DEFAULT_CRF,
    audio_path: str | None = None,
) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    args = build_encode_args(
        frame_pattern=frame_pattern,
        fps=fps,
        output_path=output_path,
        codec=codec,
        crf=crf,
        audio_path=audio_path,
    )
    run_checked(args)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg did not write {output_path}")
