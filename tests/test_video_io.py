import os
import subprocess
import tempfile

import pytest

from video_io import (
    CODECS,
    DEFAULT_CODEC,
    DEFAULT_CRF,
    build_encode_args,
    decode_frame_args,
    decode_frames,
    encode_frames,
    extract_audio,
    get_codec,
    iter_frame_chunks,
    probe_audio_codec_name,
    probe_codec_name,
    probe_frame_rate,
)


def test_predict_choice_literals_match_codec_table():
    """Cog only accepts a list literal for Input choices, so predict.py cannot
    reference CODECS directly. This keeps the two lists from drifting.
    """
    source = open(os.path.join(os.path.dirname(__file__), "..", "predict.py"), encoding="utf-8").read()
    for name in CODECS:
        assert f'"{name}"' in source
    # Cog ignores non-literal defaults and then requires the input at predict time.
    assert 'default="libx265"' in source
    assert "default=18" in source
    assert DEFAULT_CODEC == "libx265"
    assert DEFAULT_CRF == 18


def test_default_codec_is_libx265():
    assert DEFAULT_CODEC == "libx265"
    spec = get_codec(DEFAULT_CODEC)
    assert spec.encoder == "libx265"
    assert spec.container == "mp4"
    assert spec.probe_codec == "hevc"
    assert spec.use_crf is True


def test_unknown_codec_is_rejected():
    with pytest.raises(ValueError, match="Unknown codec"):
        get_codec("libx264 -vf evil")


@pytest.mark.parametrize("name", list(CODECS))
def test_codec_map(name):
    spec = CODECS[name]
    args = build_encode_args(
        frame_pattern="frames/%06d.png",
        fps="24/1",
        output_path=f"out.{spec.container}",
        codec=name,
        crf=18,
        audio_path=None,
    )
    assert args[0] == "ffmpeg"
    assert spec.encoder in args
    assert spec.pix_fmt in args
    joined = " ".join(args)
    if spec.use_crf:
        assert "-crf" in args
        assert args[args.index("-crf") + 1] == "18"
    else:
        assert "-crf" not in args
    assert "evil" not in joined
    assert args[-1] == f"out.{spec.container}"


def test_crf_is_omitted_for_prores_and_ffv1():
    for name in ("prores_ks", "ffv1"):
        args = build_encode_args(
            frame_pattern="frames/%06d.png",
            fps="25/1",
            output_path="out.bin",
            codec=name,
            crf=18,
        )
        assert "-crf" not in args


def test_lossy_codecs_pad_to_even_dimensions():
    for name in ("libx265", "libx264", "libvpx-vp9"):
        args = build_encode_args(
            frame_pattern="frames/%06d.png",
            fps="24/1",
            output_path="out.bin",
            codec=name,
            crf=18,
        )
        assert "pad=ceil(iw/2)*2:ceil(ih/2)*2" in args


def test_audio_encoder_follows_codec():
    args = build_encode_args(
        frame_pattern="frames/%06d.png",
        fps="24/1",
        output_path="out.mp4",
        codec="libx265",
        crf=20,
        audio_path="audio.wav",
    )
    assert args[args.index("-c:a") + 1] == "aac"
    assert "-shortest" in args

    prores = build_encode_args(
        frame_pattern="frames/%06d.png",
        fps="24/1",
        output_path="out.mov",
        codec="prores_ks",
        crf=20,
        audio_path="audio.wav",
    )
    assert prores[prores.index("-c:a") + 1] == "pcm_s16le"
    assert "-crf" not in prores


def test_frame_chunks_keep_context_except_at_edges():
    chunks = list(iter_frame_chunks(10, chunk_size=4, context=2))
    assert chunks == [
        (0, 6, 0, 4),
        (2, 10, 4, 8),
        (6, 10, 8, 10),
    ]


def test_frame_chunks_cover_every_frame_once():
    kept = []
    for _load_start, _load_end, keep_start, keep_end in iter_frame_chunks(7, 3, 2):
        kept.extend(range(keep_start, keep_end))
    assert kept == list(range(7))


def _ffmpeg(*args: str) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])


@pytest.mark.parametrize("name", list(CODECS))
def test_ffmpeg_round_trip(name):
    spec = CODECS[name]
    with tempfile.TemporaryDirectory() as tmp:
        frames = os.path.join(tmp, "frames")
        os.makedirs(frames)
        _ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=0.4",
            "-pix_fmt",
            "rgb24",
            "-start_number",
            "0",
            os.path.join(frames, "%06d.png"),
        )
        output = os.path.join(tmp, f"out.{spec.container}")
        encode_frames(
            frame_pattern=os.path.join(frames, "%06d.png"),
            fps="10/1",
            output_path=output,
            codec=name,
            crf=28,
        )
        assert os.path.getsize(output) > 0
        assert probe_codec_name(output) == spec.probe_codec


def test_decode_uses_vsync_passthrough():
    args = decode_frame_args("in.mp4", "frames")
    assert args[args.index("-vsync") + 1] == "0"
    assert "-fps_mode" not in args


def test_decode_preserves_frame_count_and_muxes_audio():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "source.mp4")
        _ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            source,
        )
        assert probe_frame_rate(source) == "10/1"
        frames = os.path.join(tmp, "in")
        assert decode_frames(source, frames) == 4
        audio = os.path.join(tmp, "audio.wav")
        assert extract_audio(source, audio) is True
        output = os.path.join(tmp, "out.mp4")
        encode_frames(
            frame_pattern=os.path.join(frames, "%06d.png"),
            fps=probe_frame_rate(source),
            output_path=output,
            codec="libx265",
            crf=28,
            audio_path=audio,
        )
        assert probe_codec_name(output) == "hevc"
        assert probe_audio_codec_name(output) == "aac"
