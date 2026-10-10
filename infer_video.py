"""Decode a video, denoise it in temporal chunks, and encode the result."""
from __future__ import annotations

import os
import shutil
import tempfile

import cv2
import numpy as np
import torch

from fastdvdnet import denoise_seq_fastdvdnet
from models import FastDVDnet
from video_io import (
    CHUNK_CONTEXT,
    TEMP_PATCH,
    decode_frames,
    encode_frames,
    extract_audio,
    iter_frame_chunks,
    probe_frame_rate,
)

CHUNK_SIZE = 16

WEIGHTS = {
    "gaussian": "model.pth",
    "clipped": "model_clipped_noise.pth",
}


def weights_dir() -> str:
    env_dir = os.getenv("FASTDVDNET_WEIGHTS_DIR")
    if env_dir:
        return env_dir
    baked = "/src/models"
    if os.path.isfile(os.path.join(baked, WEIGHTS["gaussian"])):
        return baked
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _load_state_dict(path: str, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    except Exception:
        # Official checkpoints are pickled state dicts. Fall back if the
        # weights-only loader rejects an older pickle opcode.
        state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} is not a state dict")
    keys = list(state.keys())
    if keys and all(str(key).startswith("module.") for key in keys):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Missing weights at {path}. The Cog image bakes them in during build."
        )
    model = FastDVDnet(num_input_frames=5)
    model.load_state_dict(_load_state_dict(path, device))
    model.to(device)
    model.eval()
    return model


def _frame_path(directory: str, index: int) -> str:
    return os.path.join(directory, f"{index:06d}.png")


def load_frames(paths: list[str], device: torch.device) -> torch.Tensor:
    frames: list[np.ndarray] = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read frame {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    array = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def save_frame(frame: torch.Tensor, path: str) -> None:
    rgb = frame.detach().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr):
        raise RuntimeError(f"Could not write frame {path}")


def denoise_video(
    *,
    video_path: str,
    output_path: str,
    model: torch.nn.Module,
    device: torch.device,
    noise_sigma: float,
    codec: str,
    crf: int,
    chunk_size: int = CHUNK_SIZE,
) -> str:
    """Denoise ``video_path`` and write ``output_path``.

    ``noise_sigma`` is the 8-bit noise level (the CLI's ``--noise_sigma``).
    It is divided by 255 before it is passed to the network. The input frames
    are denoised as they are; no synthetic noise is added.
    """
    if noise_sigma < 5 or noise_sigma > 55:
        raise ValueError("noise_sigma must be between 5 and 55")

    with tempfile.TemporaryDirectory(prefix="fastdvdnet-") as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        source_frames = decode_frames(video_path, in_dir)
        # The temporal window indexes up to two frames on either side. Repeat the
        # last frame so clips shorter than 3 frames still have a valid window.
        n_frames = source_frames
        if source_frames < 3:
            last = _frame_path(in_dir, source_frames - 1)
            for index in range(source_frames, 3):
                shutil.copy(last, _frame_path(in_dir, index))
            n_frames = 3
        fps = probe_frame_rate(video_path)
        audio_path = os.path.join(tmp, "audio.wav")
        mux_audio = extract_audio(video_path, audio_path)
        noise_std = torch.tensor([noise_sigma / 255.0], dtype=torch.float32, device=device)

        for load_start, load_end, keep_start, keep_end in iter_frame_chunks(
            n_frames, chunk_size, CHUNK_CONTEXT
        ):
            paths = [_frame_path(in_dir, index) for index in range(load_start, load_end)]
            sequence = load_frames(paths, device)
            with torch.no_grad():
                denoised = denoise_seq_fastdvdnet(
                    sequence,
                    noise_std,
                    TEMP_PATCH,
                    model,
                )
            rel_start = keep_start - load_start
            for offset, frame_index in enumerate(range(keep_start, keep_end)):
                if frame_index >= source_frames:
                    continue
                save_frame(denoised[rel_start + offset], _frame_path(out_dir, frame_index))

        encode_frames(
            frame_pattern=os.path.join(out_dir, "%06d.png"),
            fps=fps,
            output_path=output_path,
            codec=codec,
            crf=crf,
            audio_path=audio_path if mux_audio else None,
        )
    return output_path
