import os
import shutil
import uuid
from pathlib import Path as LocalPath

import torch
from cog import BasePredictor, Input, Path

from infer_video import WEIGHTS, denoise_video, load_model, weights_dir
from video_io import DEFAULT_CODEC, DEFAULT_CRF, get_codec, output_name


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load both FastDVDnet checkpoints once per worker."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        directory = weights_dir()
        self.device = device
        self.models = {
            name: load_model(os.path.join(directory, filename), device)
            for name, filename in WEIGHTS.items()
        }

    def predict(
        self,
        video: Path = Input(description="Input video to denoise"),
        noise_sigma: int = Input(
            description="Noise level in 8-bit units. Divided by 255 before the model. The network does not estimate noise.",
            default=25,
            ge=5,
            le=55,
        ),
        weights: str = Input(
            description="gaussian uses model.pth. clipped uses model_clipped_noise.pth for clipped AWGN.",
            default="gaussian",
            # Cog resolves choices from the source literal. Keep this in sync with WEIGHTS.
            choices=["gaussian", "clipped"],
        ),
        codec: str = Input(
            description="Output video codec. libx265 and libx264 write MP4, VP9 writes WebM, ProRes writes MOV, FFV1 writes MKV.",
            default=DEFAULT_CODEC,
            # Cog resolves choices from the source literal. Keep this in sync with CODECS.
            choices=["libx265", "libx264", "libvpx-vp9", "prores_ks", "ffv1"],
        ),
        crf: int = Input(
            description="Quality for libx265, libx264, and VP9. Lower is higher quality. Ignored for ProRes and FFV1.",
            default=DEFAULT_CRF,
            ge=0,
            le=51,
        ),
    ) -> Path:
        """Denoise a video with FastDVDnet and encode it with ffmpeg."""
        if weights not in self.models:
            raise ValueError(f"Unknown weights {weights!r}. Choose one of: {', '.join(WEIGHTS)}")
        get_codec(codec)

        suffix = LocalPath(str(video)).suffix or ".mp4"
        input_path = LocalPath("/tmp") / f"fastdvdnet-input-{uuid.uuid4().hex}{suffix}"
        shutil.copy(str(video), input_path)

        output_path = LocalPath("/tmp") / f"fastdvdnet-{uuid.uuid4().hex}-{output_name(codec)}"
        denoise_video(
            video_path=str(input_path),
            output_path=str(output_path),
            model=self.models[weights],
            device=self.device,
            noise_sigma=float(noise_sigma),
            codec=codec,
            crf=int(crf),
        )
        return Path(str(output_path))
