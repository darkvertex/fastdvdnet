"""Runpod Serverless worker for FastDVDnet video denoising.

Checkpoints load once before the worker accepts jobs. The platform's
execution timeout is what stops a long job; this process does not add one.
Set the endpoint execution timeout and the request policy above Replicate's
ten-minute cap. The maximum is 7 days (604800000 ms).
"""
from __future__ import annotations

import base64
import os
import shutil
import tempfile

import runpod
import torch

from infer_video import WEIGHTS, denoise_video, load_model, weights_dir
from serverless_io import (
    BucketTarget,
    choose_output_encoding,
    content_type_for_codec,
    materialize_video,
    parse_job_input,
    resolve_bucket,
)
from video_io import output_name

_WORKER: tuple[torch.device, dict[str, torch.nn.Module]] | None = None


def load_worker_models() -> tuple[torch.device, dict[str, torch.nn.Module]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory = weights_dir()
    models = {
        name: load_model(os.path.join(directory, filename), device)
        for name, filename in WEIGHTS.items()
    }
    return device, models


def get_worker() -> tuple[torch.device, dict[str, torch.nn.Module]]:
    global _WORKER
    if _WORKER is None:
        _WORKER = load_worker_models()
    return _WORKER


def handler(job: dict) -> dict[str, str]:
    """Denoise one video and return a URL or a base64 body."""
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    params = parse_job_input(job.get("input"))
    device, models = get_worker()
    if params.weights not in models:
        allowed = ", ".join(WEIGHTS)
        raise ValueError(f"Unknown weights {params.weights!r}. Choose one of: {allowed}")
    bucket = resolve_bucket(job)
    work = tempfile.mkdtemp(prefix="fastdvdnet-job-")
    try:
        input_path = materialize_video(params.video, work)
        filename = output_name(params.codec, stem="fastdvdnet")
        output_path = os.path.join(work, filename)
        denoise_video(
            video_path=input_path,
            output_path=output_path,
            model=models[params.weights],
            device=device,
            noise_sigma=float(params.noise_sigma),
            codec=params.codec,
            crf=params.crf,
            on_chunk=lambda done, total: _report_progress(job, done, total),
        )
        size = os.path.getsize(output_path)
        encoding = choose_output_encoding(size, bucket)
        content_type = content_type_for_codec(params.codec)
        if encoding == "url":
            if bucket is None:
                raise RuntimeError("url output requires bucket credentials")
            video = _upload(job, output_path, filename, bucket, content_type)
        else:
            video = _read_base64(output_path)
        return {
            "video": video,
            "encoding": encoding,
            "content_type": content_type,
            "filename": filename,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _upload(
    job: dict,
    output_path: str,
    filename: str,
    bucket: BucketTarget,
    content_type: str,
) -> str:
    from runpod.serverless.utils import rp_upload

    job_id = job.get("id")
    prefix = job_id if isinstance(job_id, str) and job_id else "local"
    return rp_upload.upload_file_to_bucket(
        file_name=filename,
        file_location=output_path,
        bucket_creds=bucket.upload_creds(),
        bucket_name=bucket.bucket_name,
        prefix=prefix,
        extra_args={"ContentType": content_type},
    )


def _read_base64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _report_progress(job: dict, done: int, total: int) -> None:
    message = f"Denoised chunk {done} of {total}"
    try:
        runpod.serverless.progress_update(job, message)
    except Exception as exc:
        print(f"progress update skipped: {exc}", flush=True)


if __name__ == "__main__":
    get_worker()
    runpod.serverless.start({"handler": handler})
