# Runpod Serverless image. Cog does not use this file; `cog build` reads cog.yaml.
# The base image already ships the CUDA 12.1 torch build Cog rewrites to.
# Do not pip-install replicate-requirements.txt: that pulls the PyPI CPU wheel.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTDVDNET_WEIGHTS_DIR=/src/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        "runpod~=1.7.6" \
        boto3 \
    && python -c "import torch; assert torch.__version__.startswith('2.1.2'), torch.__version__"

# Same checkpoints and size checks as cog.yaml. This layer stays cached
# independently of later code copies.
RUN mkdir -p /src/models \
    && curl -fsSL -o /src/models/model.pth https://github.com/m-tassano/fastdvdnet/raw/c8fdf6182a0340e89dd18f5df25b47337cbede6f/model.pth \
    && curl -fsSL -o /src/models/model_clipped_noise.pth https://github.com/m-tassano/fastdvdnet/raw/c8fdf6182a0340e89dd18f5df25b47337cbede6f/model_clipped_noise.pth \
    && test "$(stat -c%s /src/models/model.pth)" -gt 9000000 \
    && test "$(stat -c%s /src/models/model_clipped_noise.pth)" -gt 9000000

COPY . /src

CMD ["python", "-u", "handler.py"]
