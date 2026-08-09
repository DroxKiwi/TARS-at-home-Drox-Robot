# CUDA runtime — compatible Ampere+ ; Blackwell (50xx) : drivers host récents requis
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TARS_HOST=0.0.0.0 \
    TARS_PORT=9743

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY static ./static

EXPOSE 9743

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9743"]
