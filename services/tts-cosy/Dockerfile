# CosyVoice 3 — AutoModel officiel + Torch cu128 (RTX 50xx / sm_120)
# Base 12.4 OK : les wheels PyTorch cu128 embarquent leurs libs CUDA.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COSYVOICE_ROOT=/opt/CosyVoice \
    COSYVOICE_MODEL_DIR=/root/.cache/cosyvoice/Fun-CosyVoice3-0.5B \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    build-essential \
    git \
    git-lfs \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && git lfs install

RUN git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git ${COSYVOICE_ROOT} \
    && cd ${COSYVOICE_ROOT} \
    && git submodule update --init --recursive

WORKDIR /app

# Blackwell (sm_120) : wheels cu128 — une seule fois, ne pas laisser pip les écraser
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchaudio==2.8.0

COPY requirements.txt .
# --no-deps sur whisper : sinon pip backtrack et re-télécharge d'anciens torch (~Go)
RUN python -m pip install --upgrade "setuptools==69.5.1" wheel \
    && pip install -r requirements.txt \
    && pip install --no-build-isolation --no-deps "openai-whisper==20231117" \
    && pip install "numba>=0.56" "more-itertools" "tqdm" \
    && (pip install --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
        "onnxruntime-gpu==1.20.2" || pip install "onnxruntime==1.20.1") \
    && python -c "import torch; assert 'cu128' in torch.__version__, torch.__version__; import whisper; print('OK', torch.__version__, whisper.__file__)"

COPY src ./src

EXPOSE 9750

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9750"]
