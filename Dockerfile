# CalmLine on Cloud Run (CPU). Models are baked into the image so cold starts don't hit the network.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/models HF_HUB_OFFLINE=0 PORT=8080
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY calmline calmline
COPY app app
COPY data/raw data/raw
COPY scripts/download_models.py scripts/download_models.py
RUN python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['unitary/toxic-bert','unitary/unbiased-toxic-roberta','BAAI/bge-small-en-v1.5','Systran/faster-whisper-base.en']]"
ENV HF_HUB_OFFLINE=1
CMD exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1
