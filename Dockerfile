ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG DATASET_URL="https://drive.usercontent.google.com/download?id=1k3c-kH7TG4l0fq0haTdvRugW9NoNWorX&export=download&authuser=0"

WORKDIR /app/env

# Copy full environment source
COPY . /app/env

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && rm -rf /var/lib/apt/lists/*

# gdown handles Google Drive large-file confirmation flows.
RUN pip install --no-cache-dir gdown

# Download and extract dataset into vatavaran/data during image build.
RUN mkdir -p /app/env/vatavaran/data \
  && gdown "${DATASET_URL}" -O /tmp/Bank_filtered.zip \
  && unzip -q /tmp/Bank_filtered.zip -d /app/env/vatavaran/data \
  && rm -f /tmp/Bank_filtered.zip

# Install package and dependencies
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH="/app/env:${PYTHONPATH:-}"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "vatavaran.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
