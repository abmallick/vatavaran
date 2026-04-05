ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app/env

# Copy full environment source
COPY . /app/env

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install package and dependencies
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH="/app/env:${PYTHONPATH:-}"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "vatavaran.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
