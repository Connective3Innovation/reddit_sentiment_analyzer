FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (for building wheels like pyarrow if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install your package (src layout)
COPY pyproject.toml setup.cfg* ./
COPY src ./src
RUN pip install -e .

# Add the API entrypoint
COPY api.py ./api.py

# Cloud Run listens on $PORT; default to 8080 locally
ENV PORT=8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
