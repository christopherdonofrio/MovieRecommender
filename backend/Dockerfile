FROM python:3.12-slim

WORKDIR /app

# System dependencies for pandas C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# CPU-only PyTorch build. this is an inference-only service (no training, no GPU), and the CPU wheel is a fraction of the size of the default
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/ backend/
COPY src/ src/

# data/ and models/ are populated at container startup by fetch_artifacts.py,
# not baked into the image -- see that file for why.

EXPOSE 8000

WORKDIR /app/backend
# Cloud Run injects a PORT env var and requires the container to listen on it;
# ${PORT:-8000} keeps 8000 as the default for local/Render use where PORT isn't set.
CMD ["sh", "-c", "python fetch_artifacts.py && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
