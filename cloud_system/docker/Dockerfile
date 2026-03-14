# ── API Server ──
FROM python:3.10-slim

WORKDIR /app

# System deps for nibabel / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build context is cloud_system/ — copy into cloud_system/ subdir
# so Python imports like 'cloud_system.api.main' resolve correctly
COPY . ./cloud_system/

# Create directories
RUN mkdir -p cloud_system/saved_models cloud_system/logs cloud_system/uploads

EXPOSE 8000

CMD ["uvicorn", "cloud_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
