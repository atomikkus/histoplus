FROM python:3.11-slim

# Install system dependencies for OpenSlide and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    libopenslide0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the project
COPY . .

# Install the package
RUN uv pip install --system --no-cache -e .

ENTRYPOINT ["python", "pipeline_report.py"]

