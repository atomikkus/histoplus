# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    openslide-tools \
    libopenslide0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the project
COPY . .

# Install the package
RUN uv pip install --system --no-cache -e .

# Set environment variable for Hugging Face token (pass at runtime)
ENV HUGGING_FACE_HUB_TOKEN=""

# Set entrypoint to histoplus CLI
ENTRYPOINT ["histoplus"]
