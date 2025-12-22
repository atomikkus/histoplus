# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
<<<<<<< HEAD
=======
# Increase shared memory for PyTorch DataLoader workers
ENV TORCH_SHARED_MEMORY_SIZE=2147483648
>>>>>>> 1afd1fb (FEAT: Update Dockerfile for GPU support and environment configuration)

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

# Install PyTorch with CUDA 12.1 support first
RUN uv pip install --system --no-cache \
    torch>=2.4.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install dependencies (torch will be skipped if already installed)
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
