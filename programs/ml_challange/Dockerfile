# Stage 1: Build stage
FROM python:3.12-slim AS build

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /optimization_challenge

# Copy the current directory contents into the container
COPY . /optimization_challenge

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends -y \
    build-essential \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev && \
    python3.12 -m ensurepip && \
    python3.12 -m venv /venv && \
    /venv/bin/pip install --disable-pip-version-check --timeout=120 --upgrade pip setuptools wheel && \
    /venv/bin/pip install --timeout=180 --retries 3 --no-cache-dir -r ./requirements.txt  --extra-index-url https://download.pytorch.org/whl/cpu && \
    /venv/bin/pip cache purge && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

# Run the application
CMD ["/venv/bin/python", "solution_script.py"]
