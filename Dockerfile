# 1. Minimalistic x86-64 Debian base
FROM debian:bookworm-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Use bash with pipefail enabled for RUN commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 2. Install system deps: curl, Python, tesseract, language data
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    ca-certificates \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy code into container
WORKDIR /app
COPY . .

# 4. Install uv and immediately setup venv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="$HOME/.local/bin:$PATH" && \
    uv venv && \
    . .venv/bin/activate && \
    uv sync --frozen && \
    ./download_models.sh
