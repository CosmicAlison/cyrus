# Dockerfile (local dev / CPU / mock mode)
# Lightweight image for local development with SURYA_USE_MOCK=true.
# No GPU required. Clones Surya repo for import compatibility but
# does NOT download large datasets (not needed in mock mode).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libnetcdf-dev \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone Surya (needed so `from surya.utils import ...` works even in mock mode)
RUN git clone --depth 1 https://github.com/NASA-IMPACT/Surya.git /surya
WORKDIR /surya
RUN uv sync --frozen
RUN uv pip install pika pyyaml scipy netCDF4 pandas

ENV VIRTUAL_ENV=/surya/.venv
ENV PATH="/surya/.venv/bin:$PATH"
ENV PYTHONPATH="/surya:$PYTHONPATH"
ENV SURYA_USE_MOCK=true

WORKDIR /app
COPY . .

CMD ["python", "runner.py"]