FROM python:3.11-slim

WORKDIR /app

# System deps for NetCDF4 / HDF5 and git clone
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libnetcdf-dev \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv — Surya's own package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone the NASA-IMPACT Surya repo into /surya
# This gives us the `surya` Python package (the custom architecture code)
# and the easy_inference scripts
RUN git clone --depth 1 https://github.com/NASA-IMPACT/Surya.git /surya

# Install Surya and all its dependencies using its own lockfile
WORKDIR /surya
RUN uv sync --frozen

# Install our extra deps (pika, etc.) into the same venv
RUN uv pip install pika python-dotenv

# Copy our service files into /app
WORKDIR /app
COPY . .

# Put the Surya venv on PATH and add /surya to PYTHONPATH
# so `from surya.model import ...` and `from surya.infer import ...` work
ENV VIRTUAL_ENV=/surya/.venv
ENV PATH="/surya/.venv/bin:$PATH"
ENV PYTHONPATH="/surya:$PYTHONPATH"

CMD ["python", "runner.py"]
