FROM python:3.11-slim

WORKDIR /app

# System deps for NetCDF4 / HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libnetcdf-dev \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (used by Surya's own tooling)
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "runner.py"]