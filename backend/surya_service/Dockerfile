FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install uv

RUN git clone https://github.com/NASA-IMPACT/Surya.git /opt/surya

WORKDIR /opt/surya

RUN uv sync

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/opt/surya:/app