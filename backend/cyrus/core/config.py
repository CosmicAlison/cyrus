import os
from functools import lru_cache


class Settings:
    # Database
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL", "postgresql://cyrus:cyrus@postgres-1:5432/cyrus"
    )

    # RabbitMQ
    RABBITMQ_URL: str = os.environ.get("RABBITMQ_URL")

    # Redis
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://:cyrus@redis-1:6379/0")

    # LLM
    FIREWORKS_KEY: str = os.environ.get("FIREWORKS_KEY")
    FIREWORKS_URL: str | None = os.environ.get("FIREWORKS_URL") 
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

    # Flask
    FLASK_ENV: str = os.environ.get("FLASK_ENV", "production")

    # Queue names
    QUEUE_SURYA_JOBS: str = "cyrus.surya_jobs"
    QUEUE_RAW_FORECAST: str = "cyrus.raw_forecast"
    QUEUE_THREATS: str = "cyrus.threats"
    QUEUE_SATOPS: str = "cyrus.satops"
    QUEUE_GRIDOPS: str = "cyrus.gridops"
    QUEUE_COMMSOPS: str = "cyrus.commsops"
    QUEUE_AGENT_REPORTS: str = "cyrus.agent_reports"

    # Redis pub/sub channel for dashboard SSE
    REDIS_DASHBOARD_CHANNEL: str = "cyrus:dashboard"

    # Severity thresholds (flare probability 0–1)
    # Controls which agents are activated
    THRESHOLD_COMMSOPS: float = float(os.environ.get("THRESHOLD_COMMSOPS", "0.25"))
    THRESHOLD_SATOPS: float = float(os.environ.get("THRESHOLD_SATOPS", "0.45"))
    THRESHOLD_GRIDOPS: float = float(os.environ.get("THRESHOLD_GRIDOPS", "0.65"))

    # Worker retry settings
    RABBITMQ_CONNECT_RETRIES: int = int(os.environ.get("RABBITMQ_CONNECT_RETRIES", "10"))
    RABBITMQ_CONNECT_DELAY: int = int(os.environ.get("RABBITMQ_CONNECT_DELAY", "5"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()