import os
from typing import Optional


class TracingConfig:
    """Tracing configuration from environment variables"""

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("TRACING_ENABLED", "false").lower() == "true"

    @staticmethod
    def get_url() -> str:
        return os.getenv("TRACING_URL", "")

    @staticmethod
    def get_project() -> str:
        return os.getenv("TRACING_PROJECT", "default")

    @staticmethod
    def get_api_key() -> Optional[str]:
        return os.getenv("TRACING_API_KEY")

    @staticmethod
    def get_batch_size() -> int:
        return int(os.getenv("TRACING_BATCH_SIZE", "20"))

    @staticmethod
    def get_flush_interval() -> float:
        return float(os.getenv("TRACING_FLUSH_INTERVAL", "5.0"))
