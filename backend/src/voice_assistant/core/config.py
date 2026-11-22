from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    whisper_cpu_threads: int = 4

    piper_model_path: Path

    conversation_max_turns: int = 5
    conversation_ttl_seconds: int = 600

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_timeout: float = 30.0
    deepseek_max_retries: int = 2

    assistant_city: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
