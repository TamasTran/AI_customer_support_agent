from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = ""
    ollama_embedding_model: str = ""
    ollama_timeout_seconds: int = 120

    # Determined empirically from OLLAMA_EMBEDDING_MODEL (see scripts/check_embedding_dimension.py);
    # not derived automatically at import time since that would require a live Ollama call.
    embedding_dimension: int = 768

    database_url: str = ""
    redis_url: str = ""

    log_level: str = "INFO"


settings = Settings()
