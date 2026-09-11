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

    # Signs PendingConfirmation tokens (see app/security.py) so a mutating tool call
    # can only execute with the exact arguments the customer was shown. No usable
    # default on purpose — a previous version defaulted to a fixed, publicly-known
    # string and only failed if a self-reported APP_ENV said non-development, which
    # is itself unset-by-default and easy to leave unset in a real deployment. Empty
    # is mandatory everywhere, matching how OLLAMA_CHAT_MODEL already hard-fails on
    # empty regardless of environment (see OllamaProvider.__init__).
    app_secret_key: str = ""


settings = Settings()

if not settings.app_secret_key:
    raise ValueError(
        "APP_SECRET_KEY must be configured (see .env.example) — it signs the "
        "confirmation tokens that gate refunds/cancellations. Set it to a random "
        "value; there is no usable default."
    )
