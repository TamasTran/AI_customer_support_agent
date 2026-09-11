from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# Exposed so main.py's startup check can refuse to boot with this value outside
# development, rather than silently letting confirmation tokens become forgeable.
DEFAULT_APP_SECRET_KEY = "dev-only-insecure-secret-change-me"


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
    # can only execute with the exact arguments the customer was shown. Not used for
    # anything else — override via APP_SECRET_KEY in production deployments.
    app_secret_key: str = DEFAULT_APP_SECRET_KEY


settings = Settings()
