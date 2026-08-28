from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./commitor.db"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    login_rate_limit: int = 10
    signup_rate_limit: int = 5
    rate_limit_window_seconds: int = 300

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "stealth/ox-alpha"
    openrouter_reasoning_model: str = "stealth/ox-alpha"
    openrouter_timeout_seconds: float = 45.0
    openrouter_reasoning_timeout_seconds: float = 90.0
    # Wall-clock cap on the whole escalation chain per /analyze request;
    # optional turns (recheck, message rewrite) are skipped once passed.
    analyze_chain_deadline_seconds: float = 75.0
    analyze_escalation_files: int = 6
    analyze_escalation_diff_lines: int = 300
    analyze_confidence_threshold: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
