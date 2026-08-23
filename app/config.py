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


@lru_cache
def get_settings() -> Settings:
    return Settings()
