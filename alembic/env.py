import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import dotenv_values
from sqlalchemy import create_engine, pool

from app.database import Base
from app.models import ApiKey, User

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    env_path = Path.cwd() / ".env"
    values = dotenv_values(env_path) if env_path.exists() else {}
    url = os.getenv("DATABASE_URL") or values.get("DATABASE_URL") or "sqlite+aiosqlite:///./commitor.db"
    url = url.replace("+asyncpg", "+psycopg2").replace("+aiosqlite", "")
    # asyncpg uses `ssl=`, but psycopg2/sync driver expects `sslmode=`
    return url.replace("ssl=", "sslmode=")


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = database_url()
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=url.startswith("sqlite"),
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
