from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.api_key import ApiKey
from app.models.user import User
from app.security.api_keys import hash_api_key
from app.security.jwt_handler import create_access_token
from app.security.passwords import hash_password


@pytest_asyncio.fixture
async def db_sessionmaker():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db

    yield factory

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_sessionmaker):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


async def _seed_user(
    factory, *, logins=None, email="owner@example.com"
) -> tuple[str, str]:
    """Create a user + API key (+ session JWT). Returns (api_key, jwt)."""
    async with factory() as session:
        user = User(
            email=email,
            hashed_password=hash_password("password123"),
            github_logins=list(logins or []),
        )
        session.add(user)
        await session.flush()
        api_key = ApiKey(
            user_id=user.id, key_hash=hash_api_key("bot-key"), label="bot"
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(user)
    jwt = create_access_token(user.id, user.plan)
    return "bot-key", jwt


@pytest.mark.asyncio
async def test_authorize_allows_listed_login(client, db_sessionmaker):
    await _seed_user(db_sessionmaker, logins=["Smasduq", "Alice"])
    resp = await client.post(
        "/auth/github/authorize",
        json={"login": "smasduq"},
        headers={"Authorization": "Bearer bot-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["authorized"] is True
    assert body["email"] == "owner@example.com"
    assert body["plan"] == "free"


@pytest.mark.asyncio
async def test_authorize_denies_unlisted_login(client, db_sessionmaker):
    await _seed_user(db_sessionmaker, logins=["Alice"])
    resp = await client.post(
        "/auth/github/authorize",
        json={"login": "Bob"},
        headers={"Authorization": "Bearer bot-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["authorized"] is False


@pytest.mark.asyncio
async def test_authorize_denies_all_when_allowlist_empty(client, db_sessionmaker):
    await _seed_user(db_sessionmaker, logins=[])
    resp = await client.post(
        "/auth/github/authorize",
        json={"login": "Smasduq"},
        headers={"Authorization": "Bearer bot-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["authorized"] is False


@pytest.mark.asyncio
async def test_authorize_rejects_bad_api_key(client, db_sessionmaker):
    await _seed_user(db_sessionmaker, logins=["Smasduq"])
    resp = await client.post(
        "/auth/github/authorize",
        json={"login": "Smasduq"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_set_and_get_logins_roundtrip(client, db_sessionmaker):
    _, jwt = await _seed_user(db_sessionmaker)
    put = await client.put(
        "/auth/github/logins",
        json={"logins": ["  Smasduq  ", "bob", "", "BOB"]},
        headers={"Authorization": f"Bearer {jwt}"},
    )
    assert put.status_code == 200
    assert sorted(put.json()["logins"]) == ["bob", "smasduq"]

    get = await client.get(
        "/auth/github/logins",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    assert get.status_code == 200
    assert sorted(get.json()["logins"]) == ["bob", "smasduq"]
