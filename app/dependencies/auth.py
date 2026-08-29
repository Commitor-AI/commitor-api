import uuid
from datetime import datetime, timezone

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.security.api_keys import hash_api_key
from app.security.jwt_handler import decode_access_token

_bearer_scheme = HTTPBearer(auto_error=False)
_WWW_AUTHENTICATE = {"WWW-Authenticate": "Bearer"}


def _require_bearer_token(credentials: HTTPAuthorizationCredentials | None, kind: str) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Not authenticated: missing bearer {kind}",
            headers=_WWW_AUTHENTICATE,
        )
    return credentials.credentials


async def get_current_user_from_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = _require_bearer_token(credentials, "session token")
    try:
        payload = decode_access_token(token)
    except pyjwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers=_WWW_AUTHENTICATE,
        )
    try:
        user_id = uuid.UUID(str(payload.get("sub")))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers=_WWW_AUTHENTICATE,
        )
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers=_WWW_AUTHENTICATE,
        )
    return user


async def get_current_user_from_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    raw_key = _require_bearer_token(credentials, "API key")
    result = await db.execute(select(ApiKey).where(ApiKey.key_hash == hash_api_key(raw_key)))
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers=_WWW_AUTHENTICATE,
        )
    if api_key.revoked_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
            headers=_WWW_AUTHENTICATE,
        )
    api_key.last_used_at = datetime.now(timezone.utc)
    await db.commit()
    user = await db.get(User, api_key.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers=_WWW_AUTHENTICATE,
        )
    return user


async def get_current_user_either(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Accept either a CLI API key or a web session JWT, so the same
    `/auth/me` account summary backs both the CLI (`whoami`) and the web
    dashboard. API key wins (it's the CLI path); on failure we fall
    through to the session JWT."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: missing bearer token",
            headers=_WWW_AUTHENTICATE,
        )
    try:
        return await get_current_user_from_api_key(credentials, db)
    except HTTPException:
        pass
    try:
        return await get_current_user_from_jwt(credentials, db)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers=_WWW_AUTHENTICATE,
        )
