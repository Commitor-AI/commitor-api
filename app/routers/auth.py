import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user_from_jwt
from app.models.api_key import ApiKey
from app.models.user import User
from app.rate_limit import login_limiter, rate_limit_dependency, signup_limiter
from app.schemas.api_key import ApiKeyCreate, ApiKeyCreated, ApiKeyRead
from app.schemas.auth import Token, UserCreate
from app.security.api_keys import generate_api_key, hash_api_key
from app.security.jwt_handler import create_access_token
from app.security.passwords import hash_password, verify_password

router = APIRouter()


@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    response_model=Token,
    dependencies=[Depends(rate_limit_dependency(signup_limiter))],
)
async def signup(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> Token:
    email = payload.email.strip().lower()
    existing_user = await db.scalar(select(User).where(User.email == email))
    if existing_user is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = User(email=email, hashed_password=hash_password(payload.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    return Token(access_token=create_access_token(user.id, user.plan))


@router.post(
    "/login",
    response_model=Token,
    dependencies=[Depends(rate_limit_dependency(login_limiter))],
)
async def login(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> Token:
    email = payload.email.strip().lower()
    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    return Token(access_token=create_access_token(user.id, user.plan))


@router.post("/api-keys", status_code=status.HTTP_201_CREATED, response_model=ApiKeyCreated)
async def create_api_key(
    payload: ApiKeyCreate,
    user: User = Depends(get_current_user_from_jwt),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyCreated:
    raw_key = generate_api_key()
    api_key = ApiKey(user_id=user.id, key_hash=hash_api_key(raw_key), label=payload.label.strip())
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    return ApiKeyCreated(id=api_key.id, label=api_key.label, api_key=raw_key, created_at=api_key.created_at)


@router.get("/api-keys", response_model=list[ApiKeyRead])
async def list_api_keys(
    user: User = Depends(get_current_user_from_jwt),
    db: AsyncSession = Depends(get_db),
) -> list[ApiKeyRead]:
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    return [ApiKeyRead.model_validate(key) for key in result.scalars()]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: uuid.UUID,
    user: User = Depends(get_current_user_from_jwt),
    db: AsyncSession = Depends(get_db),
) -> Response:
    api_key = await db.scalar(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id))
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    if api_key.revoked_at is None:
        api_key.revoked_at = datetime.now(timezone.utc)
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
