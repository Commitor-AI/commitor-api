import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import Plan


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    plan: Plan
    created_at: datetime


class Me(BaseModel):
    """Account summary behind a presented credential (`GET /auth/me`)."""

    email: EmailStr
    plan: Plan
    admin: bool = Field(description="Whether the backend verifies this account as an admin.")


class GithubAuthorizeRequest(BaseModel):
    """A GitHub sender login the Commitor bot wants to run analysis for."""

    login: str = Field(min_length=1, max_length=100)


class GithubAuthorizeResponse(BaseModel):
    """Whether a GitHub sender may drive the app under this account's key."""

    authorized: bool
    email: EmailStr
    plan: Plan


class GithubLoginsUpdate(BaseModel):
    """Replace the account's GitHub-app allowlist (lowercase normalized)."""

    logins: list[str] = Field(default_factory=list)


class GithubLoginsRead(BaseModel):
    """The account's current GitHub-app allowlist."""

    logins: list[str]
