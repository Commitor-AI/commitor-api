import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, JSON, String, false, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.api_key import ApiKey


class Plan(str, enum.Enum):
    free = "free"
    pro = "pro"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    plan: Mapped[Plan] = mapped_column(Enum(Plan, name="plan"), default=Plan.free, server_default=Plan.free.value)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Logins this account allows to drive the Commitor GitHub App under
    # its quota (lowercase GitHub usernames). Empty = no one except the
    # account owner may use the app through this key.
    github_logins: Mapped[list[str]] = mapped_column(JSON, default=list, server_default="[]")

    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def allows_github_login(self, login: str) -> bool:
        wanted = {g.strip().lower() for g in (self.github_logins or []) if g.strip()}
        return login.strip().lower() in wanted
