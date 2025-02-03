from typing import TypedDict

from sqlalchemy import Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ApiServiceBase(DeclarativeBase):
    """Base class."""

    pass


class ApiKeyInfo(TypedDict):
    """API key info.

    Attributes:
        user_id (str): ID of the user who owns this API key.
        is_admin (bool): Whether the user is an admin.
        subscription_id (str): ID of the associated subscription.
        is_subscription_active (bool): Whether the subscription is active (credits are available).
        hmac_secret (str | None): The secret key for HMAC signature generation.
    """

    user_id: str
    is_admin: bool
    subscription_id: str
    is_subscription_active: bool
    hmac_secret: str | None


class ApiKeyEntity(ApiServiceBase):
    """Table for API keys."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    api_key: Mapped[str] = mapped_column(
        nullable=False, unique=True, comment="The API key"
    )
    user_id: Mapped[str] = mapped_column(
        nullable=False, comment="ID of the user who owns this API key"
    )
    is_admin: Mapped[bool] = mapped_column(
        nullable=False, default=False, comment="Whether the user is an admin"
    )
    subscription_id: Mapped[str] = mapped_column(
        nullable=False, comment="ID of the associated subscription"
    )
    is_subscription_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the subscription is active (credits are available)",
    )
    hmac_secret: Mapped[str] = mapped_column(
        nullable=True, comment="The secret key for HMAC signature generation"
    )

    def __repr__(self) -> str:
        """String representation of the API key."""
        return (
            f"<APIKeyEntity(id={self.id}, "
            f"api_key={self.api_key}, "
            f"user_id={self.user_id}, "
            f"is_admin={self.is_admin}, "
            f"subscription_id={self.subscription_id}, "
            f"is_subscription_active={self.is_subscription_active})>"
        )

    def to_dict(self) -> ApiKeyInfo:
        """Convert the object to a dictionary."""
        return ApiKeyInfo(
            user_id=self.user_id,
            is_admin=self.is_admin,
            subscription_id=self.subscription_id,
            is_subscription_active=self.is_subscription_active,
            hmac_secret=self.hmac_secret,
        )
