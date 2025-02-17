from sqlalchemy import Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aana.core.models.api_service import ApiKey
from aana.storage.models.base import TimeStampEntity, timestamp


class ApiServiceBase(DeclarativeBase):
    """Base class."""

    pass


class ApiKeyEntity(ApiServiceBase, TimeStampEntity):
    """Table for API keys."""

    __tablename__ = "api_keys"
    __entity_name__ = "api key"

    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    key_id: Mapped[str] = mapped_column(
        nullable=False, unique=True, comment="The API key id in api gateway"
    )
    user_id: Mapped[str] = mapped_column(
        nullable=False, index=True, comment="ID of the user who owns this API key"
    )
    api_key: Mapped[str] = mapped_column(
        nullable=False, index=True, unique=True, comment="The API key"
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Whether the user is an admin"
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
    expired_at: Mapped[timestamp] = mapped_column(
        nullable=False, comment="The expiration date of the API key"
    )

    def __repr__(self) -> str:
        """String representation of the API key."""
        return (
            f"<APIKeyEntity(id={self.id}, "
            f"key_id={self.key_id}, "
            f"user_id={self.user_id}, "
            f"api_key={self.api_key}, "
            f"is_admin={self.is_admin}, "
            f"subscription_id={self.subscription_id}, "
            f"hmac_secret={self.hmac_secret}, "
            f"is_subscription_active={self.is_subscription_active}), "
            f"expired_at={self.expired_at}, "
            f"created_at={self.created_at}, "
            f"updated_at={self.updated_at}>"
        )

    def to_model(self) -> ApiKey:
        """Convert the object to a dictionary."""
        return ApiKey(
            api_key=self.api_key,
            user_id=self.user_id,
            is_admin=self.is_admin,
            subscription_id=self.subscription_id,
            is_subscription_active=self.is_subscription_active,
            hmac_secret=self.hmac_secret,
        )
