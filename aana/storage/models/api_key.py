from sqlalchemy import Boolean
from sqlalchemy.orm import Mapped, mapped_column

from aana.storage.models.base import BaseEntity, TimeStampEntity


class ApiKeyEntity(BaseEntity, TimeStampEntity):
    """Table for API keys."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    api_key: Mapped[str] = mapped_column(
        nullable=False, unique=True, comment="The API key"
    )
    user_id: Mapped[str] = mapped_column(
        nullable=False, comment="ID of the user who owns this API key"
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
            f"user_id={self.user_id}, "
            f"subscription_id={self.subscription_id}, "
            f"is_subscription_active={self.is_subscription_active}, "
            f"updated_at={self.updated_at})>"
        )

    def to_dict(self) -> dict:
        """Convert the object to a dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
