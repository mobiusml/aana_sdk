from sqlalchemy import Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ApiServiceBase(DeclarativeBase):
    """Base class."""

    pass


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
    subscription_id: Mapped[str] = mapped_column(
        nullable=False, comment="ID of the associated subscription"
    )
    is_subscription_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the subscription is active (credits are available)",
    )

    def __repr__(self) -> str:
        """String representation of the API key."""
        return (
            f"<APIKeyEntity(id={self.id}, "
            f"user_id={self.user_id}, "
            f"subscription_id={self.subscription_id}, "
            f"is_subscription_active={self.is_subscription_active})>"
        )

    def to_dict(self) -> dict:
        """Convert the object to a dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
