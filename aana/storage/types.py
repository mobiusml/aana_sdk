from enum import Enum
from typing import TypeAlias

from sqlalchemy import String, TypeDecorator

from aana.api.models.media_id import MediaId


class DbType(str, Enum):
    """Engine types for relational database."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"

class MediaIdType(TypeDecorator):
    """Custom type for handling MediaId objects with SQLAlchemy."""

    impl = String

    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert a MediaId instance to a string value for storage."""
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        """Convert a string value from the database back into a MediaId instance."""
        if value is None:
            return value
        return MediaId(value)


MediaIdSqlType: TypeAlias = MediaIdType
