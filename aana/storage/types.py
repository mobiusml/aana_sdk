from typing import TypeAlias

from sqlalchemy import String, TypeDecorator

from aana.core.models.media import MediaId


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
