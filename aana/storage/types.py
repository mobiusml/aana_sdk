import datetime
from typing import TypeAlias

from sqlalchemy import String
from sqlalchemy.types import DateTime, TypeDecorator

MediaIdSqlType: TypeAlias = String(36)


class TimezoneAwareDateTime(TypeDecorator):
    """A custom SQLAlchemy type decorator for timezone-aware datetime objects.

    This implementation converts naive datetime objects to UTC-aware datetime
    objects when binding parameters and when loading from the database.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert naive datetime objects to UTC-aware datetime before storing."""
        if value is None:
            return value
        # If the datetime is naive, assume it is UTC and set the tzinfo.
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        else:
            # Otherwise, normalize to UTC.
            value = value.astimezone(datetime.timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        """Ensure that datetime objects loaded from the database are UTC-aware."""
        if value is None:
            return value
        # If value is naive (as may happen with SQLite), assume it's in UTC.
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value
