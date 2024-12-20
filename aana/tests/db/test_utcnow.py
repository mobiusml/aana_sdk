# ruff: noqa: S101
from datetime import datetime, timedelta, timezone

from aana.storage.utcnow import utcnow


def test_utcnow(db_session):
    """Tests the utcnow() function."""
    current_time_utc = datetime.now(tz=timezone.utc)
    result = db_session.execute(utcnow()).scalar()
    result = result.replace(tzinfo=timezone.utc)  # Make result offset-aware
    assert isinstance(result, datetime)
    assert current_time_utc - result < timedelta(seconds=1)
