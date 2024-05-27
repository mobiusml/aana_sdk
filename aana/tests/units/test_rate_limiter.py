import sys

import pytest

from aana.api.event_handlers.event_manager import EventManager
from aana.api.event_handlers.rate_limit_handler import RateLimitHandler
from aana.exceptions.runtime import TooManyRequestsException


def test_rate_limiter_single():
    """Tests that the rate limiter raises if the rate limit is exceeded."""
    event_manager = EventManager()
    rate_limiter = RateLimitHandler(1, 1.0)

    event_manager.register_handler_for_events(rate_limiter, ["foo"])
    event_manager.handle("foo")
    with pytest.raises(TooManyRequestsException):
        event_manager.handle("foo")


def test_rate_limiter_multiple():
    """Tests that the rate limiter raises if the rate limit is exceeded."""
    event_manager = EventManager()
    rate_limiter = RateLimitHandler(1, 1.0)

    event_manager.register_handler_for_events(rate_limiter, ["foo", "bar"])
    event_manager.handle("foo")
    with pytest.raises(TooManyRequestsException):
        event_manager.handle("bar")


def test_rate_limiter_noraise():
    """Tests that the rate limiter raises if the rate limit is exceeded."""
    event_manager = EventManager()
    # Smallest possible value such that 1+x>1, should never run into rate limit
    rate_limiter = RateLimitHandler(1, sys.float_info.epsilon)
    event_manager.register_handler_for_events(rate_limiter, ["foo"])
    for _ in range(10):
        event_manager.handle("foo")


def test_event_manager_discriminates():
    """Tests that the rate limiter raises if the rate limit is exceeded."""
    event_manager = EventManager()
    rate_limiter = RateLimitHandler(1, sys.float_info.max)
    event_manager.register_handler_for_events(rate_limiter, ["bar"])
    # Should not raise
    event_manager.handle("foo")
