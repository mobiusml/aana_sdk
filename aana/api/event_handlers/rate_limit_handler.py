from time import monotonic

from typing_extensions import override

from aana.api.event_handlers.event_handler import EventHandler
from aana.exceptions.runtime import TooManyRequestsException


class RateLimitHandler(EventHandler):
    """Event handler that raises TooManyRequestsException if the rate limit is exceeded.

    Attributes:
        capacity (int): number of resources (requests) per interval
        rate (float): the interval for the limit in seconds
    """

    capacity: int
    interval: float

    _calls: list

    def __init__(self, capacity: int, interval: float):
        """Constructor."""
        self.capacity = capacity
        self.interval = interval
        self._calls = []

    def _clear_expired(self, expired: float):
        """Removes expired items from list of resources.

        Arguments:
            expired: timestamp before which to clear, as output from time.monotonic()
        """
        while self._calls and self._calls[0] < expired:
            self._calls.pop(0)

    def _acquire(self):
        """Checks if we can acquire (process) a resource.

        Raises:
            TooManyRequestsException: if the rate limit has been reached
        """
        now = monotonic()
        expired = now - self.interval
        self._clear_expired(expired)
        if len(self._calls) < self.capacity:
            self._calls.append(now)
        else:
            raise TooManyRequestsException(self.capacity, self.interval)

    @override
    def handle(self, event_name: str, *args, **kwargs):
        """Handle the event by checking against rate limiting parameters.

        Arguments:
            event_name (str): the name of the event to handle
            *args (list): args for the event
            **kwargs (dict): keyword args for the event

        Raises:
            TooManyRequestsException: if the rate limit has been reached
        """
        # if the endpoint execution is deferred, we don't want to rate limit it
        defer = kwargs.get("defer", False)
        if not defer:
            self._acquire()
