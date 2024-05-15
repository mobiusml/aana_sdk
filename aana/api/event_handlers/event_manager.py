from collections import defaultdict
from collections.abc import MutableMapping

from aana.api.event_handlers.event_handler import EventHandler
from aana.exceptions.runtime import (
    HandlerNotRegisteredException,
)


class EventManager:
    """Class for event manager. Not guaranteed to be thread safe."""

    _handlers: MutableMapping[str, EventHandler]

    def __init__(self):
        """Constructor."""
        self._handlers = defaultdict(list)

    def handle(self, event_name: str, *args, **kwargs):
        """Trigger event handlers for `event_name`.

        Arguments:
            event_name (str): name of event
            *args (list): specific args
            **kwargs (dict): specific args
        """
        for handler in self._handlers[event_name]:
            handler.handle(event_name, *args, **kwargs)

    def register_handler_for_events(
        self, handler: EventHandler, event_names: list[str]
    ):
        """Adds a handler to the event handler list.

        Arguments:
            handler (EventHandler): the handler to deregister
            event_names (list[str]): the events from which this handler is to be deregistered
        """
        for event_name in event_names:
            if handler not in self._handlers[event_name]:
                self._handlers[event_name].append(handler)

    def deregister_handler_from_event(self, handler: EventHandler, event_name: str):
        """Removes a handler from the event handler list.

        Arguments:
            handler (EventHandler): the handler to remove
            event_name (str): the name of the event from which the handler should be removed
        Raises:
            HandlerNotRegisteredException: if the handler isn't registered. (embed in try-except to suppress)
        """
        try:
            self._handlers[event_name].remove(handler)
        except ValueError as e:
            raise HandlerNotRegisteredException() from e

    def deregister_handler_from_all_events(self, handler: EventHandler):
        """Removes a handler from all event handlers.

        Arguments:
            handler (EventHandler): the exact instance of the handler to remove.

        Raises:
            HandlerNotRegisteredException: if the handler isn't registered. (embed in try-except to suppress)
        """
        has_removed = False
        for handler_list in self._handlers.values():
            if handler in handler_list:
                handler_list.remove(handler)
                has_removed = True
        if not has_removed:
            raise HandlerNotRegisteredException()
