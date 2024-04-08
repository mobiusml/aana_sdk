from aana.api.event_handlers.event_handler import EventHandler
from aana.exceptions.general import (
    HandlerAlreadyRegisteredException,
    HandlerNotRegisteredException,
)


class EventManager:
    """Class for event manager. Not guaranteed to be threadsafe."""

    def __init__(self):
        """Constructor."""
        # TODO: 
        self._handlers = list[EventHandler]()

    def handle(self, event_name: str, *args, **kwargs):
        """Trigger event handlers for `event_name`.

        Arguments:
            event_name (str): name of event
            *args (list): specific args
            **kwargs (dict): specific args
        """
        for handler in self._handlers:
            handler.handle(event_name, *args, **kwargs)

    def register_handler(self, handler: EventHandler):
        """Adds a handler to the event handler list.

        Arguments:
            handler (EventHandler): the handler to register
        """
        if handler in self._handlers:
            raise HandlerAlreadyRegisteredException()
        self._handlers.append(handler)

    def deregister_handler(self, handler: EventHandler):
        """Removes a handler from the event handler list.

        Arguments:
            handler (EventHandler): the handler to remove
        Raises:
            ValueError: if the handler isn't reqistered. (embed in try-except to suppress)
        """
        try:
            self._handlers.remove(handler)
        except ValueError as e:
            raise HandlerNotRegisteredException() from e
