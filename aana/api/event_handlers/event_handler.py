from abc import ABC, abstractmethod


class EventHandler(ABC):
    """Base class for event handlers. Not guaranteed to be thread safe."""

    @abstractmethod
    def handle(self, event_name: str, *args, **kwargs):
        """Handles an event of the given name.

        Arguments:
            event_name (str): name of the event to handle
            *args (list): specific, context-dependent args
            **kwargs (dict): specific, context-dependent args
        """
        pass
